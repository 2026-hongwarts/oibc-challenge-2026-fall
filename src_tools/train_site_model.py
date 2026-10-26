"""발전소 단위 LightGBM 학습·예측. 옵션을 안 주면 E4-00(baseline.py 재현).

    python src_tools/train_site_model.py --base champion                 기준 모델을 내 PC에서 다시 만들기
    python src_tools/train_site_model.py --exp E2-02x2 --base champion --add features_weather:add_ceiling
                                                                         기준 모델 + 내 변경 1개 (보통 이렇게 쓴다)
    python src_tools/train_site_model.py --base champion --mode test     기준 모델로 실제 평가 기간 예측 → outputs/preds_test/
    python src_tools/train_site_model.py --exp E3-01 --drop-dead         옵션만 직접 주기 (기본값은 E4-00)

--base 이름|champion: outputs/preds/이름.json 의 설정을 불러오고, 명령줄에 준 옵션으로 덮어쓴다.
    --add 는 기존 함수 뒤에 이어 붙는다. --exp 를 빼면 같은 이름으로 다시 만든다.

시드 하나만 바꿔도 가상 평가연도 벌금이 약 0.5% 움직인다(E4-00 seed 0 vs 1: +0.51%).
그래서 채택 여부는 --seeds 2 결과를 기준 모델(역시 시드 2개)과 비교해 정한다.

E4-00 = baseline.py 와 같은 설정
    round 10 모델과 round 17 모델 예측의 평균 · 컷오프 전 2년 · 태양고도 > 0 행 · 정지일 포함
    타깃 CF(0~1.2) · L2 · n_estimators 600, learning_rate 0.06, num_leaves 127, min_child_samples 200,
    subsample 0.8, colsample_bytree 0.8 · 밤(태양고도 ≤ 0) 예측은 0
    차이: 한 라운드만 빠진 예보 행을 다른 라운드로 채웠고(build_base), random_state=0 으로 고정.

--add 로 넘기는 함수 규칙 (모듈:함수, 여러 번 가능)
    f(df) -> df. df 에는 학습 행(is_fit=True, 컷오프 전 학습 기간의 모든 시각)과 예측 행(is_fit=False)이 함께 들어온다.
    새로 생긴 열은 자동으로 피처에 추가된다. 행 수와 순서는 그대로 둔다(df 는 발전소별로 시각순).
    발전량(energy, cf)으로 만드는 값은 df[df.is_fit] 에서 계산한다. 예측 행의 energy·cf 는 누설을 막으려고
    NaN 으로 지워서 넘기고, is_dead 도 False 로 둔다.

    # 예: features_weather.py
    def add_weather_features(df):
        df["wx_cloud_gap"] = df.wa_cloud_r17 - df.wb_cloud_r17
        return df
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from oibc_common import PREDS, PREDS_TEST, ROOT, TEST_FROM, VALID_FROM, WORK, k_weight

WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press"]
WB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq",
           "wb_vis", "wb_wind_speed", "wb_dew_point"]
COMMON = ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "capacity_kw", "lat", "lon"]
PARAMS = dict(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
OBJECTIVE = {"l2": "regression", "l1": "regression_l1"}
DEFAULTS = dict(rounds="avg", years=2, drop_dead=False, weight="none", objective="l2",
                add=[], params={}, seed=0, seeds=1)

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def round_feats(r: int) -> list[str]:
    return [f"{c}_r{r}" for c in WA_COLS + WB_COLS]


def model_feature_sets(rounds: str) -> list[list[str]]:
    """모델마다 쓸 예보 열. avg 는 모델 2개(10, 17)를 학습해 예측을 평균한다."""
    if rounds == "avg":
        return [round_feats(10), round_feats(17)]
    if rounds == "both":
        return [round_feats(10) + round_feats(17)]
    return [round_feats(int(rounds))]


def load_frame(mode: str, years: float) -> pd.DataFrame:
    base = pd.read_parquet(WORK / "base.parquet")
    cut = VALID_FROM if mode == "valid" else TEST_FROM
    fit = base[base.ts < cut]
    if years:
        fit = fit[fit.ts >= cut - pd.DateOffset(months=int(round(years * 12)))]
    pred = base[base.split == "valid"] if mode == "valid" else pd.read_parquet(WORK / "base_test.parquet")
    del base
    df = pd.concat([fit.assign(is_fit=True), pred.assign(is_fit=False)], ignore_index=True)
    df["site_id"] = df.site_id.astype(str)
    return df


def apply_hooks(df: pd.DataFrame, specs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    added: list[str] = []
    for spec in specs:
        mod, fn = spec.split(":")
        before = list(df.columns)
        out = getattr(importlib.import_module(mod), fn)(df)
        if len(out) != len(df):
            raise ValueError(f"{spec}: 행 수가 바뀜 ({len(df)} → {len(out)})")
        df = out
        new = [c for c in df.columns if c not in before]
        added += new
        log(f"--add {spec}: 새 피처 {new}")
    return df, added


def load_base_config(name: str) -> tuple[str, dict]:
    """--base 로 지정한 실험의 설정. 'champion' 이면 outputs/preds/champion.txt 의 이름."""
    if name == "champion":
        name = (PREDS / "champion.txt").read_text(encoding="utf-8").strip()
    path = PREDS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"--base {name}: 설정 파일 {path.relative_to(ROOT)} 이 없음 (git pull 했는지 확인)")
    j = json.loads(path.read_text(encoding="utf-8"))
    cfg = {k: j[k] for k in DEFAULTS if k in j and k != "params"}
    if "params_override" in j:
        cfg["params"] = j["params_override"]
    else:   # 예전 형식: 전체 파라미터에서 기본값과 다른 것만 덮어쓰기로 본다
        cfg["params"] = {k: v for k, v in j.get("params", {}).items()
                         if k not in ("objective", "random_state") and PARAMS.get(k) != v}
    return name, cfg


def resolve_config(args: argparse.Namespace) -> tuple[str, dict, str | None]:
    """기본값 ← --base 설정 ← 명령줄 옵션 순서로 덮어쓴다. --add 는 이어 붙인다."""
    cfg = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
           for k, v in DEFAULTS.items()}
    base_name = None
    if args.base:
        base_name, base_cfg = load_base_config(args.base)
        cfg.update(base_cfg)
    for k in ["rounds", "years", "drop_dead", "weight", "objective", "seed", "seeds"]:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.params is not None:
        cfg["params"] = {**cfg["params"], **json.loads(args.params)}
    cfg["add"] = list(dict.fromkeys(cfg["add"] + (args.add or [])))
    exp = args.exp or base_name
    if not exp:
        raise SystemExit("--exp 실험 번호를 주거나 --base 로 다시 만들 실험을 지정하세요")
    return exp, cfg, base_name


def main() -> None:
    ap = argparse.ArgumentParser(description="발전소 단위 LightGBM 학습·예측 (기본값 = E4-00)")
    ap.add_argument("--exp", default=None, help="실험 번호 (예: E2-02x2). 결과 파일 이름이 됨")
    ap.add_argument("--base", default=None, metavar="이름|champion",
                    help="이 실험의 설정에서 출발 (보통 champion)")
    ap.add_argument("--mode", choices=["valid", "test"], default="valid",
                    help="valid: 2024-05 까지 학습 → 가상 평가연도 예측 / test: 2025-05 까지 학습 → 실제 평가 기간 예측")
    ap.add_argument("--rounds", choices=["avg", "both", "10", "17"], default=None, help="기본 avg")
    ap.add_argument("--years", type=float, default=None, help="컷오프 전 학습 기간(년). 0 이면 전체. 기본 2")
    ap.add_argument("--drop-dead", action=argparse.BooleanOptionalAction, default=None,
                    help="정지 추정일(is_dead)을 학습에서 뺌")
    ap.add_argument("--weight", choices=["none", "cap", "cap_k"], default=None,
                    help="표본 가중치: cap = 설비용량, cap_k = 설비용량 × k(11~15시 3). 기본 none")
    ap.add_argument("--objective", choices=list(OBJECTIVE), default=None, help="기본 l2")
    ap.add_argument("--add", action="append", default=None, metavar="모듈:함수", help="피처 함수")
    ap.add_argument("--params", default=None, help='LightGBM 파라미터 덮어쓰기 JSON. 예: \'{"num_leaves": 63}\'')
    ap.add_argument("--seed", type=int, default=None, help="첫 시드. 기본 0")
    ap.add_argument("--seeds", type=int, default=None,
                    help="시드 개수. 2 이상이면 seed, seed+1, … 모델의 예측을 평균 (채택 판단은 2로). 기본 1")
    args = ap.parse_args()
    exp, cfg, base_name = resolve_config(args)
    log(f"실험 {exp}" + (f" (base {base_name})" if base_name else "") + f": {cfg}")

    df = load_frame(args.mode, cfg["years"])
    df.loc[~df.is_fit, ["energy", "cf"]] = np.nan          # 누설 방지: 예측 행의 실적을 지운다
    df["is_dead"] = df.is_dead.eq(True) & df.is_fit
    log(f"mode={args.mode}: 학습 후보 {int(df.is_fit.sum()):,}행 / 예측 {int((~df.is_fit).sum()):,}행")
    df, added = apply_hooks(df, cfg["add"])

    tr = df[df.is_fit & (df.elev > 0)]
    if cfg["drop_dead"]:
        tr = tr[~tr.is_dead.astype(bool)]
    y = (tr.energy / tr.capacity_kw).clip(0, 1.2).astype("float32")
    w = {"none": None, "cap": tr.capacity_kw.to_numpy(),
         "cap_k": tr.capacity_kw.to_numpy() * k_weight(tr.ts)}[cfg["weight"]]
    pr = df[~df.is_fit]
    params = {**PARAMS, **cfg["params"], "objective": OBJECTIVE[cfg["objective"]],
              "random_state": cfg["seed"]}
    log(f"학습 {len(tr):,}행 ({tr.ts.min():%Y-%m-%d} ~ {tr.ts.max():%Y-%m-%d}), 예측 {len(pr):,}행")

    sets = model_feature_sets(cfg["rounds"])
    seeds = [cfg["seed"] + i for i in range(cfg["seeds"])]
    preds, feature_sets = [], []
    for cols in sets:
        feats = cols + COMMON + added
        feature_sets.append(feats)
        X_tr, X_pr = tr[feats].astype("float32"), pr[feats].astype("float32")
        for s in seeds:
            model = lgb.LGBMRegressor(**{**params, "random_state": s}).fit(X_tr, y, sample_weight=w)
            preds.append(model.predict(X_pr))
            log(f"모델 {len(preds)}/{len(sets) * len(seeds)} 학습 완료 (피처 {len(feats)}개, seed {s})")
    p = np.clip(np.mean(preds, axis=0), 0, None)
    p = np.where(pr.elev.to_numpy() > 0, p, 0.0) * pr.capacity_kw.to_numpy()
    out = pr[["site_id", "ts"]].reset_index(drop=True).assign(pred_kwh=p)

    out_dir = PREDS if args.mode == "valid" else PREDS_TEST
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp}.parquet"
    out.to_parquet(path, index=False)
    meta = {"exp": exp, "mode": args.mode, "base": base_name,
            **{k: v for k, v in cfg.items() if k != "params"}, "params_override": cfg["params"],
            "params": params, "features": feature_sets, "n_train": len(tr),
            "train_period": [str(tr.ts.min()), str(tr.ts.max())], "created": datetime.now().isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"저장 {path.relative_to(ROOT)} ({len(out):,}행)")

    if args.mode == "valid":
        from evaluate import report
        report(path)


if __name__ == "__main__":
    main()
