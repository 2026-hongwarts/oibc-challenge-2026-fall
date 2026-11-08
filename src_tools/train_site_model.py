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

from oibc_common import PREDS, PREDS_TEST, ROOT, TEST_FROM, TZ, VALID_FROM, WORK, k_weight

WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press"]
WB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq",
           "wb_vis", "wb_wind_speed", "wb_dew_point"]
COMMON = ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "capacity_kw", "lat", "lon"]
PARAMS = dict(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
OBJECTIVE = {"l2": "regression", "l1": "regression_l1"}
DEFAULTS = dict(model="lgb", rounds="avg", years=2, drop_dead=False, weight="none", objective="l2",
                add=[], drop=[], params={}, seed=0, seeds=1, target="cf", fit_where=None, cat=[])
# CatBoost·XGBoost 초기값. --params 로 덮어쓴다(모델마다 이름이 다름)
CAT_PARAMS = dict(iterations=1000, depth=8, learning_rate=0.1, l2_leaf_reg=3, thread_count=-1,
                  verbose=0, allow_writing_files=False)
XGB_PARAMS = dict(n_estimators=800, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=50, tree_method="hist", n_jobs=-1)


def cat_levels(col: str) -> list:
    """범주형 컬럼의 범주 목록을 데이터와 무관하게 고정한다(학습·예측의 범주 번호가 같도록)."""
    if col == "hour":
        return list(range(24))
    sites = pd.read_csv(ROOT / "01_participant_dataset" / "sites.csv")
    if col in ("site_id", "wa_station", "wb_station"):
        return sorted(sites[col].astype(str).unique())
    raise ValueError(f"--cat {col}: 범주 목록이 정해지지 않은 컬럼")


def to_X(d: pd.DataFrame, feats: list[str], cats: list[str]) -> pd.DataFrame:
    """숫자 피처는 float32, --cat 컬럼은 고정 범주의 pandas category 로."""
    X = d[[c for c in feats if c not in cats]].astype("float32")
    for c in cats:
        v = d[c].astype(str) if c != "hour" else d[c].astype(int)
        X[c] = pd.Categorical(v, categories=cat_levels(c))
    return X[feats]


def make_model(kind: str, objective: str, override: dict, seed: int, has_cat: bool = False):
    """kind: lgb | cat | xgb. objective: l1 | l2. override: 그 모델의 파라미터 덮어쓰기.
    has_cat: 범주형 입력이 있으면 XGBoost 는 enable_categorical 을 켠다(LightGBM 은 category dtype 을 자동 인식)."""
    if kind == "lgb":
        return lgb.LGBMRegressor(**{**PARAMS, **override, "objective": OBJECTIVE[objective], "random_state": seed})
    if kind == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**{**CAT_PARAMS, **override, "random_seed": seed,
                                    "loss_function": "MAE" if objective == "l1" else "RMSE"})
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**{**XGB_PARAMS, **override, "random_state": seed, "enable_categorical": has_cat,
                               "objective": "reg:absoluteerror" if objective == "l1" else "reg:squarederror"})
    raise ValueError(f"모르는 모델 {kind}")

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def clear_ref(d: pd.DataFrame) -> np.ndarray:
    """CSI 타깃의 기준: Haurwitz 청천 일사 / 1000 (하한 0.05)."""
    c = d.cosz.clip(lower=0.01).to_numpy()
    return np.maximum(1.098 * c * np.exp(-0.057 / c), 0.05).astype("float32")


def round_feats(r: int) -> list[str]:
    return [f"{c}_r{r}" for c in WA_COLS + WB_COLS]


def model_feature_sets(rounds: str) -> list[list[str]]:
    """모델마다 쓸 예보 열. avg 는 모델 2개(10, 17)를 학습해 예측을 평균한다."""
    if rounds == "avg":
        return [round_feats(10), round_feats(17)]
    if rounds == "both":
        return [round_feats(10) + round_feats(17)]
    return [round_feats(int(rounds))]


def cut_window(cut: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """--cut YYYY-MM-DD: 그날 01:00 부터 1년을 검증 구간으로 (다른 해 검증용)."""
    start = pd.Timestamp(f"{cut} 01:00", tz=TZ)
    return start, start + pd.DateOffset(years=1)


def load_frame(mode: str, years: float, cut: str | None = None) -> pd.DataFrame:
    base = pd.read_parquet(WORK / "base.parquet")
    if cut:
        start, end = cut_window(cut)
    else:
        start = VALID_FROM if mode == "valid" else TEST_FROM
    fit = base[base.ts < start]
    if years:
        fit = fit[fit.ts >= start - pd.DateOffset(months=int(round(years * 12)))]
    if cut:
        pred = base[(base.ts >= start) & (base.ts < end)]
    else:
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
    if args.model is not None and args.model != cfg.get("model", "lgb"):
        cfg["params"] = {}                     # 다른 모델로 바꾸면 이전 모델의 파라미터 덮어쓰기는 버린다
    for k in ["model", "rounds", "years", "drop_dead", "weight", "objective", "seed", "seeds", "target", "fit_where"]:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.params is not None:
        cfg["params"] = {**cfg["params"], **json.loads(args.params)}
    if args.reset_add:
        cfg["add"] = []
    cfg["add"] = list(dict.fromkeys(cfg["add"] + (args.add or [])))
    cfg["cat"] = list(dict.fromkeys(cfg.get("cat", []) + (args.cat or [])))
    if args.reset_drop:
        cfg["drop"] = []
    cfg["drop"] = list(dict.fromkeys(cfg.get("drop", []) + (args.drop or [])))
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
    ap.add_argument("--cut", default=None, metavar="YYYY-MM-DD",
                    help="다른 해 검증: 이 날짜 전까지로 학습하고 그날부터 1년을 채점 (예: 2023-06-01). 결과 이름 끝에 _cut연도")
    ap.add_argument("--rounds", choices=["avg", "both", "10", "17"], default=None, help="기본 avg")
    ap.add_argument("--years", type=float, default=None, help="컷오프 전 학습 기간(년). 0 이면 전체. 기본 2")
    ap.add_argument("--drop-dead", action=argparse.BooleanOptionalAction, default=None,
                    help="정지 추정일(is_dead)을 학습에서 뺌")
    ap.add_argument("--weight", choices=["none", "cap", "cap_k", "recent"], default=None,
                    help="표본 가중치: cap = 설비용량, cap_k = 설비용량 × k(11~15시 3). 기본 none")
    ap.add_argument("--objective", choices=list(OBJECTIVE), default=None, help="기본 l2")
    ap.add_argument("--model", choices=["lgb", "cat", "xgb"], default=None,
                    help="lgb(기본) | cat(CatBoost) | xgb(XGBoost). 다른 모델로 바꾸면 --params 덮어쓰기는 초기화")
    ap.add_argument("--add", action="append", default=None, metavar="모듈:함수", help="피처 함수")
    ap.add_argument("--reset-add", action="store_true", help="--base 의 피처 함수 목록을 비우고 --add 로만 채운다")
    ap.add_argument("--drop", action="append", default=None, metavar="변수",
                    help="입력에서 뺄 피처. 예보 변수는 이름만 주면 _r10·_r17 둘 다 뺀다 (예: wa_ground_press)")
    ap.add_argument("--reset-drop", action="store_true", help="--base 의 뺄 피처 목록을 비우고 --drop 으로만 채운다")
    ap.add_argument("--cat", action="append", default=None, metavar="컬럼",
                    help="범주형으로 쓸 컬럼 (hour, site_id, wa_station, wb_station). 피처에 없으면 추가한다. "
                         "LightGBM 은 범주형 분할, XGBoost 는 enable_categorical")
    ap.add_argument("--params", default=None, help='LightGBM 파라미터 덮어쓰기 JSON. 예: \'{"num_leaves": 63}\'')
    ap.add_argument("--seed", type=int, default=None, help="첫 시드. 기본 0")
    ap.add_argument("--seeds", type=int, default=None,
                    help="시드 개수. 2 이상이면 seed, seed+1, … 모델의 예측을 평균 (채택 판단은 2로). 기본 1")
    ap.add_argument("--target", choices=["cf", "csi"], default=None,
                    help="cf(기본): 발전율 / csi: 발전율 ÷ 청천 기준(Haurwitz 일사/1000). csi 는 가중치에 기준을 곱해 "
                         "발전율 단위 오차와 같은 손실이 되게 한다")
    ap.add_argument("--fit-where", default=None, metavar="조건",
                    help='이 조건(pandas query)을 만족하는 학습 행만 쓴다. 예측은 전체 행. 예: "wa_cloud_r17 >= 90"')
    args = ap.parse_args()
    exp, cfg, base_name = resolve_config(args)
    if args.cut:
        if args.mode != "valid":
            raise SystemExit("--cut 은 --mode valid 에서만 쓴다")
        exp = f"{exp}_cut{args.cut[:4]}"
    log(f"실험 {exp}" + (f" (base {base_name})" if base_name else "") + f": {cfg}")

    df = load_frame(args.mode, cfg["years"], args.cut)
    df.loc[~df.is_fit, ["energy", "cf"]] = np.nan          # 누설 방지: 예측 행의 실적을 지운다
    df["is_dead"] = df.is_dead.eq(True) & df.is_fit
    log(f"mode={args.mode}: 학습 후보 {int(df.is_fit.sum()):,}행 / 예측 {int((~df.is_fit).sum()):,}행")
    df, added = apply_hooks(df, cfg["add"])

    tr = df[df.is_fit & (df.elev > 0)]
    if cfg["drop_dead"]:
        tr = tr[~tr.is_dead.astype(bool)]
    if cfg.get("fit_where"):
        n0 = len(tr)
        tr = tr.query(cfg["fit_where"])
        log(f"--fit-where {cfg['fit_where']}: 학습 행 {n0:,} → {len(tr):,}")
    y = (tr.energy / tr.capacity_kw).clip(0, 1.2).astype("float32")
    recent = np.where(tr.ts >= tr.ts.max() - pd.DateOffset(months=12), 2.0, 1.0)   # 컷오프 전 12개월 × 2
    w = {"none": None, "cap": tr.capacity_kw.to_numpy(),
         "cap_k": tr.capacity_kw.to_numpy() * k_weight(tr.ts), "recent": recent}[cfg["weight"]]
    pr = df[~df.is_fit]
    if cfg.get("target", "cf") == "csi":
        ref_tr = clear_ref(tr)
        y = (y / ref_tr).clip(0, 3.0).astype("float32")
        w = ref_tr if w is None else w * ref_tr
    params = {**PARAMS, **cfg["params"], "objective": OBJECTIVE[cfg["objective"]],
              "random_state": cfg["seed"]}
    log(f"학습 {len(tr):,}행 ({tr.ts.min():%Y-%m-%d} ~ {tr.ts.max():%Y-%m-%d}), 예측 {len(pr):,}행")

    sets = model_feature_sets(cfg["rounds"])
    seeds = [cfg["seed"] + i for i in range(cfg["seeds"])]
    preds, feature_sets = [], []
    for cols in sets:
        feats = [c for c in cols + COMMON + added
                 if not any(c == d or c in (f"{d}_r10", f"{d}_r17") for d in cfg["drop"])]
        cats = cfg.get("cat", [])
        feats += [c for c in cats if c not in feats]          # 범주형으로 쓸 컬럼이 피처에 없으면 추가 (발전소·지점 ID)
        feature_sets.append(feats)
        X_tr, X_pr = to_X(tr, feats, cats), to_X(pr, feats, cats)
        for s in seeds:
            model = make_model(cfg["model"], cfg["objective"], cfg["params"], s, has_cat=bool(cats)).fit(
                X_tr, y, sample_weight=w)
            preds.append(model.predict(X_pr))
            log(f"모델 {len(preds)}/{len(sets) * len(seeds)} 학습 완료 (피처 {len(feats)}개, seed {s})")
    p = np.clip(np.mean(preds, axis=0), 0, None)
    if cfg.get("target", "cf") == "csi":
        p = np.clip(p * clear_ref(pr), 0, 1.2)
    p = np.where(pr.elev.to_numpy() > 0, p, 0.0) * pr.capacity_kw.to_numpy()
    out = pr[["site_id", "ts"]].reset_index(drop=True).assign(pred_kwh=p)

    out_dir = PREDS if args.mode == "valid" else PREDS_TEST
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp}.parquet"
    out.to_parquet(path, index=False)
    meta = {"exp": exp, "mode": args.mode, "cut": args.cut, "base": base_name,
            **{k: v for k, v in cfg.items() if k != "params"}, "params_override": cfg["params"],
            "params": params, "features": feature_sets, "n_train": len(tr),
            "train_period": [str(tr.ts.min()), str(tr.ts.max())], "created": datetime.now().isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"저장 {path.relative_to(ROOT)} ({len(out):,}행)")

    if args.cut:
        from evaluate import evaluate_preds, read_preds
        start, end = cut_window(args.cut)
        a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy"])
        a = a[(a.ts >= start) & (a.ts < end)].reset_index(drop=True)
        a["site_id"] = a.site_id.astype(str)
        r = evaluate_preds(read_preds(path), actual=a)
        print(f"\n[{exp}] 다른 해 검증 {start:%Y-%m} ~ {end - pd.Timedelta(hours=1):%Y-%m} · 참조 구성 · 발전소 {r['n_sites']}곳")
        print(f"  벌금          {r['penalty']:>15,.0f}원   (수익의 {r['penalty'] / r['revenue'] * 100:.2f}%)")
        print(f"  11~15시 벌금  {r['peak']:>15,.0f}원")
        print("  비교하려면 기준 모델도 같은 --cut 으로 돌린다")
    elif args.mode == "valid":
        from evaluate import report
        report(path)


if __name__ == "__main__":
    main()
