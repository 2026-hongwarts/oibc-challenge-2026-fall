"""발전소 단위 LightGBM 학습·예측. 옵션을 안 주면 E4-00(baseline.py 재현).

    python train_site_model.py --exp E4-00                     가상 평가연도 예측 → preds/E4-00.parquet, 바로 채점
    python train_site_model.py --exp E3-01 --drop-dead         정지일 빼고 학습
    python train_site_model.py --exp E2-01 --rounds both       두 라운드 예보를 한 모델에
    python train_site_model.py --exp E2-02 --add features_weather:add_weather_features
    python train_site_model.py --exp E4-00 --mode test         컷오프 2025-06 으로 학습 → preds_test/E4-00.parquet
    python train_site_model.py --exp E2-01x2 --rounds both --seeds 2    채택 판단용(시드 2개 평균)

시드 하나만 바꿔도 가상 평가연도 벌금이 약 0.5% 움직인다(E4-00 seed 0 vs 1: +0.51%).
그래서 채택 여부는 --seeds 2 결과를 기준 모델(역시 시드 2개)과 비교해 정한다.

E4-00 = baseline.py 와 같은 설정
    round 10 모델과 round 17 모델 예측의 평균 · 컷오프 전 2년 · 태양고도 > 0 행 · 정지일 포함
    타깃 CF(0~1.2) · L2 · n_estimators 600, learning_rate 0.06, num_leaves 127, min_child_samples 200,
    subsample 0.8, colsample_bytree 0.8 · 밤(태양고도 ≤ 0) 예측은 0
    차이: 한 라운드만 빠진 예보 행을 다른 라운드로 채웠고(build_base), random_state=0 으로 고정.

--add 로 넘기는 함수 규칙 (모듈:함수, 여러 번 가능)
    f(df) -> df. df 에는 학습 행(is_fit=True, 컷오프 전 학습 기간의 모든 시각)과 예측 행(is_fit=False)이 함께 들어온다.
    새로 생긴 열은 자동으로 피처에 추가된다. 행 수와 순서는 그대로 둔다.
    발전량(energy, cf)으로 만드는 값은 반드시 df[df.is_fit] 에서만 계산한다(아니면 누설).

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

from oibc_common import PREDS, ROOT, TEST_FROM, VALID_FROM, WORK, k_weight

WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press"]
WB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq",
           "wb_vis", "wb_wind_speed", "wb_dew_point"]
COMMON = ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "capacity_kw", "lat", "lon"]
PARAMS = dict(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
OBJECTIVE = {"l2": "regression", "l1": "regression_l1"}

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


def main() -> None:
    ap = argparse.ArgumentParser(description="발전소 단위 LightGBM 학습·예측 (기본값 = E4-00)")
    ap.add_argument("--exp", required=True, help="실험 번호 (예: E2-01). 결과 파일 이름이 됨")
    ap.add_argument("--mode", choices=["valid", "test"], default="valid",
                    help="valid: 2024-05 까지 학습 → 가상 평가연도 예측 / test: 2025-05 까지 학습 → 실제 평가 기간 예측")
    ap.add_argument("--rounds", choices=["avg", "both", "10", "17"], default="avg")
    ap.add_argument("--years", type=float, default=2, help="컷오프 전 학습 기간(년). 0 이면 전체")
    ap.add_argument("--drop-dead", action="store_true", help="정지 추정일(is_dead)을 학습에서 뺌")
    ap.add_argument("--weight", choices=["none", "cap", "cap_k"], default="none",
                    help="표본 가중치: cap = 설비용량, cap_k = 설비용량 × k(11~15시 3)")
    ap.add_argument("--objective", choices=list(OBJECTIVE), default="l2")
    ap.add_argument("--add", action="append", default=[], metavar="모듈:함수", help="피처 함수")
    ap.add_argument("--params", default="{}", help='LightGBM 파라미터 덮어쓰기 JSON. 예: \'{"num_leaves": 63}\'')
    ap.add_argument("--seed", type=int, default=0, help="첫 시드")
    ap.add_argument("--seeds", type=int, default=1,
                    help="시드 개수. 2 이상이면 seed, seed+1, … 모델의 예측을 평균 (채택 판단은 2로)")
    args = ap.parse_args()

    df = load_frame(args.mode, args.years)
    log(f"mode={args.mode}: 학습 후보 {int(df.is_fit.sum()):,}행 / 예측 {int((~df.is_fit).sum()):,}행")
    df, added = apply_hooks(df, args.add)

    tr = df[df.is_fit & (df.elev > 0)]
    if args.drop_dead:
        tr = tr[~tr.is_dead.astype(bool)]
    y = (tr.energy / tr.capacity_kw).clip(0, 1.2).astype("float32")
    w = {"none": None, "cap": tr.capacity_kw.to_numpy(),
         "cap_k": tr.capacity_kw.to_numpy() * k_weight(tr.ts)}[args.weight]
    pr = df[~df.is_fit]
    params = {**PARAMS, **json.loads(args.params), "objective": OBJECTIVE[args.objective],
              "random_state": args.seed}
    log(f"학습 {len(tr):,}행 ({tr.ts.min():%Y-%m-%d} ~ {tr.ts.max():%Y-%m-%d}), 예측 {len(pr):,}행")

    sets = model_feature_sets(args.rounds)
    seeds = [args.seed + i for i in range(args.seeds)]
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

    out_dir = PREDS if args.mode == "valid" else ROOT / "preds_test"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{args.exp}.parquet"
    out.to_parquet(path, index=False)
    meta = {**vars(args), "params": params, "features": feature_sets, "n_train": len(tr),
            "train_period": [str(tr.ts.min()), str(tr.ts.max())], "created": datetime.now().isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"저장 {path.relative_to(ROOT)} ({len(out):,}행)")

    if args.mode == "valid":
        from evaluate import report
        report(path)


if __name__ == "__main__":
    main()
