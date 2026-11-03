"""② 예보 피처 — 다른 해 백테스트 (공용 스크립트는 고치지 않는다).

피처 후보를 가상 평가연도(2024-06 ~ 2025-05) 하나로 골랐기 때문에, 그해에만 맞는 것을 골랐을 위험을 확인한다.
학습 컷오프를 1년 앞(2023-06-01 01:00)으로 옮겨, 2021-06 ~ 2023-05로 학습하고 2023-06 ~ 2024-05를 예측·채점한다.
학습 방식(기준 모델 E1-01x2: 두 발행분 한 모델, L1, 2년, 시드 2개)과 채점식(evaluate.py)은 팀 것을 그대로 쓴다.
그 해에는 발전소가 148 → 253곳으로 늘어나는 중이라, 있는 발전소만으로 채점한다.

    python analysis_lab/E2_backtest.py E1-01 E2-04 E2-15 E2-16      # 저장소 맨 위 폴더에서
    BT_SEED0=2 python analysis_lab/E2_backtest.py E1-01 E2-24        # 시드 2·3 (결과 이름 끝에 @s2)
    BT_YEAR=2022 python analysis_lab/E2_backtest.py E1-01 E2-24      # 다른 해: 2022-06 ~ 2023-05 (이름 끝에 @y2022)
항상 E1-01(기준 모델)을 맨 앞에 넣는다. 같은 해·같은 시드의 E1-01 대비로 비교한다.
2022년 시험연도는 후보 선택에 한 번도 쓰지 않은 '마지막 확인용'이다. 데이터가 2021-01 부터라 학습이 약 16개월로 짧다.
"""
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src_tools"))
import train_site_model as tsm                                    # noqa: E402
from evaluate import group_frame, load_composition, penalty_frame, summarize   # noqa: E402
from oibc_common import TZ, WORK, load_smp                       # noqa: E402

YEAR = int(os.environ.get("BT_YEAR", "2023"))
BT_FROM = pd.Timestamp(f"{YEAR}-06-01 01:00", tz=TZ)
BT_TO = pd.Timestamp(f"{YEAR + 1}-06-01 01:00", tz=TZ)
FIT_FROM = BT_FROM - pd.DateOffset(years=2)
SEED0 = int(os.environ.get("BT_SEED0", "0"))
SEEDS = (SEED0, SEED0 + 1)
TAG = ("" if YEAR == 2023 else f"@y{YEAR}") + ("" if SEED0 == 0 else f"@s{SEED0}")
CONFIGS = {
    "E1-01": [],                                                  # 기준 모델
    "E2-04": ["features_weather:add_day_summary"],
    "E2-15": ["features_weather:add_day_summary_2prov"],
    "E2-16": ["features_weather:add_day_summary_2prov_nocloud"],
    "E2-18": ["features_weather:add_e16_cloud_std"],
    "E2-19": ["features_weather:add_e16_rain_hours"],
    "E2-20": ["features_weather:add_e16_day_trange"],
    "E2-21": ["features_weather:add_e16_prev_rain"],
    "E2-22": ["features_weather:add_e16_fleet_day"],
    "E2-23": ["features_weather:add_e16_pprob"],
    "E2-24": ["features_weather:add_e21_prev_more"],
    "E2-25": ["features_weather:add_e21_prev2_rain"],
    "E2-26": ["features_weather:add_e21_day_trange"],
    "E2-27": ["features_weather:add_station_day"],
    "E2-28": ["features_weather:add_station_day_prevb"],
    "E2-30": ["features_weather:add_e24_humid"],
    "E2-34": ["features_weather:add_e24_no_rain_a"],
    "E2-35": ["features_weather:add_e24_no_trange_b"],
    "E2-36": ["features_weather:add_e24_no_prev_cloud"],
    "E2-37": ["features_weather:add_e24_no_prev_trange"],
    "E2-40": ["features_weather:add_e24_fog"],
    "E2-46": ["features_weather:add_e40_no_prev_trange"],
    "E2-47": ["features_weather:add_e24_fog_0509"],
    "E2-48": ["features_weather:add_e24_fog_0711"],
}
OUT = ROOT / "analysis_lab" / "E2_backtest_results.csv"

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def load_window() -> pd.DataFrame:
    base = pd.read_parquet(WORK / "base.parquet")
    fit = base[(base.ts >= FIT_FROM) & (base.ts < BT_FROM)]
    pred = base[(base.ts >= BT_FROM) & (base.ts < BT_TO)]
    df = pd.concat([fit.assign(is_fit=True), pred.assign(is_fit=False)], ignore_index=True)
    df["site_id"] = df.site_id.astype(str)
    df["energy_true"] = df.energy                                 # 채점용으로만 따로 둔다
    df.loc[~df.is_fit, ["energy", "cf"]] = np.nan                 # 누설 방지: 예측 행 실적을 지운다
    df["is_dead"] = df.is_dead.eq(True) & df.is_fit
    return df


def run(df: pd.DataFrame, adds: list[str]) -> pd.DataFrame:
    d, added = tsm.apply_hooks(df.copy(), adds)
    tr = d[d.is_fit & (d.elev > 0)]
    pr = d[~d.is_fit]
    feats = tsm.model_feature_sets("both")[0] + tsm.COMMON + added
    y = (tr.energy / tr.capacity_kw).clip(0, 1.2).astype("float32")
    params = {**tsm.PARAMS, "objective": tsm.OBJECTIVE["l1"]}
    preds = []
    for s in SEEDS:
        m = lgb.LGBMRegressor(**{**params, "random_state": s}).fit(tr[feats].astype("float32"), y)
        preds.append(m.predict(pr[feats].astype("float32")))
    p = np.clip(np.mean(preds, axis=0), 0, None)
    p = np.where(pr.elev.to_numpy() > 0, p, 0.0) * pr.capacity_kw.to_numpy()
    return pr[["site_id", "ts"]].reset_index(drop=True).assign(pred_kwh=p)


def score(df: pd.DataFrame, preds: pd.DataFrame) -> dict:
    actual = df.loc[~df.is_fit, ["site_id", "ts", "energy_true"]].rename(columns={"energy_true": "energy"})
    m = penalty_frame(group_frame(actual.reset_index(drop=True), preds, load_composition()), load_smp("train"))
    return summarize(m)


def main() -> None:
    names = sys.argv[1:] or list(CONFIGS)
    df = load_window()
    log(f"학습 {FIT_FROM:%Y-%m} ~ {BT_FROM:%Y-%m} {int(df.is_fit.sum()):,}행 / 예측·채점 {int((~df.is_fit).sum()):,}행, "
        f"발전소 {df.loc[~df.is_fit, 'site_id'].nunique()}곳")
    rows, ref = [], None
    for n in names:
        r = score(df, run(df, CONFIGS[n]))
        if n == "E1-01":
            ref = r
        row = {"config": n + TAG, "penalty": r["penalty"], "peak": r["peak"]}
        if ref is not None:
            row["vs_E1-01_%"] = (r["penalty"] / ref["penalty"] - 1) * 100
            row["peak_vs_E1-01_%"] = (r["peak"] / ref["peak"] - 1) * 100
            row["months_better"] = int((r["by_month"] < ref["by_month"]).sum())
        rows.append(row)
        log(f"{n}: 벌금 {r['penalty']:,.0f}원" + (f"  기준 대비 {row['vs_E1-01_%']:+.2f}%" if ref else ""))
        out = pd.DataFrame(rows)
        if OUT.exists():                                          # 이전 결과와 합치되 같은 config 는 새 값으로
            old = pd.read_csv(OUT)
            out = pd.concat([old[~old.config.isin(out.config)], out], ignore_index=True)
        out.to_csv(OUT, index=False)
    log(f"저장 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
