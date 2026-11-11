"""학습 데이터 정리 (11/11 파이프라인 점검 4번) — 부분 정지·저하 기간을 학습에서 뺀다.

    --add features_clean:add_low_month --drop low_month --fit-where "low_month == 0"

문제: X13(정지일 제외)은 하루 단위·발전 0 근처만 걸렀다. 그런데 몇 달 동안 평소의 60% 정도로만 도는 발전소가 있다
    (예: PV218 2024-08 ~ 2025-01). 상대 수준 0.85 미만이 2개월 이상 이어진 곳 15곳, 3개월 이상 6곳.
    이런 기간을 그대로 학습하면 정상 운전 발전량을 낮게 배운다.
방법(학습 행 df.is_fit 만 사용 — 예측 행·평가 기간 통계는 쓰지 않음):
    1. 발전소 × 달(YYYY-MM)마다 해가 높은 시각(고도 > 15°) 발전율 평균
    2. 같은 달 모든 발전소의 그 값 중앙값으로 나눔 → 그 달의 상대 수준
    3. 그 발전소의 상대 수준 중앙값(여러 달)으로 다시 나눔 → 평소 대비 수준
    4. 평소 대비 0.85 미만이면 그 발전소-달의 학습 행을 low_month = 1 (피처로는 쓰지 않음, --drop low_month)
    고도 > 15° 행이 50개 미만인 발전소-달은 판단하지 않음(0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESH = 0.85


def add_low_month(df):
    fit = df.is_fit.to_numpy() & (df.elev > 15).to_numpy() & df.energy.notna().to_numpy()
    t = df.loc[fit, ["site_id", "ts", "energy", "capacity_kw"]].copy()
    t["ym"] = t.ts.dt.strftime("%Y-%m")
    t["cf"] = t.energy / t.capacity_kw
    g = t.groupby(["site_id", "ym"]).cf.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 50]
    g["ratio"] = g["mean"] / g.groupby("ym")["mean"].transform("median")
    g["rel"] = g.ratio / g.groupby("site_id").ratio.transform("median")
    low = g.loc[g.rel < THRESH, ["site_id", "ym"]].assign(low_month=1)
    ym = df.ts.dt.strftime("%Y-%m")
    key = pd.DataFrame({"site_id": df.site_id.to_numpy(), "ym": ym.to_numpy()})
    m = key.merge(low, on=["site_id", "ym"], how="left").low_month.fillna(0).to_numpy()
    df["low_month"] = np.where(df.is_fit.to_numpy(), m, 0).astype("int8")
    n = int(len(low))
    print(f"  add_low_month: 학습 기간 발전소-달 {len(g):,}개 중 평소 대비 {THRESH} 미만 {n}개 → 학습에서 뺌", flush=True)
    return df
