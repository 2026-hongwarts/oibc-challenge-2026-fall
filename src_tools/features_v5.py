"""피처 v5 — 참고 코드(건물 전력 6위 '일조·일사 예측')에서 온 2단계 보조 모델.

A1  add_aux_obs: 학습 기간 실측 기상(B 자외선지수 = 일사 대용, A·B 운량)을 예보로 먼저 예측하는 보조 모델을 만들고,
    그 예측을 발전량 모델의 피처로 넣는다.

규정 §6 "학습 기간에 제공된 실측 기상은 학습 단계의 보조로만 쓸 수 있다. 평가 입력에는 넣지 않는다."
  - 실측은 보조 모델의 정답으로만 쓴다(학습 행의 마지막 시각까지만 읽음).
  - 평가(예측) 행에는 예보로 만든 보조 모델 예측만 들어간다(실측 없음).
누설 방지
  - 학습 행의 보조 예측은 날짜 블록 4개로 나눈 교차 예측(out-of-fold): 그 블록을 빼고 학습한 모델이 예측.
    → 발전량 모델이 '정답을 본 보조 예측'으로 학습하지 않는다.
  - 예측 행은 학습 행 전체로 학습한 보조 모델이 예측.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from oibc_common import DATA

AUX_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
TARGETS = [("wb", "wb_uv_idx", "wx_aux_uv_b"), ("wa", "wa_cloud", "wx_aux_cloud_a"), ("wb", "wb_cloud", "wx_aux_cloud_b")]
RAW = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wb_cloud", "wb_temp", "wb_rel_hum",
       "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point", "wb_ceiling"]
N_BLOCKS = 4


def _inputs(df: pd.DataFrame) -> list[str]:
    cols = [f"{c}_r{r}" for c in RAW for r in (10, 17)]
    cols += ["elev", "azim", "cosz", "hour", "lat", "lon"]
    cols += [c for c in df.columns if c.startswith(("wx_day_", "wx_prev_", "wx_am_", "wx_wa_cloud_r17_", "wx_wa_humidity_r17_"))
             and not c.startswith("wx_aux_")]
    return [c for c in cols if c in df.columns]


def _observed(prov: str, var: str, last: pd.Timestamp) -> pd.DataFrame:
    st = f"{prov}_station"
    o = pd.read_parquet(DATA / "train" / f"observed_{prov}.parquet", columns=[st, "ts", var])
    o["ts"] = pd.to_datetime(o.ts).dt.as_unit("ns")
    o = o[o.ts <= last].dropna()
    if var == "wb_cloud":
        o = o[o[var] <= 100]
    if var == "wb_uv_idx":
        o = o[o[var] <= 15]
    o[st] = o[st].astype(str)
    o = o.rename(columns={var: "_obs"})
    return o.groupby([st, "ts"], as_index=False)._obs.mean()      # 같은 지점·시각 중복 행은 평균으로 하나로


def add_aux_obs(df):
    last = df.ts[df.is_fit].max()
    day = (df.elev > 0).to_numpy()
    feats = _inputs(df)
    X = df[feats].astype("float32")
    fit_day = df.is_fit.to_numpy() & day
    # 날짜 블록: 학습 행의 날짜를 시간순으로 4등분
    dates = df.date.to_numpy()
    fit_dates = np.sort(pd.unique(dates[fit_day]))
    edges = fit_dates[np.linspace(0, len(fit_dates) - 1, N_BLOCKS + 1).astype(int)]
    block = np.full(len(df), -1)
    for b in range(N_BLOCKS):
        hi = edges[b + 1] if b < N_BLOCKS - 1 else fit_dates[-1] + np.timedelta64(1, "D")
        block[fit_day & (dates >= edges[b]) & (dates < hi)] = b
    for prov, var, name in TARGETS:
        st = f"{prov}_station"
        obs = _observed(prov, var, last)
        key = df[[st, "ts"]].assign(**{st: df[st].astype(str)})
        y = key.merge(obs, on=[st, "ts"], how="left")._obs.to_numpy()
        has_y = fit_day & ~np.isnan(y)
        out = np.full(len(df), np.nan, dtype="float32")
        for b in range(N_BLOCKS):                                   # 학습 행: 교차 예측
            tr = has_y & (block != b)
            te = fit_day & (block == b)
            m = lgb.LGBMRegressor(**{**AUX_PARAMS, "random_state": b}).fit(X[tr], y[tr])
            out[te] = m.predict(X[te])
        m = lgb.LGBMRegressor(**{**AUX_PARAMS, "random_state": 9}).fit(X[has_y], y[has_y])   # 예측 행: 전체 학습
        te = ~df.is_fit.to_numpy() & day
        out[te] = m.predict(X[te])
        df[name] = out
    return df
