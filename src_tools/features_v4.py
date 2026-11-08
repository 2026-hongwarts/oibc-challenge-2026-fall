"""피처 v4 — 계획서 v4 의 11개 축 밖에서 찾은 후보 (analysis_lab/F5_빠진축_탐색.py 근거).

X17 달력: 요일·공휴일. 휴일 11~15시 실적이 예측보다 1.4 ~ 2.7%p 낮고(세 해 같은 방향), 2024-25 봄 맑은 휴일에
    실적 < 예측의 50% 인 발전소-날이 몰림(출력제어 흔적). 요일·공휴일은 미리 알 수 있는 값.
X18 패널 오염: 마지막 비 이후 날수, 7일 강수 합 (신호 약함).
X19 눈 덮임: 마지막 눈 이후 날수, 직전 3일 강설 합, 그날 최고기온 (눈 온 다음 날 −6 / −6 / −14%).
모두 예보와 달력만 쓴다. 대상일 이전 날의 예보는 그날 17시 이전에 발행된 것이라 쓸 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features_v3 import _day, _shift_days

# 공휴일 (대체·임시 공휴일, 선거일 포함) + 근로자의 날(5/1). 2020-06 ~ 2026-05
HOLIDAYS = np.array([
    "2020-06-06", "2020-08-15", "2020-08-17", "2020-09-30", "2020-10-01", "2020-10-02", "2020-10-03", "2020-10-09",
    "2020-12-25",
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01", "2021-05-05", "2021-05-19", "2021-06-06",
    "2021-08-15", "2021-08-16", "2021-09-20", "2021-09-21", "2021-09-22", "2021-10-03", "2021-10-04", "2021-10-09",
    "2021-10-11", "2021-12-25",
    "2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-03-01", "2022-03-09", "2022-05-05", "2022-05-08",
    "2022-06-01", "2022-06-06", "2022-08-15", "2022-09-09", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-03",
    "2022-10-09", "2022-10-10", "2022-12-25",
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-03-01", "2023-05-05", "2023-05-27",
    "2023-05-29", "2023-06-06", "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-02", "2023-10-03",
    "2023-10-09", "2023-12-25",
    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-05",
    "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-01",
    "2024-10-03", "2024-10-09", "2024-12-25",
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-01", "2025-03-03", "2025-05-05",
    "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-05", "2025-10-06", "2025-10-07",
    "2025-10-08", "2025-10-09", "2025-12-25",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-01", "2026-03-02", "2026-05-05", "2026-05-24",
    "2026-05-25",
    "2021-05-01", "2022-05-01", "2023-05-01", "2024-05-01", "2025-05-01", "2026-05-01",
], dtype="datetime64[D]")


def _offset_vals(df: pd.DataFrame, col: str, hours: int) -> np.ndarray:
    """같은 발전소의 hours 시간 뒤(음수면 앞) 값. 다음 대상일에 속하는 시각(전날 17시 이후 발행 예보)은 NaN."""
    q = df[["site_id", "ts"]].copy()
    q["ts"] = q.ts + pd.Timedelta(hours=hours)
    look = df[["site_id", "ts", "date", col]].drop_duplicates(["site_id", "ts"]).rename(columns={"date": "_d", col: "_v"})
    m = q.merge(look, on=["site_id", "ts"], how="left")
    v = m._v.to_numpy(dtype=float)
    if hours > 0:
        v = np.where((m._d > df.date.reset_index(drop=True)).to_numpy(), np.nan, v)
    return v


def add_hour_wide(df):
    """X03b: X03(A 운량·습도의 ±1시간, 3시간 평균)을 넓힘 — ±2시간·5시간 평균, B 운량·습도까지."""
    for v in ["wa_cloud_r17", "wb_cloud_r17", "wa_humidity_r17", "wb_rel_hum_r17"]:
        o = {k: _offset_vals(df, v, k) for k in (-2, -1, 1, 2)}
        cur = df[v].to_numpy(dtype=float)
        df[f"wx_{v}_prev1"], df[f"wx_{v}_next1"] = o[-1], o[1]
        df[f"wx_{v}_avg3"] = np.nanmean(np.vstack([o[-1], cur, o[1]]), axis=0)
        df[f"wx_{v}_avg5"] = np.nanmean(np.vstack([o[-2], o[-1], cur, o[1], o[2]]), axis=0)
    return df


def add_ceiling(df):
    """V04: B 운고(구름 바닥 높이) 두 발행분을 원값 그대로. EDA 에서 핵심 변수로 꼽았지만 baseline 선택에 없던 변수.
    X01 에서는 다른 원변수 8개와 묶어 넣어 기각 → 단독 효과 확인."""
    df["wx_raw_wb_ceiling_r10"] = df.wb_ceiling_r10
    df["wx_raw_wb_ceiling_r17"] = df.wb_ceiling_r17
    return df


def add_wind_vec(df):
    """V08 (참고: 동서발전 태양광 6위 코드): 풍향은 0° = 360° 인 원형 값 → sin·cos, 8방위 구간(0 ~ 7),
    바람 벡터(풍속 × sin·cos). 17시 발행분 A·B."""
    for p, d, s in [("a", "wa_wind_dir_r17", "wa_wind_speed_r17"), ("b", "wb_wind_dir_r17", "wb_wind_speed_r17")]:
        rad = np.deg2rad(df[d].to_numpy(dtype=float))
        df[f"wx_wdir_sin_{p}"] = np.sin(rad)
        df[f"wx_wdir_cos_{p}"] = np.cos(rad)
        df[f"wx_wdir_sector_{p}"] = ((df[d].to_numpy(dtype=float) + 22.5) // 45) % 8
        df[f"wx_wind_u_{p}"] = df[s].to_numpy(dtype=float) * np.sin(rad)
        df[f"wx_wind_v_{p}"] = df[s].to_numpy(dtype=float) * np.cos(rad)
    return df


def add_temp_anom(df):
    """V09 (참고: 동서발전 태양광 6위 코드): 기온 − (그 달·그 시각의 학습 기간 평균 기온).
    같은 기온이라도 계절·시각 대비 높으면 맑음 신호. 평균은 학습 행(is_fit)으로만 계산(평가 기간 통계 금지)."""
    fit = df.is_fit.to_numpy()
    for p, c in [("a", "wa_temp_r17"), ("b", "wb_temp_r17")]:
        clim = df.loc[fit].groupby(["month", "hour"])[c].mean().rename("_clim")
        m = df[["month", "hour"]].merge(clim, left_on=["month", "hour"], right_index=True, how="left")
        df[f"wx_tanom_{p}"] = df[c].to_numpy() - m._clim.to_numpy()
    return df


def add_calendar(df):
    days = df.date.to_numpy().astype("datetime64[D]")
    dow = df.date.dt.dayofweek
    df["cal_dow"] = dow.astype("int8")
    df["cal_off"] = ((dow >= 5).to_numpy() | np.isin(days, HOLIDAYS)).astype("int8")
    return df


def _days_since(df: pd.DataFrame, event_col: str, cap: int = 60) -> np.ndarray:
    """발전소별, 전날까지 마지막으로 사건이 있었던 날로부터 지난 날수."""
    sd = df.groupby(["site_id", "date"], observed=True)[event_col].first().reset_index()
    sd = sd.sort_values(["site_id", "date"])
    last = sd.date.where(sd[event_col]).groupby(sd.site_id).ffill()
    last = last.groupby(sd.site_id).shift(1)
    sd["_ds"] = ((sd.date - last).dt.days).fillna(cap).clip(upper=cap)
    return df[["site_id", "date"]].merge(sd[["site_id", "date", "_ds"]], on=["site_id", "date"], how="left")._ds.to_numpy()


def add_days_since_rain(df):
    df["_rain_ev"] = _day(df, df.wa_rain_r17, "sum", None) >= 5.0
    df["wx_dsr"] = _days_since(df, "_rain_ev")
    df["_rain_d"] = _day(df, df.wa_rain_r17, "sum", None)
    names = [f"_r{k}" for k in range(1, 8)]
    for k, n in enumerate(names, start=1):
        df = _shift_days(df, ["_rain_d"], [n], days=k)
    df["wx_rain7"] = df[names].fillna(0).sum(axis=1)
    return df.drop(columns=["_rain_ev", "_rain_d"] + names)


def add_snow_persist(df):
    df["_snow_d"] = _day(df, df.wa_snow_r17, "sum", None) + _day(df, df.wb_snow_r17, "sum", None) / 10
    df["_snow_ev"] = df._snow_d >= 0.5
    df["wx_dss"] = _days_since(df, "_snow_ev", cap=30)
    names = ["_s1", "_s2", "_s3"]
    for k, n in enumerate(names, start=1):
        df = _shift_days(df, ["_snow_d"], [n], days=k)
    df["wx_snow3"] = df[names].fillna(0).sum(axis=1)
    df["wx_day_tmax_a"] = _day(df, df.wa_temp_r17, "max", None)
    return df.drop(columns=["_snow_d", "_snow_ev"] + names)
