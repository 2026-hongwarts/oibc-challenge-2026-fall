"""우리 피처 모듈. 검증을 거친 것만 둔다.

add_daily_v1: 예보 하루 요약 11개. 팀원 코드(features_weather.add_weather_features, E2-final)를 2026-11-08 에
    우리가 검증한 뒤(코드 누설 점검, 원 단위 재현, 선택에 안 쓴 2023-24·2022-23 검증, 리더보드 확인) 옮겨 고정했다.
    값은 원본과 완전히 같다(analysis_lab 검증 기록). 이후 팀원 코드 변경과 무관하다.

    그날 (같은 대상일 예보)
      wx_day_cloud     A 운량의 해 뜬 시간 평균
      wx_day_trange    A 일교차 (하루 최고 − 최저, 모든 시각)
      wx_day_trange_b  B 일교차
      wx_day_rain      B 총강수 하루 합
      wx_day_rain_a    A 강우 하루 합
    전날 (전 대상일 예보 — 예측 시점 전에 이미 발행됨)
      wx_prev_rain, wx_prev_rain_a, wx_prev_cloud, wx_prev_trange
    아침 안개 신호 (같은 대상일 06 ~ 10시)
      wx_am_vis_min    B 시정 최솟값
      wx_am_humid      A 습도 평균

    python src_tools/train_site_model.py --exp ... --base champion --add features_ours:add_daily_v1
"""
from __future__ import annotations

import pandas as pd


def _shift_days(df: pd.DataFrame, cols: list[str], names: list[str], days: int = 1) -> pd.DataFrame:
    """발전소 × 대상일 값을 days 일 뒤 날짜에 붙인다 (그날 값 → 다음 날의 '전날 값')."""
    d = df.groupby(["site_id", "date"], observed=True)[cols].first().reset_index()
    d["date"] = d.date + pd.Timedelta(days=days)
    d = d.rename(columns=dict(zip(cols, names)))
    m = df[["site_id", "date"]].merge(d, on=["site_id", "date"], how="left")
    for n in names:
        df[n] = m[n].to_numpy()
    return df


def add_daily_v1(df: pd.DataFrame) -> pd.DataFrame:
    key = [df.site_id, df.date]
    df["wx_day_cloud"] = df.wa_cloud_r17.where(df.elev > 0).groupby(key).transform("mean")
    ta = df.wa_temp_r17.groupby(key)
    df["wx_day_trange"] = ta.transform("max") - ta.transform("min")
    df["wx_day_rain"] = df.wb_total_liq_r17.groupby(key).transform("sum")
    tb = df.wb_temp_r17.groupby(key)
    df["wx_day_trange_b"] = tb.transform("max") - tb.transform("min")
    df["wx_day_rain_a"] = df.wa_rain_r17.groupby(key).transform("sum")
    df = _shift_days(df, ["wx_day_rain", "wx_day_rain_a"], ["wx_prev_rain", "wx_prev_rain_a"])
    df = _shift_days(df, ["wx_day_cloud", "wx_day_trange"], ["wx_prev_cloud", "wx_prev_trange"])
    am = (df.hour >= 6) & (df.hour <= 10)
    df["wx_am_vis_min"] = df.wb_vis_r17.where(am).groupby(key).transform("min")
    df["wx_am_humid"] = df.wa_humidity_r17.where(am).groupby(key).transform("mean")
    return df
