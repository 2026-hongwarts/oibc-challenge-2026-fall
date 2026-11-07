"""피처 v2 (docs/피처_엔지니어링_계획.md 의 H1~H3). 실험 하나 = 함수 하나.

    python src_tools/train_site_model.py --exp E6-01x2 --base champion --add features_v2:add_b_day
    python src_tools/train_site_model.py --exp E6-02x2 --base champion --add features_v2:add_cloudbase
    python src_tools/train_site_model.py --exp E6-03x2 --base champion --add features_v2:add_day_spread

모두 예보·태양 기하로만 만든다(발전량을 쓰지 않음). 하루 값은 같은 발전소·같은 대상일(date)의 해 뜬 시간으로 낸다.
행 수와 순서는 바꾸지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _day(df: pd.DataFrame, s: pd.Series, how: str, min_elev: float = 0.0) -> pd.Series:
    return s.where(df.elev > min_elev).groupby([df.site_id, df.date]).transform(how)


def add_b_day(df: pd.DataFrame) -> pd.DataFrame:
    """H1 B사 하루 요약: 해 뜬 시간 최소 운고, 자외선 감쇠비(자외선 ÷ 맑은 날 기대치) 하루 평균, 최대 강수확률."""
    uv_clear = 12.5 * df.cosz.clip(lower=0) ** 2.42                    # Madronich 근사
    ratio = (df.wb_uv_idx_r17 / uv_clear.clip(lower=0.5)).clip(0, 2)
    df["wx_day_ceil_min_b"] = _day(df, df.wb_ceiling_r17, "min")
    df["wx_day_uvratio_b"] = _day(df, ratio, "mean", min_elev=15)
    df["wx_day_pprob_max_b"] = _day(df, df.wb_precip_prob_r17, "max")
    return df


def add_cloudbase(df: pd.DataFrame) -> pd.DataFrame:
    """H2 구름 바닥 물리: 상승응결고도(125m × (기온 − 이슬점))와 B 운고와의 차이 (시각)."""
    df["wx_lcl"] = 125.0 * (df.wa_temp_r17 - df.wb_dew_point_r17)
    df["wx_ceil_minus_lcl"] = df.wb_ceiling_r17 - df.wx_lcl
    return df


def add_day_spread(df: pd.DataFrame) -> pd.DataFrame:
    """H3 하루 단위 예보 흔들림: 두 발행분 운량 차, 두 제공사 운량 차, 하루 운량 표준편차 (A)."""
    df["wx_day_roundgap_a"] = _day(df, (df.wa_cloud_r17 - df.wa_cloud_r10).abs(), "mean")
    df["wx_day_provgap"] = _day(df, df.wa_cloud_r17 - df.wb_cloud_r17, "mean")
    df["wx_day_cloud_std_a"] = _day(df, df.wa_cloud_r17, "std")
    return df
