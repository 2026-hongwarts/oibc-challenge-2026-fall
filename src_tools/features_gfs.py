"""GFS 외부 예보 피처 (gfs_fetch.py → gfs_decode.py 가 만든 outputs/work/gfs_kr.parquet 을 시각으로 붙인다).

모든 값은 시각(ts)에만 의존하고 발전소마다 같다(규정 §7: 외부 기상은 모든 발전소에 같은 값, 고정 영역 요약).
대상일 D 의 값은 D 전날 00시 UTC 실행(전날 12:40 KST 경 공개)에서 온다(규정 §4·§6).
GFS 0.5° 자료가 없는 2021-01 ~ 2021-05 와 밤 시각은 빈칸(NaN).

    G1  add_gfs_kr       한반도 상자 전체 평균 일사량·운량 (2개)
    G2  add_gfs_regions  1° 고정 지역 42곳 + 전체 평균 (86개)
"""
from __future__ import annotations

import pandas as pd

from oibc_common import WORK

_CACHE: dict[str, pd.DataFrame] = {}


def _gfs() -> pd.DataFrame:
    if "g" not in _CACHE:
        g = pd.read_parquet(WORK / "gfs_kr.parquet")
        g["ts"] = pd.to_datetime(g.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
        _CACHE["g"] = g
    return _CACHE["g"]


def _attach(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = _gfs()[["ts"] + cols]
    m = df[["ts"]].merge(g, on="ts", how="left")
    for c in cols:
        df[c] = m[c].to_numpy()
    return df


def add_gfs_kr(df):
    return _attach(df, ["gfs_dswrf_kr", "gfs_tcdc_kr"])


def add_gfs_regions(df):
    cols = [c for c in _gfs().columns if c.startswith("gfs_")]
    return _attach(df, cols)
