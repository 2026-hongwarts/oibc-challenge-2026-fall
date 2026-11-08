"""GFS 외부 예보 피처 (gfs_fetch.py → gfs_decode.py 가 만든 outputs/work/gfs_kr.parquet 을 시각으로 붙인다).

모든 값은 시각(ts)에만 의존하고 발전소마다 같다(규정 §7: 외부 기상은 모든 발전소에 같은 값, 고정 영역 요약).
대상일 D 의 값은 D 전날 00시 UTC 실행(전날 12:40 KST 경 공개)에서 온다(규정 §4·§6).
GFS 0.5° 자료가 없는 2021-01 ~ 2021-05 와 밤 시각은 빈칸(NaN).

    G1  add_gfs_kr       한반도 상자 전체 평균 일사량·운량 (2개)
    G2  add_gfs_regions  1° 고정 지역 42곳 + 전체 평균 (86개)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oibc_common import WORK, solar

_CACHE: dict[str, pd.DataFrame] = {}
KR_CENTER = (36.0, 127.5)          # 고정 상자(북위 33 ~ 39°, 동경 124 ~ 131°)의 가운데


def _center(col: str) -> tuple[float, float]:
    """gfs_dswrf_r{위도}_{경도} → 그 1° 지역의 중심. gfs_dswrf_kr → 상자 중심. (발전소 위치와 무관)"""
    if col.endswith("_kr"):
        return KR_CENTER
    la, lo = col.split("_r")[-1].split("_")
    return int(la) + 0.5, int(lo) + 0.5


def _allocate_hourly(g: pd.DataFrame) -> pd.DataFrame:
    """GFS 일사량은 3시간 평균(블록 안 세 시각에 같은 값)이다. 블록 평균은 지키면서
    그 지역 중심의 태양 cos 천정각에 비례해 시각별로 나눈다: 값_h = 블록평균 × cosz_h / 블록 안 cosz 평균.
    중심점이 지역마다 고정이라 같은 시각의 값은 모든 발전소에 같다(규정 §7)."""
    hour = g.ts.dt.hour
    blk = g.ts.dt.normalize().astype("int64") // 3_600_000_000_000 * 10 + (hour - 4) // 3
    for c in [c for c in g.columns if c.startswith("gfs_dswrf_")]:
        la, lo = _center(c)
        cz = np.clip(solar(g.ts.dt.tz_localize(None), la, lo)[2], 0, None).astype("float64")
        m = pd.Series(cz).groupby(blk.to_numpy()).transform("mean").to_numpy()
        f = np.where(m > 0.01, cz / np.where(m > 0.01, m, 1), 1.0)
        g[c] = g[c].to_numpy() * f
    return g


def _gfs() -> pd.DataFrame:
    if "g" not in _CACHE:
        g = pd.read_parquet(WORK / "gfs_kr.parquet")
        g["ts"] = pd.to_datetime(g.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
        _CACHE["g"] = _allocate_hourly(g)
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
