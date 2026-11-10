"""GFS 외부 예보 피처 (gfs_fetch.py → gfs_decode.py 가 만든 표를 시각으로 붙인다).

모든 값은 시각(ts)에만 의존하고 발전소마다 같다(규정 §7: 외부 기상은 모든 발전소에 같은 값, 고정 영역 요약).
대상일 D 의 값은 D 전날 00시 UTC 실행(전날 12:40 KST 경 공개)에서 온다(규정 §4·§6). 마감 뒤 공개 파일은 해석 단계에서 뺐다.
GFS 0.5°·1° 자료가 없는 2021-01 ~ 2021-05 와 밤 시각은 빈칸(NaN).

    G1   add_gfs_kr             한반도 상자 전체 평균 일사량·운량 (2개)                       outputs/work/gfs_kr.parquet
    G2   add_gfs_regions        1° 고정 지역 42곳 + 전체 평균 (86개)                          〃
    G3   add_gfs_cells_pca      0.5° 칸 195개 × 2변수를 주성분 15개씩으로 (30개)              outputs/work/gfs_cells.parquet
         주성분은 그 실행의 학습 행 마지막 시각까지의 자료로만 구한다(평가 기간 통계 금지, 규정 §6·§7 이 허용한 방식)
    G5a  add_gfs_layers_kr      하층·중층·상층 운량 전체 평균 (3개)                           outputs/work/gfs_layers.parquet (1°)
    G5b  add_gfs_layers_regions 하층·중층·상층 운량 1° 지역 42곳 + 전체 평균 (129개)          〃
    G7   add_gefs_regions       GEFS 앙상블 평균·표준편차(일사량·운량) 1° 지역 42곳 + 전체 평균 (172개)  outputs/work/gefs_kr.parquet
    G7k  add_gefs_kr            GEFS 앙상블 평균·표준편차(일사량·운량) 상자 전체 평균 (4개)              〃
         GEFS = NOAA 전 지구 앙상블(31 멤버), 같은 00시 UTC 실행(13시 KST 경 공개). 표준편차 = 예보의 불확실성.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oibc_common import WORK, solar

_CACHE: dict[str, pd.DataFrame] = {}
KR_CENTER = (36.0, 127.5)          # 고정 상자(북위 33 ~ 39°, 동경 124 ~ 131°)의 가운데


def _center(col: str) -> tuple[float, float]:
    """열 이름 → 그 값의 고정 기준점 (발전소 위치와 무관).
    gfs_{var}_kr → 상자 중심, gfs_{var}_r{위도}_{경도} → 1° 지역 중심, gfs_c_{var}_{33p5}_{127p0} → 그 격자점."""
    if col.endswith("_kr"):
        return KR_CENTER
    if col.startswith("gfs_c_"):
        la, lo = col.split("_")[-2:]
        return float(la.replace("p", ".")), float(lo.replace("p", "."))
    la, lo = col.split("_r")[-1].split("_")
    return int(la) + 0.5, int(lo) + 0.5


def _allocate_hourly(g: pd.DataFrame) -> pd.DataFrame:
    """GFS 일사량은 3시간 평균(블록 안 세 시각에 같은 값)이다. 블록 평균은 지키면서
    그 기준점의 태양 cos 천정각에 비례해 시각별로 나눈다: 값_h = 블록평균 × cosz_h / 블록 안 cosz 평균.
    기준점이 고정이라 같은 시각의 값은 모든 발전소에 같다(규정 §7)."""
    hour = g.ts.dt.hour
    blk = (g.ts.dt.normalize().astype("int64") // 3_600_000_000_000 * 10 + (hour - 4) // 3).to_numpy()
    ts_naive = g.ts.dt.tz_localize(None)
    cols = [c for c in g.columns if "dswrf" in c]
    new = {}
    for c in cols:
        la, lo = _center(c)
        cz = np.clip(solar(ts_naive, la, lo)[2], 0, None).astype("float64")
        m = pd.Series(cz).groupby(blk).transform("mean").to_numpy()
        f = np.where(m > 0.01, cz / np.where(m > 0.01, m, 1), 1.0)
        new[c] = g[c].to_numpy() * f
    if new:                                       # 열 순서는 그대로 (피처 순서가 바뀌면 같은 설정도 결과가 달라짐)
        g = g.copy()
        g.loc[:, cols] = pd.DataFrame(new, index=g.index)[cols]
    return g


def _load(fname: str) -> pd.DataFrame:
    if fname not in _CACHE:
        g = pd.read_parquet(WORK / fname)
        g["ts"] = pd.to_datetime(g.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
        _CACHE[fname] = _allocate_hourly(g)
    return _CACHE[fname]


def _gfs() -> pd.DataFrame:           # 예전 이름 (호환)
    return _load("gfs_kr.parquet")


def _attach(df: pd.DataFrame, table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = df[["ts"]].merge(table[["ts"] + cols], on="ts", how="left")
    add = pd.DataFrame({c: m[c].to_numpy(dtype="float32") for c in cols}, index=df.index)
    return pd.concat([df, add], axis=1)


def add_gfs_kr(df):
    return _attach(df, _gfs(), ["gfs_dswrf_kr", "gfs_tcdc_kr"])


def add_gfs_regions(df):
    g = _gfs()
    return _attach(df, g, [c for c in g.columns if c.startswith("gfs_")])


def add_gfs_cells_pca(df, k: int = 15):
    from sklearn.decomposition import PCA
    g = _load("gfs_cells.parquet")
    last = df.ts[df.is_fit].max()
    out = g[["ts"]].copy()
    for var in ("dswrf", "tcdc"):
        cols = [c for c in g.columns if c.startswith(f"gfs_c_{var}_")]
        X = g[cols].to_numpy(dtype="float64")
        ok = ~np.isnan(X).any(axis=1)
        fit = ok & (g.ts <= last).to_numpy()                      # 학습 기간 시각만으로 주성분을 구함
        pca = PCA(n_components=k, random_state=0).fit(X[fit])
        Z = np.full((len(g), k), np.nan)
        Z[ok] = pca.transform(X[ok])
        for i in range(k):
            out[f"gfs_pc_{var}_{i + 1}"] = Z[:, i]
    return _attach(df, out, [c for c in out.columns if c.startswith("gfs_pc_")])


def add_gfs_stats(df):
    """G6 (참고: HEFTCom2024 4위 GEB — 격자를 평균뿐 아니라 최대·최소·분위수로 요약):
    고정 상자 0.5° 칸 195개의 일사량·운량을 시각마다 최대·최소·25/75% 분위수·표준편차로 (10개). 모든 발전소에 같은 값."""
    g = _load("gfs_cells.parquet")
    out = g[["ts"]].copy()
    for var in ("dswrf", "tcdc"):
        X = g[[c for c in g.columns if c.startswith(f"gfs_c_{var}_")]].to_numpy(dtype="float64")
        out[f"gfs_st_{var}_max"] = np.nanmax(X, axis=1)
        out[f"gfs_st_{var}_min"] = np.nanmin(X, axis=1)
        out[f"gfs_st_{var}_p25"] = np.nanpercentile(X, 25, axis=1)
        out[f"gfs_st_{var}_p75"] = np.nanpercentile(X, 75, axis=1)
        out[f"gfs_st_{var}_std"] = np.nanstd(X, axis=1)
    return _attach(df, out, [c for c in out.columns if c.startswith("gfs_st_")])


def add_gfs_layers_kr(df):
    g = _load("gfs_layers.parquet")
    return _attach(df, g, [c for c in g.columns if c.endswith("_kr")])


def add_gfs_layers_regions(df):
    g = _load("gfs_layers.parquet")
    return _attach(df, g, [c for c in g.columns if c.startswith("gfs_")])


def add_gefs_regions(df):
    g = _load("gefs_kr.parquet")
    return _attach(df, g, [c for c in g.columns if c.startswith("gefs_")])


def add_gefs_kr(df):
    g = _load("gefs_kr.parquet")
    return _attach(df, g, [c for c in g.columns if c.startswith("gefs_") and c.endswith("_kr")])
