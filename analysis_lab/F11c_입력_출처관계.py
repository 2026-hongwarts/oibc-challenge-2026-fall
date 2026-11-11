"""입력 단계 출처 어긋남: 지역 예보(A·B, 지점 평균)와 GFS(상자 평균)의 하루 한낮 값 관계가 평가 기간에 바뀌었나.
학습 기간(2021-06 ~ 2025-05)에서 월별 선형 관계를 맞추고, 그 잔차의 월 평균을 해마다 비교한다(z = 평가 − 과거 해 평균 / 과거 해 표준편차)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from oibc_common import WORK

V = ["wa_cloud_r17", "wb_cloud_r17", "wa_humidity_r17", "wb_uv_idx_r17", "wa_temp_r17", "wb_temp_r17", "wb_rel_hum_r17"]


def local(path):
    b = pd.read_parquet(path, columns=["wa_station", "wb_station", "ts"] + V)
    b = b[(b.ts.dt.hour >= 11) & (b.ts.dt.hour <= 15)]
    a = b.drop_duplicates(["wa_station", "ts"]).groupby("ts")[[c for c in V if c.startswith("wa_")]].mean()
    bb = b.drop_duplicates(["wb_station", "ts"]).groupby("ts")[[c for c in V if c.startswith("wb_")]].mean()
    x = a.join(bb)
    return x.groupby(x.index.normalize()).mean()


loc = pd.concat([local(WORK / "base.parquet"), local(WORK / "base_test.parquet")])
loc = loc[~loc.index.duplicated()]
g = pd.read_parquet(WORK / "gfs_kr.parquet", columns=["ts", "gfs_dswrf_kr", "gfs_tcdc_kr"])
g["ts"] = pd.to_datetime(g.ts).dt.tz_convert("Asia/Seoul")
g = g[(g.ts.dt.hour >= 11) & (g.ts.dt.hour <= 15)]
g = g.groupby(g.ts.dt.normalize())[["gfs_dswrf_kr", "gfs_tcdc_kr"]].mean()
d = loc.join(g, how="inner").dropna()
d["mo"] = d.index.month
d["yr"] = np.where(d.index.month >= 6, d.index.year, d.index.year - 1)
train = d.yr <= 2024
pairs = [("wa_cloud_r17", "gfs_tcdc_kr"), ("wb_cloud_r17", "gfs_tcdc_kr"), ("wb_uv_idx_r17", "gfs_dswrf_kr"),
         ("wa_humidity_r17", "gfs_tcdc_kr"), ("wa_cloud_r17", "wb_cloud_r17"), ("wa_temp_r17", "wb_temp_r17"),
         ("wa_humidity_r17", "wb_rel_hum_r17")]
order = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
for yv, xv in pairs:
    res = pd.Series(np.nan, index=d.index)
    for mo, gg in d.groupby("mo"):
        t = gg[train.loc[gg.index]]
        A = np.c_[np.ones(len(t)), t[xv]]
        coef = np.linalg.lstsq(A, t[yv], rcond=None)[0]
        res.loc[gg.index] = gg[yv] - (coef[0] + coef[1] * gg[xv])
    m = res.groupby([d.yr, d.mo]).mean().unstack(0)
    past = m[[c for c in m.columns if c <= 2024 and c >= 2021]]
    z = (m[2025] - past.mean(1)) / past.std(1)
    print(f"{yv} ~ {xv}: 평가 잔차(z) " + " ".join(f"{k}월 {m.loc[k, 2025]:+.1f}({z[k]:+.1f})" for k in order))
