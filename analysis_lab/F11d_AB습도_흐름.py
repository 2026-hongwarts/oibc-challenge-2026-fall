"""A 습도 − B 습도 관계가 평가 기간에 바뀌었나: 해마다·주마다, 그리고 제공사 안(습도 ~ 운량·기온) 관계도."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from oibc_common import WORK

V = ["wa_cloud_r17", "wa_humidity_r17", "wa_temp_r17", "wb_cloud_r17", "wb_rel_hum_r17", "wb_temp_r17", "wb_dew_point_r17",
     "wa_humidity_r10", "wb_rel_hum_r10"]


def daily(path):
    b = pd.read_parquet(path, columns=["wa_station", "wb_station", "ts"] + V)
    a = b.drop_duplicates(["wa_station", "ts"]).groupby("ts")[[c for c in V if c.startswith("wa_")]].mean()
    bb = b.drop_duplicates(["wb_station", "ts"]).groupby("ts")[[c for c in V if c.startswith("wb_")]].mean()
    x = a.join(bb)
    return x.groupby((x.index - pd.Timedelta(hours=1)).normalize()).mean()      # 하루 전체 평균(지점 평균)


d = pd.concat([daily(WORK / "base.parquet"), daily(WORK / "base_test.parquet")])
d = d[~d.index.duplicated()].dropna()
d["mo"] = d.index.month
d["yr"] = np.where(d.index.month >= 6, d.index.year, d.index.year - 1)
train = (d.yr <= 2024) & (d.yr >= 2021)


def resid(yv, xs):
    r = pd.Series(np.nan, index=d.index)
    for mo, gg in d.groupby("mo"):
        t = gg[train.loc[gg.index]]
        A = np.c_[np.ones(len(t)), t[xs].to_numpy()]
        c = np.linalg.lstsq(A, t[yv], rcond=None)[0]
        r.loc[gg.index] = gg[yv] - np.c_[np.ones(len(gg)), gg[xs].to_numpy()] @ c
    return r


for yv, xs in [("wa_humidity_r17", ["wb_rel_hum_r17"]), ("wa_humidity_r10", ["wb_rel_hum_r10"]),
               ("wa_humidity_r17", ["wa_cloud_r17", "wa_temp_r17"]), ("wb_rel_hum_r17", ["wb_cloud_r17", "wb_temp_r17"]),
               ("wb_rel_hum_r17", ["wb_temp_r17", "wb_dew_point_r17"]), ("wa_humidity_r17", ["wb_temp_r17", "wb_dew_point_r17"])]:
    r = resid(yv, xs)
    yr = r.groupby(d.yr).agg(["mean", lambda s: (s.groupby(d.mo.loc[s.index]).mean() < 0).sum()])
    yr.columns = ["해 평균 잔차", "음수인 달 수"]
    wk = r["2025-04-01":"2025-08-31"].groupby(pd.Grouper(freq="2W")).mean()
    print(f"\n{yv} ~ {' + '.join(xs)}")
    print("  " + " | ".join(f"{int(k)}-{str(int(k) + 1)[2:]}: {v['해 평균 잔차']:+.2f} (음수 {int(v['음수인 달 수'])}/12)" for k, v in yr.iterrows()))
    print("  2주 평균(2025-04 ~ 08): " + " ".join(f"{i:%m-%d} {v:+.1f}" for i, v in wk.items()))
