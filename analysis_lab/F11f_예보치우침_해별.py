"""F11f — 예보(17시) − 실측의 해별 평균 치우침(학습 기간만). 어느 제공사 예보가 해마다 흘러가는지 본다 (docs/피처_근거_분석.md §8).
실측은 진단(학습 보조)에만 쓰고 모델 입력에는 넣지 않는다(규정 §6)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from oibc_common import DATA

for prov, fv, extra in [("wa", "wa_humidity", ["wa_cloud", "wa_temp"]), ("wb", "wb_rel_hum", ["wb_cloud", "wb_temp", "wb_uv_idx"])]:
    st = f"{prov}_station"
    f = pd.read_parquet(DATA / "train" / f"forecast_{prov}.parquet", columns=[st, "round", "ts", fv] + extra)
    f = f[f["round"] == 17]
    o = pd.read_parquet(DATA / "train" / f"observed_{prov}.parquet", columns=[st, "ts", fv] + extra)
    for c in [fv] + extra:                                   # README 이상값 제거
        o[c] = o[c].where(o[c].between(-60, 15 if "uv" in c else 100))
    for x in (f, o):
        x["ts"] = pd.to_datetime(x.ts).dt.tz_convert("Asia/Seoul")
    o = o.groupby([st, "ts"], as_index=False).mean(numeric_only=True)
    m = f.merge(o, on=[st, "ts"], suffixes=("_f", "_o"))
    m = m[m.ts < pd.Timestamp("2025-06-01", tz="Asia/Seoul")]
    m["yr"] = np.where(m.ts.dt.month >= 6, m.ts.dt.year, m.ts.dt.year - 1)
    mid = m[(m.ts.dt.hour >= 11) & (m.ts.dt.hour <= 15)]
    print(f"\n{prov.upper()} 예보(17시) − 실측, 해별 평균 (전체 시각 / 11~15시)")
    for c in [fv] + extra:
        a = (m[c + "_f"] - m[c + "_o"]).groupby(m.yr).mean()
        b = (mid[c + "_f"] - mid[c + "_o"]).groupby(mid.yr).mean()
        print(f"  {c:12s} " + " | ".join(f"{int(k)}-{str(int(k) + 1)[2:]}: {a[k]:+.2f} / {b[k]:+.2f}" for k in b.index))
