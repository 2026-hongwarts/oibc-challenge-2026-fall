"""출력제어 흔적의 크기: 묶음 단위(참조 구성) 벌금을 날 종류(봄·가을·그 밖 × 평일·휴일)로 나눔."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from compare_years import actual
from evaluate import group_frame, load_composition, penalty_frame, read_preds
from features_v4 import HOLIDAYS
from oibc_common import PREDS, load_smp

ENS = sys.argv[1] if len(sys.argv) > 1 else "E11-10T1x2"
hol = pd.DatetimeIndex(HOLIDAYS.astype("datetime64[ns]"))
smp = load_smp("train")
for y, cut in [("2024-25", None), ("2023-24", "2023-06-01"), ("2022-23", "2022-06-01")]:
    p = read_preds(PREDS / (f"{ENS}_cut{cut[:4]}.parquet" if cut else f"{ENS}.parquet"))
    m = penalty_frame(group_frame(actual(cut), p, load_composition()), smp)
    d = (m.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
    off = (d.dt.dayofweek >= 5) | d.isin(hol)
    mo = d.dt.month
    season = np.select([mo.isin([3, 4, 5]), mo.isin([9, 10, 11])], ["봄", "가을"], "그밖")
    m["cat"] = pd.Series(season, index=m.index) + np.where(off, "·휴일", "·평일")
    mid = m[(m.k == 3)]
    tot = m.pen.sum()
    out = []
    for c, g in m.groupby("cat"):
        gm = mid[mid.cat == c]
        dd = gm.groupby(gm.ts.dt.normalize()).agg(f=("fcst", "sum"), a=("actual", "sum"))
        out.append({"날": c, "날수": g.ts.dt.normalize().nunique(), "벌금 비중%": g.pen.sum() / tot * 100,
                    "한낮 예측÷실적": gm.fcst.sum() / gm.actual.sum(),
                    "과대 15%↑ 날 비율%": ((dd.f / dd.a) > 1.15).mean() * 100,
                    "한낮 과대 벌금 비중%": gm[gm.fcst > gm.actual].pen.sum() / max(gm.pen.sum(), 1) * 100})
    t = pd.DataFrame(out).set_index("날")
    print(f"\n{y} (벌금 {tot:,.0f})")
    print(t.round(2).to_string())
    # 낙관 상한: 봄 휴일 한낮만 그해 최적 배율로 고쳤다면
    sel = (m.cat == "봄·휴일") & (m.k == 3)
    best = min(np.arange(0.80, 1.01, 0.01), key=lambda a: (m.w[sel] * (m.fcst[sel] * a - m.actual[sel]).abs()).sum())
    new = m.pen.copy()
    new[sel] = 0.10 * m.w[sel] * (m.fcst[sel] * best - m.actual[sel]).abs()
    print(f"  봄 휴일 한낮만 배율 {best:.2f} 로 고쳤다면(같은 해에서 고른 상한): 벌금 {(new.sum() / tot - 1) * 100:+.2f}%")
