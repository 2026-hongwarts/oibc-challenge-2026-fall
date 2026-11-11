"""본 발전소만의 수준 치우침: 해마다 최적 배율, 다른 해에서 정한 배율을 옮겨 쓴 효과, 학습 이력 길이별 치우침, 실적의 해별 맑은 한낮 발전율(노화)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from compare_years import actual
from evaluate import evaluate_preds, read_preds
from oibc_common import PREDS, WORK

ENS = sys.argv[1] if len(sys.argv) > 1 else "E11-10T1x2"
FOLDS = [("2024-25", None, "2024-06-01"), ("2023-24", "2023-06-01", "2023-06-01"), ("2022-23", "2022-06-01", "2022-06-01")]
base = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "capacity_kw", "elev", "wa_cloud_r17", "hour"])
base["site_id"] = base.site_id.astype(str)
AL = np.round(np.arange(0.95, 1.031, 0.005), 3)
curves, info = {}, {}
for y, cut, start in FOLDS:
    s = pd.Timestamp(f"{start} 01:00", tz="Asia/Seoul")
    hist = base[(base.ts < s) & (base.ts >= s - pd.DateOffset(years=2))].groupby("site_id").ts.agg(["min", "size"])
    seen = set(hist.index)
    p = read_preds(PREDS / (f"{ENS}_cut{cut[:4]}.parquet" if cut else f"{ENS}.parquet"))
    a = actual(cut)
    a = a[a.site_id.isin(seen)].reset_index(drop=True)          # 실제 평가 조건 모사: 본 발전소만으로 묶음을 만든다
    p = p[p.site_id.isin(seen)].reset_index(drop=True)
    curves[y] = pd.Series({al: evaluate_preds(p.assign(pred_kwh=p.pred_kwh * al), actual=a)["penalty"] for al in AL})
    # 이력 길이별 치우침(본 발전소, 11~15시)
    m = p.merge(a, on=["site_id", "ts"])
    m = m[m.site_id.isin(seen) & (m.ts.dt.hour >= 11) & (m.ts.dt.hour <= 15)]
    days = (s - hist["min"]).dt.days.reindex(m.site_id).to_numpy()
    m["이력"] = pd.cut(days, [0, 365, 600, 2000], labels=["1년 미만", "1 ~ 1.6년", "1.6년 이상(2년 가득)"])
    info[y] = m.groupby("이력", observed=True).apply(lambda g: g.pred_kwh.sum() / g.energy.sum())
print(f"기준 {ENS}. 본 발전소만으로 만든 묶음(처음 보는 발전소 제외), 예측 α 배, 참조 구성 벌금 변화 %\n")
t = pd.DataFrame({y: (c / c[1.0] - 1) * 100 for y, c in curves.items()})
print(t.loc[[0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0, 1.01]].round(2).to_string())
for y, c in curves.items():
    print(f"  {y}: 최적 α {c.idxmin():.3f} ({(c.min() / c[1.0] - 1) * 100:+.2f}%)")
print("\n다른 두 해의 최적 α 평균을 옮겨 쓴 효과 (그해 자료를 안 보고 정한 α):")
for y in curves:
    others = [curves[o].idxmin() for o in curves if o != y]
    al = AL[np.argmin(np.abs(AL - np.mean(others)))]
    print(f"  {y}: α {al:.3f} → {(curves[y][al] / curves[y][1.0] - 1) * 100:+.2f}%")
print("\n본 발전소 11 ~ 15시 예측÷실적, 학습 기간 안 이력 길이별")
print(pd.DataFrame(info).round(3).to_string())

