"""F12 — L1 손실의 묶음 합 치우침을 사후 배율로 고치면 얼마나 줄어드나 (2026-11-11, docs/피처_근거_분석.md §8).

배경(F11·이번 분석): 발전소별 L1 은 중앙값을 맞히는데, 벌금은 묶음 합에 매긴다. 맑음 쪽 예보에서는 분포가 아래로 꼬리가 길어
중앙값 > 평균 → 과대, 흐림 예보에서는 반대 → 과소. 검증 연도에서는 처음 보는 발전소(과소 예측)가 이 치우침을 가렸다.
실제 평가 기간에는 처음 보는 발전소가 없으므로 **본 발전소만으로 만든 묶음**으로 채점한다.

규칙(결과를 보기 전에 정함): 배율은 남은 두 해에서 정하고(두 해 벌금 합 최소) 그해에 적용 = 한 해 빼고 정하기.
    U  : 하나의 배율 α
    C3 : 그 행(발전소·시각)의 A 운량 예보(17시) 구간별 배율 3개 — ≤50 / 51 ~ 89 / ≥90
판정: 세 해 모두 벌금 감소.

    python analysis_lab/F12_L1_수준보정.py [앙상블 이름 = E11-10T1x2] [구성 csv(없으면 참조 구성)]
"""
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from compare_years import actual
from evaluate import load_composition, read_preds
from oibc_common import PENALTY_RATE, PREDS, WORK, load_smp

ENS = sys.argv[1] if len(sys.argv) > 1 else "E11-10T1x2"
COMP = sys.argv[2] if len(sys.argv) > 2 else None
FOLDS = [("2024-25", None, "2024-06-01"), ("2023-24", "2023-06-01", "2023-06-01"), ("2022-23", "2022-06-01", "2022-06-01")]
EDGES = [-1, 50, 89, 100]
GRID = np.round(np.arange(0.94, 1.061, 0.01), 2)


def fold_arrays(base, comp, w, y, cut, start):
    s = pd.Timestamp(f"{start} 01:00", tz="Asia/Seoul")
    seen = set(base.loc[(base.ts < s) & (base.ts >= s - pd.DateOffset(years=2)), "site_id"].unique())
    a = actual(cut)
    a = a[a.site_id.isin(seen)]
    p = read_preds(PREDS / (f"{ENS}_cut{cut[:4]}.parquet" if cut else f"{ENS}.parquet"))
    m = a.merge(p, on=["site_id", "ts"], how="left").merge(base[["site_id", "ts", "wa_cloud_r17"]], on=["site_id", "ts"], how="left")
    m["g"] = m.site_id.map(comp)
    m["b"] = pd.cut(m.wa_cloud_r17.fillna(50), EDGES, labels=False)
    P = m.pivot_table(index="ts", columns=["g", "b"], values="pred_kwh", aggfunc="sum", fill_value=0.0)
    A = m.pivot_table(index="ts", columns="g", values="energy", aggfunc="sum", fill_value=0.0)
    P = P.reindex(columns=pd.MultiIndex.from_product([["G1", "G2"], range(3)]), fill_value=0.0)
    ww = P.index.map(w).to_numpy(dtype="float64")
    return P.to_numpy().reshape(len(P), 2, 3), A[["G1", "G2"]].to_numpy(), ww


def pen(arr, al):
    P, A, w = arr
    return PENALTY_RATE * float(w @ np.abs((P * np.asarray(al)).sum(2) - A).sum(1))


def main():
    base = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "wa_cloud_r17"])
    base["site_id"] = base.site_id.astype(str)
    comp = load_composition(COMP)
    w = load_smp("train").set_index("ts").w
    arrs = {y: fold_arrays(base, comp, w, y, c, s) for y, c, s in FOLDS}
    p0 = {y: pen(a, (1, 1, 1)) for y, a in arrs.items()}
    share = {y: a[0].sum((0, 1)) / a[0].sum() for y, a in arrs.items()}
    print(f"{ENS}, 본 발전소만, {'참조 구성' if COMP is None else COMP}. 운량 구간 예측량 비중: "
          + " | ".join(f"{y} " + "/".join(f"{v * 100:.0f}%" for v in s) for y, s in share.items()))
    cands = {"U": [(g, g, g) for g in GRID], "C3": list(itertools.product(GRID, repeat=3))}
    for name, cs in cands.items():
        print(f"\n{name}")
        for y in arrs:
            others = [o for o in arrs if o != y]
            best = min(cs, key=lambda al: sum(pen(arrs[o], al) / p0[o] for o in others))
            own = min(cs, key=lambda al: pen(arrs[y], al))
            print(f"  {y}: 다른 두 해로 정한 배율 {best} → {(pen(arrs[y], best) / p0[y] - 1) * 100:+.2f}%"
                  f"   (그해 최적 {own} {(pen(arrs[y], own) / p0[y] - 1) * 100:+.2f}%, 낙관치)")
        allbest = min(cs, key=lambda al: sum(pen(arrs[o], al) / p0[o] for o in arrs))
        print(f"  세 해 모두로 정한 배율(평가 기간에 쓸 값): {allbest} → "
              + " / ".join(f"{(pen(arrs[o], allbest) / p0[o] - 1) * 100:+.2f}%" for o in arrs))


if __name__ == "__main__":
    main()
