"""F14 — 지금 가진 예측 파일만으로(재학습 없음) 두 가지를 본다 (2026-11-11, docs/피처_근거_분석.md §11).

A. 시각 평활: 벌금의 약 55% 가 하루 안 시각 배분 오차(F13a)다. 구름이 오는 시각을 틀리는 것이 원인이면,
   묶음 예측을 앞뒤 시각과 섞어 부드럽게 하는 것만으로 줄어들 수 있다.
   묶음 청천 기준 c_t = Σ 용량 × Haurwitz 청천(cos 천정각, 학습 스크립트의 CSI 기준과 같음)으로 나눈 비율을
   같은 날 안에서 (w, 1 − 2w, w) 로 섞고 다시 c_t 를 곱한다. 해 진 시각(c_t = 0)은 그대로 0.
   w 는 남은 두 해에서 정해 그해에 적용(한 해 빼고 정하기). 판정: 세 해 모두 벌금 감소.
B. 구성원 빼기: E12-10c 다섯 구성원의 같은 비중 평균 vs 하나씩 뺀 넷의 평균(같은 방식으로 이 스크립트 안에서 섞음).

참조 구성, 전체 발전소(지금 판정 규칙과 같은 조건). 학습 기간 SMP·k 가중.
    python analysis_lab/F14_시각평활_구성원.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from compare_years import actual
from evaluate import load_composition
from oibc_common import PENALTY_RATE, PREDS, WORK, load_smp

MEMBERS = ["E11-12x2", "E11-11x2", "G5bT1x2", "S3x2", "V07T1x2"]
FOLDS = [("2024-25", None, ""), ("2023-24", "2023-06-01", "_cut2023"), ("2022-23", "2022-06-01", "_cut2022")]
WS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def main():
    comp = load_composition(None)
    w = load_smp("train").set_index("ts").w
    base = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "cosz", "capacity_kw"])
    base["site_id"] = base.site_id.astype(str)
    cz = base.cosz.clip(lower=0.01)
    base["clr"] = np.where(base.cosz > 0.01, 1.098 * cz * np.exp(-0.057 / cz), 0.0) * base.capacity_kw
    res = {}
    for y, cut, tag in FOLDS:
        a = actual(cut)
        keys = a[["site_id", "ts"]]
        preds = []
        for m in MEMBERS:
            p = pd.read_parquet(PREDS / f"{m}{tag}.parquet")
            p["site_id"] = p.site_id.astype(str)
            p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
            preds.append(keys.merge(p, on=["site_id", "ts"], how="left").pred_kwh.fillna(0).to_numpy())
        d = a.merge(base[["site_id", "ts", "clr"]], on=["site_id", "ts"], how="left")
        d["g"] = d.site_id.map(comp)
        P = np.vstack(preds)                                   # 구성원 × 행
        def group_sum(v):
            s = pd.DataFrame({"g": d.g, "ts": d.ts, "v": v}).pivot_table(index="ts", columns="g", values="v", aggfunc="sum", fill_value=0)
            return s[["G1", "G2"]]
        A = group_sum(d.energy.to_numpy())
        C = group_sum(d.clr.fillna(0).to_numpy())
        ww = A.index.map(w).to_numpy(dtype="float64")
        day = (A.index - pd.Timedelta(hours=1)).normalize()
        res[y] = dict(A=A.to_numpy(), C=C.to_numpy(), w=ww, day=np.asarray(day), idx=A.index,
                      G={k: group_sum(P[i]).to_numpy() for i, k in enumerate(MEMBERS)})

    def pen(r, F):
        return PENALTY_RATE * float(r["w"] @ np.abs(F - r["A"]).sum(1))

    def smooth(r, F, wt):
        if wt == 0:
            return F
        C = r["C"]
        ratio = np.clip(np.where(C > 0, F / np.where(C > 0, C, 1), 0.0), 0, 1.5)
        # 그날 청천 기준 최대의 5% 미만인 시각(해 뜰 녘·질 녘)은 비율이 불안정 → 이웃으로 쓰지 않는다
        cmax = pd.DataFrame(C).groupby(r["day"]).transform("max").to_numpy()
        ok = C >= 0.05 * cmax
        prev = np.vstack([ratio[:1], ratio[:-1]])
        nxt = np.vstack([ratio[1:], ratio[-1:]])
        same_p = np.r_[False, r["day"][1:] == r["day"][:-1]][:, None] & np.vstack([ok[:1], ok[:-1]]) & ok
        same_n = np.r_[r["day"][:-1] == r["day"][1:], False][:, None] & np.vstack([ok[1:], ok[-1:]]) & ok
        prev = np.where(same_p, prev, ratio)                  # 다른 날·해 진 시각은 자기 값으로
        nxt = np.where(same_n, nxt, ratio)
        s = wt * prev + (1 - 2 * wt) * ratio + wt * nxt
        return np.where(C > 0, s * C, F)

    print("A. 시각 평활 (E12-10c 다섯 구성원 같은 비중, 참조 구성) — 벌금 변화 %")
    ens = {y: np.mean([r["G"][m] for m in MEMBERS], axis=0) for y, r in res.items()}
    base_pen = {y: pen(r, ens[y]) for y, r in res.items()}
    tab = pd.DataFrame({y: {wt: (pen(r, smooth(r, ens[y], wt)) / base_pen[y] - 1) * 100 for wt in WS} for y, r in res.items()})
    print(tab.round(2).to_string())
    for y in res:
        others = [o for o in res if o != y]
        wt = min(WS, key=lambda v: sum(tab.loc[v, o] for o in others))
        print(f"  {y}: 다른 두 해로 정한 w = {wt:.2f} → {tab.loc[wt, y]:+.2f}%")

    print("\nB. 구성원 하나씩 빼기 (같은 비중, 참조 구성) — 다섯 평균 대비 벌금 변화 %")
    for m in MEMBERS:
        rest = [k for k in MEMBERS if k != m]
        row = [f"{(pen(r, np.mean([r['G'][k] for k in rest], axis=0)) / base_pen[y] - 1) * 100:+.2f}%" for y, r in res.items()]
        print(f"  {m:9s} 빼기: " + " / ".join(row))
    print("\n  구성원 한낮(11~15시) 묶음 오차 상관(2024-25, G1+G2)")
    r = res["2024-25"]
    mid = np.asarray(pd.DatetimeIndex(r["idx"]).hour.isin([11, 12, 13, 14, 15]))
    E = pd.DataFrame({m: (r["G"][m] - r["A"])[mid].ravel() for m in MEMBERS})
    print(E.corr().round(3).to_string())


if __name__ == "__main__":
    main()
