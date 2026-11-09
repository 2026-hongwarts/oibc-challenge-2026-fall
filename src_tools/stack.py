"""스태킹(결합) — 여러 모델의 예측을 대회 벌금이 가장 작아지는 비중으로 섞는다 (참고: 태양광 5위의 선형 메타 모델).

    python src_tools/stack.py M1 M2 M3 [--comp 구성.csv] [--by-hour] [--out 이름]

- 목표: 묶음(G1·G2) × 시각 합계의 0.1 × SMP × k × |Σ b_m P_m − 실적| 합을 최소화하는 비중 b (b ≥ 0, Σ b = 1).
  선형계획법(scipy linprog, HiGHS)으로 정확히 푼다. 사이트 단위 오차가 아니라 채점식 그대로.
- --by-hour: 시간대 블록(6~9, 10~12, 13~15, 16~20시)마다 다른 비중.
- 과적합 확인(교차): 세 해 중 두 해로 비중을 정하고 나머지 한 해에서 평가 → 같은 비중(1/M)·가장 좋은 단일 모델과 비교.
- 최종 비중: 세 해 전체로 다시 맞춘 값. 리더보드는 쓰지 않는다(규정 §2).
- --out 을 주면 세 해 예측 파일(outputs/preds/이름{,_cut2023,_cut2022}.parquet)과 비중(.json)을 저장 → blend.py --weights 로 평가 기간에도 같은 비중.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from evaluate import load_composition
from oibc_common import PREDS, WORK, load_smp
from train_site_model import cut_window

YEARS = [("2024-25", None), ("2023-24", "2023-06-01"), ("2022-23", "2022-06-01")]
BLOCKS = [(6, 9), (10, 12), (13, 15), (16, 20)]


def group_table(models: list[str], cut: str | None, comp: pd.Series) -> pd.DataFrame:
    base = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "split"])
    base["site_id"] = base.site_id.astype(str)
    if cut:
        s, e = cut_window(cut)
        a = base[(base.ts >= s) & (base.ts < e)]
    else:
        a = base[base.split == "valid"]
    d = a[["site_id", "ts", "energy"]]
    for m in models:
        p = pd.read_parquet(PREDS / (f"{m}_cut{cut[:4]}.parquet" if cut else f"{m}.parquet"))
        p["site_id"] = p.site_id.astype(str)
        d = d.merge(p.rename(columns={"pred_kwh": m}), on=["site_id", "ts"], how="inner")
    d["g"] = d.site_id.map(comp)
    g = d.groupby(["g", "ts"])[["energy"] + models].sum().reset_index()
    g = g.merge(load_smp("train")[["ts", "w"]], on="ts", how="left")
    g["hour"] = g.ts.dt.hour
    return g


def solve(tabs: list[pd.DataFrame], models: list[str]) -> np.ndarray:
    """min Σ w |P b − a|, b ≥ 0, Σ b = 1  →  변수 [b (M), e (N)], e ≥ ±(P b − a)."""
    P = np.vstack([t[models].to_numpy() for t in tabs])
    a = np.concatenate([t.energy.to_numpy() for t in tabs])
    w = np.concatenate([t.w.to_numpy() for t in tabs])
    keep = w > 0
    P, a, w = P[keep], a[keep], w[keep]
    n, m = P.shape
    sc = 1.0 / max(a.max(), 1.0)                          # 수치 안정을 위해 크기 맞춤
    P, a = P * sc, a * sc
    c = np.concatenate([np.zeros(m), w / w.mean()])
    from scipy.sparse import csr_matrix, hstack, identity, vstack
    I = identity(n, format="csr")
    Ps = csr_matrix(P)
    A_ub = vstack([hstack([Ps, -I]), hstack([-Ps, -I])]).tocsr()
    b_ub = np.concatenate([a, -a])
    A_eq = csr_matrix(np.concatenate([np.ones(m), np.zeros(n)])[None, :])
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (m + n), method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    return r.x[:m]


def pen(t: pd.DataFrame, models: list[str], b: np.ndarray) -> float:
    return float(0.1 * (t.w * np.abs(t[models].to_numpy() @ b - t.energy)).sum())


def pen_by_hour(t, models, bh: dict) -> float:
    tot = 0.0
    for (lo, hi), b in bh.items():
        s = t[t.hour.between(lo, hi)]
        tot += pen(s, models, b)
    rest = t[~t.hour.between(BLOCKS[0][0], BLOCKS[-1][1])]
    return tot + pen(rest, models, np.full(len(models), 1 / len(models)))


def fit(tabs, models, by_hour: bool):
    if not by_hour:
        return solve(tabs, models)
    return {blk: solve([t[t.hour.between(*blk)] for t in tabs], models) for blk in BLOCKS}


def evaluate(t, models, wts, by_hour: bool) -> float:
    return pen_by_hour(t, models, wts) if by_hour else pen(t, models, wts)


def main() -> None:
    ap = argparse.ArgumentParser(description="채점식 직접 최소화 스태킹")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--comp", default=None, help="구성 CSV (기본: 참조 구성)")
    ap.add_argument("--by-hour", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    comp = load_composition(args.comp)
    M = args.models
    tabs = {y: group_table(M, c, comp) for y, c in YEARS}
    eq = np.full(len(M), 1 / len(M))

    print(f"모델 {M}  구성 {args.comp or '참조'}  시간대별 {args.by_hour}")
    print(f"{'평가 해':8s} {'같은 비중':>14s} {'최선 단일':>14s} {'스태킹(교차)':>14s}  다른 두 해로 정한 비중")
    wins = []
    for y, _ in YEARS:
        others = [tabs[o] for o, _ in YEARS if o != y]
        wts = fit(others, M, args.by_hour)
        t = tabs[y]
        p_eq, p_st = pen(t, M, eq), evaluate(t, M, wts, args.by_hour)
        p_best = min(pen(t, M, np.eye(len(M))[i]) for i in range(len(M)))
        wins.append(p_st < p_eq)
        show = {k: np.round(v, 2).tolist() for k, v in wts.items()} if args.by_hour else np.round(wts, 3).tolist()
        print(f"{y:8s} {p_eq:14,.0f} {p_best:14,.0f} {p_st:14,.0f}  ({(p_st / p_eq - 1) * 100:+.2f}% vs 같은 비중)  {show}")
    final = fit(list(tabs.values()), M, args.by_hour)
    print("교차 평가에서 세 해 모두 같은 비중보다 좋음:" , all(wins))
    print("최종 비중(세 해 전체):", {k: np.round(v, 3).tolist() for k, v in final.items()} if args.by_hour else dict(zip(M, np.round(final, 3))))

    if args.out and not args.by_hour:
        for y, c in YEARS:
            ps = []
            for m in M:
                p = pd.read_parquet(PREDS / (f"{m}_cut{c[:4]}.parquet" if c else f"{m}.parquet"))
                p["site_id"] = p.site_id.astype(str)
                ps.append(p.set_index(["site_id", "ts"]).pred_kwh.rename(m))
            d = pd.concat(ps, axis=1)
            out = d.index.to_frame(index=False).assign(pred_kwh=(d.to_numpy() @ final).clip(min=0))
            path = PREDS / (f"{args.out}_cut{c[:4]}.parquet" if c else f"{args.out}.parquet")
            out.to_parquet(path, index=False)
            path.with_suffix(".json").write_text(json.dumps({"exp": path.stem, "stack": dict(zip(M, final.round(4).tolist())),
                                                             "comp": args.comp}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"저장 outputs/preds/{args.out}{{,_cut2023,_cut2022}}.parquet  → 평가 기간: blend.py --weights {' '.join(f'{v:.4f}' for v in final)}")


if __name__ == "__main__":
    main()
