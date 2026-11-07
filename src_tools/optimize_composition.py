"""구성(G1/G2) 최적화 — 기준 모델의 가상 평가연도 오차로 발전소 배정을 다시 정한다.

    python src_tools/optimize_composition.py                       기준 모델(champion) 오차로
    python src_tools/optimize_composition.py --preds E5-01x2       다른 실험 오차로
    → outputs/compositions/<예측이름>.csv (site_id,group), 그리고 검증 결과 출력

방법: 오차 R(시각 × 발전소, 예측 − 실적, kWh)에서 발전소 1개를 반대 묶음으로 옮기는 이동 중 최선을 반복
    (oibc_common.local_search, 묶음별 설비용량 20,000kW 이상 유지). 시작점은 참조 구성 + 무작위 여러 개.
검증 (같은 데이터로 맞추고 평가하면 효과가 부풀려지므로)
    1. 홀수 달로 최적화 → 짝수 달로 평가, 반대도
    2. 1년 전체로 정한 최종 구성을 다른 해(2023-24) 오차에 그대로 적용
최종 구성은 1년 전체로 최적화한 것. 편입 전이라 실적이 없는 시각의 오차는 0으로 둔다.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from oibc_common import (CAP_MIN_FRAC, OUTPUTS, PREDS, WORK, group_penalty, load_sites, load_smp,
                         local_search)


def residuals(name: str, window: tuple[pd.Timestamp, pd.Timestamp] | None, sites: list[str]):
    """(시각 × 발전소) 오차 행렬, 가중치 w(SMP·k), 시각 인덱스."""
    p = pd.read_parquet(PREDS / f"{name}.parquet")
    p["site_id"] = p.site_id.astype(str)
    p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
    a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "split"])
    a = a[a.split == "valid"] if window is None else a[(a.ts >= window[0]) & (a.ts < window[1])]
    a["site_id"] = a.site_id.astype(str)
    m = a.merge(p, on=["site_id", "ts"])
    if len(m) != len(a):
        raise SystemExit(f"{name}: 실적 행 {len(a):,} 중 예측이 있는 행 {len(m):,}")
    m["e"] = m.pred_kwh - m.energy
    R = m.pivot(index="ts", columns="site_id", values="e").reindex(columns=sites).fillna(0.0)
    w = load_smp("train").set_index("ts").w.reindex(R.index).to_numpy()
    return R.to_numpy(), w, R.index


def best_of(R, w, starts, cap):
    xs = [local_search(R, w, x0, cap) for x0 in starts]
    return min(xs, key=lambda x: group_penalty(R, x, w))


def main() -> None:
    ap = argparse.ArgumentParser(description="구성 최적화")
    ap.add_argument("--preds", default=None, help="오차를 낼 실험 이름 (기본: champion)")
    ap.add_argument("--starts", type=int, default=6, help="무작위 시작점 수")
    args = ap.parse_args()
    name = args.preds or (PREDS / "champion.txt").read_text(encoding="utf-8").strip()

    s = load_sites()
    sites = s.site_id.tolist()
    cap = s.capacity_kw.to_numpy()
    ref = (s.current_group == "A").to_numpy()
    rng = np.random.default_rng(0)

    def rand_split():
        while True:
            x = rng.random(len(cap)) < 0.5
            f = cap[x].sum() / cap.sum()
            if CAP_MIN_FRAC <= f <= 1 - CAP_MIN_FRAC:
                return x

    R, w, idx = residuals(name, None, sites)
    month = idx.month.to_numpy()
    lb = float(w @ np.abs(R.sum(1)))
    over = lambda x, ww, base: (group_penalty(R, x, ww) / base - 1) * 100
    starts = [ref] + [rand_split() for _ in range(args.starts)]
    print(f"[{name}] 가상 평가연도 발전소 {R.shape[1]}곳 × {R.shape[0]:,}시각")
    print(f"  하한(한 포트폴리오) 대비: 참조 구성 +{over(ref, w, lb):.2f}%")

    print("\n검증 1: 절반 달로 정하고 나머지 달로 평가 (참조 구성 대비 벌금 변화)")
    for fit, ev, label in [(month % 2 == 1, month % 2 == 0, "홀수→짝수"), (month % 2 == 0, month % 2 == 1, "짝수→홀수")]:
        x = best_of(R, w * fit, starts, cap)
        d = (group_penalty(R, x, w * ev) / group_penalty(R, ref, w * ev) - 1) * 100
        print(f"  {label}: {d:+.2f}%  (옮긴 발전소 {int((x != ref).sum())}곳)")

    x = best_of(R, w, starts, cap)
    g = np.where(x, "G1", "G2")
    capg = {k: float(cap[g == k].sum()) for k in ("G1", "G2")}
    print(f"\n최종 구성(1년 전체로 최적화): 옮긴 발전소 {int((x != ref).sum())}곳, "
          f"G1 {capg['G1']:,.1f}kW / G2 {capg['G2']:,.1f}kW, 같은 해 벌금 변화 "
          f"{(group_penalty(R, x, w) / group_penalty(R, ref, w) - 1) * 100:+.2f}% (낙관치)")

    other = f"{name}_cut2023"
    if (PREDS / f"{other}.parquet").exists():
        from train_site_model import cut_window
        R2, w2, _ = residuals(other, cut_window("2023-06-01"), sites)
        d = (group_penalty(R2, x, w2) / group_penalty(R2, ref, w2) - 1) * 100
        print(f"검증 2: 최종 구성을 다른 해(2023-24) 오차에 적용 → 참조 구성 대비 {d:+.2f}%")
    else:
        print(f"검증 2: {other} 가 없어 건너뜀")

    out = OUTPUTS / "compositions"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"site_id": sites, "group": g}).to_csv(out / f"{name}.csv", index=False)
    print(f"저장 outputs/compositions/{name}.csv")


if __name__ == "__main__":
    main()
