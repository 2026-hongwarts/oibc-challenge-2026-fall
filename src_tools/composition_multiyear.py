"""여러 해 오차로 구성 정하기 (설명회 슬 26 "찾은 구성이 내년에도 통할까? · 연도별 안정성") + 늦게 편입된 큰 발전소 민감도.

    python src_tools/composition_multiyear.py --preds E12-10cx2
        → 검증 표, outputs/compositions/{이름}_my.csv (세 해 오차로 정한 구성)

오차 R: 2024-25 = {이름}, 2023-24 = {이름}_cut2023, 2022-23 = {이름}_cut2022 (각 해를 그 전 2년으로 학습한 예측).
편입 전이라 실적이 없는 시각은 오차 0 (optimize_composition 과 같음).

검증 (미리 정한 판정): 해마다 그 해를 남겨 두고
    여러 해판 = 남은 두 해 오차를 시각 방향으로 이어 붙여 최적화
    한 해판   = 남은 두 해 중 최근 해 하나로 최적화 (지금 방식과 같은 조건)
  → 남겨 둔 해의 벌금을 참조 구성 대비로 비교. 세 해 모두에서 여러 해판이 낮으면 채택.
최종 구성(채택 시) = 세 해 오차를 이어 붙여 최적화.
A3 민감도: 최종 구성에서 늦게 편입된 큰 발전소(기본 PV011, PV021)를 반대 묶음으로 옮겼을 때
    실적이 있는 시각만의 벌금 변화(2024-25). 용량 하한에 걸리면 표시.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from oibc_common import CAP_MIN_FRAC, OUTPUTS, PREDS, group_penalty, load_sites, local_search
from optimize_composition import residuals
from train_site_model import cut_window

YEARS = [("2024-25", ""), ("2023-24", "2023-06-01"), ("2022-23", "2022-06-01")]


def fit(Rs, ws, cap, ref, rng, starts=6):
    R = np.vstack(Rs)
    w = np.concatenate(ws)

    def rand_split():
        while True:
            x = rng.random(len(cap)) < 0.5
            f = cap[x].sum() / cap.sum()
            if CAP_MIN_FRAC <= f <= 1 - CAP_MIN_FRAC:
                return x
    xs = [local_search(R, w, x0, cap) for x0 in [ref] + [rand_split() for _ in range(starts)]]
    return min(xs, key=lambda x: group_penalty(R, x, w))


def main() -> None:
    ap = argparse.ArgumentParser(description="여러 해 오차로 구성 + 늦게 편입된 큰 발전소 민감도")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--late", nargs="*", default=["PV011", "PV021"])
    args = ap.parse_args()
    s = load_sites()
    sites = s.site_id.tolist()
    cap = s.capacity_kw.to_numpy()
    ref = (s.current_group == "A").to_numpy()
    data = {}
    for y, c in YEARS:
        name = args.preds if not c else f"{args.preds}_cut{c[:4]}"
        R, w, idx = residuals(name, None if not c else cut_window(c), sites)
        data[y] = (R, w)
        print(f"{y}: {R.shape[0]:,}시각, 실적이 한 번이라도 있는 발전소 {(np.abs(R).sum(0) > 0).sum()}곳")

    print("\n검증: 그 해를 남겨 두고 정한 구성으로 그 해 벌금 (참조 구성 대비)")
    win = []
    for k, (y, _) in enumerate(YEARS):
        others = [o for o, _ in YEARS if o != y]
        recent = others[0]                                    # YEARS 는 최근 해부터
        Rh, wh = data[y]
        base = group_penalty(Rh, ref, wh)
        rng = np.random.default_rng(0)
        x_my = fit([data[o][0] for o in others], [data[o][1] for o in others], cap, ref, rng)
        rng = np.random.default_rng(0)
        x_1 = fit([data[recent][0]], [data[recent][1]], cap, ref, rng)
        d_my = (group_penalty(Rh, x_my, wh) / base - 1) * 100
        d_1 = (group_penalty(Rh, x_1, wh) / base - 1) * 100
        win.append(d_my < d_1)
        print(f"  {y} 남김: 한 해판({recent}) {d_1:+.2f}%  여러 해판({' + '.join(others)}) {d_my:+.2f}%  "
              f"→ {'여러 해판 나음' if d_my < d_1 else '한 해판 나음'} ({d_my - d_1:+.2f}%p)")
    ok = all(win)
    print(f"판정: {'채택 (세 해 모두 여러 해판이 낮음)' if ok else '불채택'}")

    rng = np.random.default_rng(0)
    x = fit([data[y][0] for y, _ in YEARS], [data[y][1] for y, _ in YEARS], cap, ref, rng)
    g = np.where(x, "G1", "G2")
    print(f"\n세 해로 정한 구성: G1 {cap[x].sum():,.1f}kW / G2 {cap[~x].sum():,.1f}kW")
    out = OUTPUTS / "compositions" / f"{args.preds}_my.csv"
    pd.DataFrame({"site_id": sites, "group": g}).to_csv(out, index=False)
    print(f"저장 {out.relative_to(OUTPUTS.parent)}")

    # A3: 늦게 편입된 큰 발전소 민감도 (지금 쓰는 구성 기준)
    cur_path = OUTPUTS / "compositions" / f"{args.preds}.csv"
    if cur_path.exists():
        cur = pd.read_csv(cur_path).set_index("site_id").group.reindex(sites).eq("G1").to_numpy()
        R, w = data["2024-25"]
        print(f"\n늦게 편입된 큰 발전소 민감도 (구성 {cur_path.name}, 2024-25, 그 발전소 실적이 있는 시각만)")
        for sid in args.late:
            if sid not in sites:
                continue
            i = sites.index(sid)
            obs = np.abs(R[:, i]) > 0
            xf = cur.copy()
            xf[i] = ~xf[i]
            fr = cap[xf].sum() / cap.sum()
            note = "" if CAP_MIN_FRAC <= fr <= 1 - CAP_MIN_FRAC else "  (용량 하한 위반이라 실제로는 못 옮김)"
            b = group_penalty(R, cur, w * obs)
            d = (group_penalty(R, xf, w * obs) / b - 1) * 100
            print(f"  {sid} {cap[i]:,.1f}kW, 지금 {'G1' if cur[i] else 'G2'}, 실적 시각 {obs.mean():.0%}: "
                  f"반대 묶음으로 옮기면 그 시각 벌금 {d:+.2f}%{note}")


if __name__ == "__main__":
    main()
