"""E5 구성(G1/G2) 방법 비교 — ⑤ 묶음·구성·분석

    python analysis_lab/E5_구성비교.py                 기준 모델(champion) 오차로 비교
    python analysis_lab/E5_구성비교.py --name E4-08x2

필요한 예측 파일 (train_site_model.py 로 만든다)
    outputs/preds/<name>.parquet            가상 평가연도 2024-06 ~ 2025-05   (--base champion)
    outputs/preds/<name>_cut2023.parquet    2023-06 ~ 2024-05               (--base champion --cut 2023-06-01)
    outputs/preds/<name>_cut2022.parquet    2022-06 ~ 2023-05               (--base champion --cut 2022-06-01)

비교하는 방법
    REF    sites.csv 참조 구성(current_group A→G1)
    CH     chloe 방식: 한 해 오차로 local_search (참조 + 무작위 6개 시작점 중 최선)
    A      같은 local_search, 오차표를 두 해 이상 이어 붙여서
    B      규칙: 같은 B 관측소(wb_station) 발전소를 G1/G2 에 설비용량 반반으로 나눔 (데이터 안 봄)
    B+     B 에서 출발한 local_search (A 와 같은 데이터)
    B+c    B+ 와 같되 규칙을 지키는 이동만: 신규(2024-06 이후 편입) 발전소가 2곳 이상인 관측소는
           두 묶음에 신규가 1곳 이상씩, 묶음별 설비용량 ≥ 20,000kW + 100kW 여유

공정한 비교: 한 해를 빼고 나머지로 구성을 정한 뒤, 뺀 해에서 참조 구성 대비 벌금 변화를 잰다.
    CH 는 남은 해 중 가장 최근 한 해로, A·B+ 는 남은 두 해로 정한다.
벌금 = Σ SMP·k·(|G1 오차| + |G2 오차|), 0.10 곱하기 전. 편입 전이라 실적이 없는 시각의 오차는 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
from oibc_common import (CAP_MIN_FRAC, CAP_MIN_KW, OUTPUTS, PREDS, TZ, WORK,  # noqa: E402
                         group_penalty, load_sites, load_smp, local_search)

YEARS = {  # 이름: (예측 파일 접미사, 구간 시작)
    "2024-25": ("", "2024-06-01"),
    "2023-24": ("_cut2023", "2023-06-01"),
    "2022-23": ("_cut2022", "2022-06-01"),
}
NEW_FROM = pd.Timestamp("2024-06-01", tz=TZ)


def residuals(name: str, start: str, sites: list[str], actual: pd.DataFrame):
    """(시각 × 발전소) 오차 행렬 R(예측 − 실적, kWh), 가중치 w(SMP·k), 시각 인덱스."""
    lo = pd.Timestamp(f"{start} 01:00", tz=TZ)
    hi = lo + pd.DateOffset(years=1)
    a = actual[(actual.ts >= lo) & (actual.ts < hi)]
    p = pd.read_parquet(PREDS / f"{name}.parquet")
    p["site_id"] = p.site_id.astype(str)
    p["ts"] = pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    m = a.merge(p, on=["site_id", "ts"])
    if len(m) != len(a):
        raise SystemExit(f"{name}: 실적 행 {len(a):,} 중 예측이 있는 행 {len(m):,}")
    m["e"] = m.pred_kwh - m.energy
    R = m.pivot(index="ts", columns="site_id", values="e").reindex(columns=sites).fillna(0.0)
    w = load_smp("train").set_index("ts").w.reindex(R.index).to_numpy()
    return R.to_numpy(), w, R.index, int(a.site_id.nunique())


def rule_split(s: pd.DataFrame, key: str = "wb_station") -> np.ndarray:
    """B: 관측소마다 설비용량이 큰 발전소부터, 그 관측소 안에서 용량이 적은 쪽 묶음에 넣는다.
    관측소 안에서 같으면 전체 용량이 적은 쪽. 결과는 G1 여부(bool)."""
    x = np.zeros(len(s), bool)
    tot = {True: 0.0, False: 0.0}
    for _, g in s.groupby(key, sort=True):
        loc = {True: 0.0, False: 0.0}
        for i in g.sort_values("capacity_kw", ascending=False).index:
            side = loc[True] < loc[False] or (loc[True] == loc[False] and tot[True] <= tot[False])
            x[i] = side
            loc[side] += s.capacity_kw[i]
            tot[side] += s.capacity_kw[i]
    return x


def align(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """G1/G2 이름만 바뀐 같은 구성을 맞춘다(벌금은 이름을 바꿔도 같다)."""
    return x if (x == ref).mean() >= 0.5 else ~x


def search(R, w, starts, cap, ref):
    """시작점마다 local_search, 최선과 전체 결과(안정성용)를 돌려준다."""
    xs = [align(local_search(R, w, x0, cap), ref) for x0 in starts]
    best = min(xs, key=lambda x: group_penalty(R, x, w))
    return best, xs


def local_search_c(R, w, x0, cap, st_new: np.ndarray, margin_kw: float = 100.0,
                   max_iter: int = 500) -> np.ndarray:
    """B+c: local_search 와 같되 두 규칙을 지키는 이동만 허용한다.
    1) 신규 발전소가 2곳 이상인 관측소는 두 묶음에 신규가 1곳 이상씩 남는다.
    2) 묶음별 설비용량 ≥ 20,000kW + margin_kw.
    st_new: 발전소별 관측소 번호(신규가 2곳 이상인 관측소의 신규 발전소만, 나머지는 -1)."""
    lo = (CAP_MIN_KW + margin_kw) / cap.sum()
    x = x0.copy()
    e1, e2 = R[:, x].sum(1), R[:, ~x].sum(1)
    cur = w @ (np.abs(e1) + np.abs(e2))
    ids = np.unique(st_new[st_new >= 0])
    for _ in range(max_iter):
        s = np.where(x, -1.0, 1.0)
        cand = w @ (np.abs(e1[:, None] + R * s) + np.abs(e2[:, None] - R * s))
        frac = (cap[x].sum() + s * cap) / cap.sum()
        bad = (frac < lo) | (frac > 1 - lo)
        for k in ids:  # 이 관측소에서 한쪽에 신규가 1곳만 남았으면 그 발전소는 못 옮김
            m = st_new == k
            for side in (True, False):
                only = m & (x == side)
                if only.sum() == 1:
                    bad |= only
        cand[bad] = np.inf
        i = int(np.argmin(cand))
        if cand[i] >= cur - 1e-9:
            break
        x[i] = ~x[i]
        e1, e2, cur = e1 + R[:, i] * s[i], e2 - R[:, i] * s[i], cand[i]
    return x


def check(s: pd.DataFrame, x: np.ndarray, is_new: np.ndarray, cap: np.ndarray) -> tuple[float, float, int]:
    """(G1 kW, G2 kW, 신규 2곳 이상 관측소 중 한쪽에 몰린 곳 수)."""
    sw = s.assign(g1=x, new=is_new)
    nw = sw[sw.new].groupby("wb_station").g1.agg(["size", "sum"])
    nw = nw[nw["size"] >= 2]
    bad = int(((nw["sum"] == 0) | (nw["sum"] == nw["size"])).sum())
    return float(cap[x].sum()), float(cap[~x].sum()), bad


def pct(R, w, x, ref) -> float:
    return (group_penalty(R, x, w) / group_penalty(R, ref, w) - 1) * 100


def main() -> None:
    ap = argparse.ArgumentParser(description="E5 구성 방법 비교")
    ap.add_argument("--name", default=None, help="예측 이름 (기본: champion)")
    ap.add_argument("--starts", type=int, default=6, help="무작위 시작점 수 (chloe 와 같게 6)")
    args = ap.parse_args()
    name = args.name or (PREDS / "champion.txt").read_text(encoding="utf-8").strip()

    s = load_sites().reset_index(drop=True)
    s["site_id"] = s.site_id.astype(str)
    sites = s.site_id.tolist()
    cap = s.capacity_kw.to_numpy()
    ref = (s.current_group == "A").to_numpy()

    actual = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy"])
    actual["site_id"] = actual.site_id.astype(str)
    actual = actual.dropna(subset=["energy"])
    first = actual.groupby("site_id").ts.min()
    is_new = s.site_id.map(first >= NEW_FROM).fillna(True).to_numpy()

    # chloe 와 같은 무작위 시작점 (rng 시드 0)
    rng = np.random.default_rng(0)

    def rand_split():
        while True:
            x = rng.random(len(cap)) < 0.5
            f = cap[x].sum() / cap.sum()
            if CAP_MIN_FRAC <= f <= 1 - CAP_MIN_FRAC:
                return x

    starts = [ref] + [rand_split() for _ in range(args.starts)]
    xb = rule_split(s)
    cnt = s[is_new].groupby("wb_station").size()
    multi = cnt[cnt >= 2].index
    codes = {k: i for i, k in enumerate(multi)}
    st_new = np.array([codes.get(st, -1) if nw else -1 for st, nw in zip(s.wb_station, is_new)])

    D = {}
    print(f"[{name}] 세 해 오차표")
    for y, (suf, start) in YEARS.items():
        R, w, idx, n = residuals(f"{name}{suf}", start, sites, actual)
        D[y] = (R, w, idx)
        lb = float(w @ np.abs(R.sum(1)))
        print(f"  {y}: 실적 있는 발전소 {n}곳, 참조 구성 벌금 {0.1 * group_penalty(R, ref, w):,.0f}원, "
              f"하한(한 묶음) 대비 +{(group_penalty(R, ref, w) / lb - 1) * 100:.2f}%")

    # 1) chloe 결과 재현: 2024-25 홀짝 달 + 다른 해
    R, w, idx = D["2024-25"]
    month = idx.month.to_numpy()
    print("\n1. chloe 방식 재현 (2024-25, 참조 구성 대비)")
    for fit, ev, label in [(month % 2 == 1, month % 2 == 0, "홀수→짝수"),
                           (month % 2 == 0, month % 2 == 1, "짝수→홀수")]:
        x, _ = search(R, w * fit, starts, cap, ref)
        print(f"  {label}: {pct(R, w * ev, x, ref):+.2f}%")
    x_ch_full, _ = search(R, w, starts, cap, ref)
    R2, w2, _ = D["2023-24"]
    print(f"  1년 전체로 정한 구성 → 2023-24: {pct(R2, w2, x_ch_full, ref):+.2f}%")
    cf = OUTPUTS / "compositions" / "chloe_E4-08x2.csv"
    if cf.exists():
        c = pd.read_csv(cf).set_index("site_id").group.reindex(sites)
        xc = align((c == "G1").to_numpy(), ref)
        print(f"  chloe 파일과 재현 구성이 같은 발전소: {int((xc == x_ch_full).sum())}/253, "
              f"chloe 파일 → 2023-24: {pct(R2, w2, xc, ref):+.2f}%")

    # 2) 한 해 빼고 비교
    print("\n2. 한 해 빼고 비교: 뺀 해에서 참조 구성 대비 벌금 변화 (낮을수록 좋음)")
    order = list(YEARS)  # 최근 순
    res = {m: {} for m in ["CH", "A", "B", "B+", "B+c"]}
    stab = {}
    folds = {"B+": [], "B+c": []}
    for test in order:
        rest = [y for y in order if y != test]
        Rt, wt, _ = D[test]
        Rc, wc, _ = D[rest[0]]                      # CH: 남은 해 중 최근 한 해
        Ra = np.vstack([D[y][0] for y in rest])     # A·B+: 남은 두 해
        wa = np.concatenate([D[y][1] for y in rest])
        x_ch, _ = search(Rc, wc, starts, cap, ref)
        x_a, xs_a = search(Ra, wa, starts, cap, ref)
        x_bp = align(local_search(Ra, wa, xb, cap), ref)
        x_bpc = align(local_search_c(Ra, wa, xb, cap, st_new), ref)
        folds["B+"].append(x_bp)
        folds["B+c"].append(x_bpc)
        for m, x in [("CH", x_ch), ("A", x_a), ("B", xb), ("B+", x_bp), ("B+c", x_bpc)]:
            res[m][test] = pct(Rt, wt, x, ref)
        stab[test] = np.mean([xx == x_a for xx in xs_a], axis=0)
        print(f"  뺀 해 {test} (CH←{rest[0]}, A·B+←{'+'.join(rest)}): "
              + "  ".join(f"{m} {res[m][test]:+.2f}%" for m in res))
    print("  평균: " + "  ".join(f"{m} {np.mean(list(res[m].values())):+.2f}%" for m in res))

    # 3) 최종 구성: 세 해 전부로 A, B+ / 그리고 점검
    Rall = np.vstack([D[y][0] for y in order])
    wall = np.concatenate([D[y][1] for y in order])
    x_a3, xs_a3 = search(Rall, wall, starts, cap, ref)
    x_bp3 = align(local_search(Rall, wall, xb, cap), ref)
    x_bpc3 = align(local_search_c(Rall, wall, xb, cap, st_new), ref)
    out = OUTPUTS / "compositions"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n3. 최종 후보 점검 (신규 2곳 이상 관측소 {len(multi)}개)")
    finals = [("CH", "CH(2024-25 전체)", x_ch_full), ("A", "A(세 해)", x_a3), ("B", "B", xb),
              ("Bp", "B+(세 해)", x_bp3), ("Bpc", "B+c(세 해, 규칙 지킴)", x_bpc3)]
    for _, label, x in finals:
        g1, g2, bad = check(s, x, is_new, cap)
        same = {y: pct(D[y][0], D[y][1], x, ref) for y in order}
        print(f"  {label}: 옮긴 발전소 {int((x != ref).sum())}곳, G1 {g1:,.1f}kW / G2 {g2:,.1f}kW, "
              f"신규가 한쪽에 몰린 관측소 {bad}개, 같은 데이터 벌금 변화(낙관치) "
              + " / ".join(f"{v:+.2f}%" for v in same.values()))
    print("\n4. 안정성")
    v = np.mean([xx == x_a3 for xx in xs_a3], axis=0)
    print(f"  A(세 해): 시작점 7개 결과가 모두 같은 발전소 {int((v == 1).sum())}/253, 평균 일치 {v.mean() * 100:.1f}%")
    for test, v in stab.items():
        print(f"  A(뺀 해 {test}): 시작점 7개가 모두 같은 발전소 {int((v == 1).sum())}/253")
    for m, xs in folds.items():
        xf = x_bp3 if m == "B+" else x_bpc3
        allx = xs + [xf]
        agree = np.mean([xx == xf for xx in allx], axis=0)
        print(f"  {m}: 뺀 해 3개 + 최종, 4번 모두 같은 묶음인 발전소 {int((agree == 1).sum())}/253, "
              f"B 와 같은 발전소 {int((xf == xb).sum())}/253")

    for tag, _, x in finals:
        pd.DataFrame({"site_id": sites, "group": np.where(x, "G1", "G2")}).to_csv(
            out / f"E5-{tag}_{name}.csv", index=False)
    print(f"\n저장 outputs/compositions/E5-{{CH,A,B,Bp,Bpc}}_{name}.csv")


if __name__ == "__main__":
    main()
