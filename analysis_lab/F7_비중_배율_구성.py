"""F7 — 이미 만든 세 해 예측만으로(재학습 없이) 앙상블 비중, 전체 배율 α, 구성을 내부 검증으로 고른다.

    python analysis_lab/F7_비중_배율_구성.py

모델: E9-11x2(CatBoost), E9-12x2(XGBoost, CSI), E9-00x2(LightGBM). 모두 A 기압 포함.
묶음 합은 예측의 선형 결합이므로 모델별 묶음 합을 한 번 구해 두고 비중·배율을 격자로 훑는다.
판정: 세 해 모두 지금 후보(Cat 0.5 / XGB 0.5, α 1, E9-04x2 구성)보다 낮을 것. 리더보드는 쓰지 않는다(규정 §2).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
from oibc_common import OUTPUTS, PREDS, WORK, load_sites, load_smp  # noqa: E402
from train_site_model import cut_window  # noqa: E402

MODELS = ["E9-11x2", "E9-12x2", "E9-00x2"]
YEARS = [("2024-25", None), ("2023-24", "2023-06-01"), ("2022-23", "2022-06-01")]
COMPS = {"참조": load_sites().set_index("site_id").current_group.map({"A": "G1", "B": "G2"}),
         "E9-04x2": pd.read_csv(OUTPUTS / "compositions" / "E9-04x2.csv").set_index("site_id").group}


def group_sums() -> dict:
    base = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "split"])
    base["site_id"] = base.site_id.astype(str)
    w = load_smp("train")[["ts", "w"]]
    out = {}
    for y, cut in YEARS:
        if cut:
            s, e = cut_window(cut)
            a = base[(base.ts >= s) & (base.ts < e)]
        else:
            a = base[base.split == "valid"]
        d = a[["site_id", "ts", "energy"]]
        for m in MODELS:
            p = pd.read_parquet(PREDS / (f"{m}_cut{cut[:4]}.parquet" if cut else f"{m}.parquet"))
            p["site_id"] = p.site_id.astype(str)
            d = d.merge(p.rename(columns={"pred_kwh": m}), on=["site_id", "ts"], how="inner")
        for cname, comp in COMPS.items():
            g = d.assign(g=d.site_id.map(comp)).groupby(["g", "ts"])[["energy"] + MODELS].sum().reset_index()
            g = g.merge(w, on="ts", how="left")
            out[(y, cname)] = (g.energy.to_numpy(), g[MODELS].to_numpy(), g.w.to_numpy())
    return out


def pen(gs, b, alpha=1.0) -> float:
    a, P, w = gs
    return float(0.1 * (w * np.abs(alpha * (P @ b) - a)).sum())


def main() -> None:
    S = group_sums()
    cur = np.array([0.5, 0.5, 0.0])
    ref = {y: pen(S[(y, "E9-04x2")], cur) for y, _ in YEARS}
    print("지금 후보 (Cat 0.5 / XGB 0.5, α 1, E9-04x2 구성):", {y: f"{v:,.0f}" for y, v in ref.items()})

    print("\n[1] 구성: 참조 구성 대비 E9-04x2 구성 (지금 비중)")
    for y, _ in YEARS:
        r = pen(S[(y, "참조")], cur)
        print(f"  {y}: {ref[y] / r * 100 - 100:+.2f}%  (2024-25 는 구성을 정한 해라 낙관치)")

    print("\n[2] 비중 격자 (E9-04x2 구성, α 1) — 지금 후보 대비 %, 세 해 모두 낮으면 *")
    grid = [np.array(v) for v in itertools.product(np.arange(0, 1.01, 0.1), repeat=3) if abs(sum(v) - 1) < 1e-9]
    rows = []
    for b in grid:
        r = [pen(S[(y, "E9-04x2")], b) / ref[y] * 100 - 100 for y, _ in YEARS]
        rows.append((tuple(np.round(b, 1)), *r, np.mean(r), all(x < 0 for x in r)))
    t = pd.DataFrame(rows, columns=["Cat/XGB/LGB", "2024-25", "2023-24", "2022-23", "평균", "세 해 모두"])
    print(t.sort_values("평균").head(12).round(2).to_string(index=False))
    print("  2-모델(LGB 0)만:")
    print(t[[b[2] == 0 for b in t["Cat/XGB/LGB"]]].sort_values("Cat/XGB/LGB").round(2).to_string(index=False))

    print("\n[3] 전체 배율 α — 해마다 벌금이 가장 낮은 α (지금 비중, E9-04x2 구성)")
    alphas = np.round(np.arange(0.95, 1.051, 0.005), 3)
    for y, _ in YEARS:
        v = [pen(S[(y, "E9-04x2")], cur, a) for a in alphas]
        i = int(np.argmin(v))
        print(f"  {y}: 최적 α {alphas[i]:.3f} ({v[i] / ref[y] * 100 - 100:+.2f}%),  α 0.99 {pen(S[(y, 'E9-04x2')], cur, 0.99) / ref[y] * 100 - 100:+.2f}%,"
              f"  α 1.01 {pen(S[(y, 'E9-04x2')], cur, 1.01) / ref[y] * 100 - 100:+.2f}%")


if __name__ == "__main__":
    main()
