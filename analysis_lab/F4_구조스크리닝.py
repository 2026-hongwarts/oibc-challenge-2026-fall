"""F4 구조 스크리닝 — 단일 변수로는 남은 신호가 없을 때, 구조적인 신호가 해를 건너 통하는지 본다.

    python analysis_lab/F4_구조스크리닝.py --name E4-08x2

1. 발전소별 지속 오차: 두 해에서 발전소별 잔차 중앙값을 배워 남은 해에 빼면 가중 절대 오차가 주는가
   (전부 빼기, 절반만 빼기)
2. 두 변수 조합 구간(5×5): 흐림 예보 구간에서 조합별 잔차 중앙값을 배워 남은 해에 적용
3. 묶음 배율: 묶음 합계 예측 × α 에서 벌금이 최소가 되는 α 가 해마다 비슷한가 (참조 구성)
잔차 = (예측 − 실적)/설비용량, 가중치 = 설비용량 × k. 범위: 태양고도 > 10°, 정지일 제외, 편입 180일 초과.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src_tools"))
from oibc_common import PREDS, WORK, load_sites, load_smp  # noqa: E402

spec = importlib.util.spec_from_file_location("f3", HERE / "F3_잔차스크리닝.py")
f3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f3)


def load_year(name: str, suffix: str, base: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(PREDS / f"{name}{suffix}.parquet")
    p["site_id"] = p.site_id.astype(str)
    p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
    d = base.merge(p, on=["site_id", "ts"]).sort_values(["site_id", "ts"]).reset_index(drop=True)
    d["res"] = (d.pred_kwh - d.energy) / d.capacity_kw
    d["wt"] = d.capacity_kw * np.where(d.hour.between(11, 15), 3, 1)
    d["age"] = (d.ts - d.start).dt.days
    return d


def wl1_change(r: pd.Series, wt: pd.Series, corr: pd.Series) -> float:
    return ((wt * (r - corr).abs()).sum() / (wt * r.abs()).sum() - 1) * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="E4-08x2")
    args = ap.parse_args()
    cols = (["site_id", "ts", "date", "month", "hour", "elev", "cosz", "energy", "capacity_kw", "is_dead", "start",
             "lat", "lon"] + [f"{v}_r{r}" for v in f3.RAW for r in (10, 17)])
    base = pd.read_parquet(WORK / "base.parquet", columns=cols)
    base["site_id"] = base.site_id.astype(str)
    years = {label: load_year(args.name, suffix, base) for label, suffix, _ in f3.YEARS}
    sub = {k: d[(d.elev > 10) & ~d.is_dead & (d.age > 180)].reset_index(drop=True) for k, d in years.items()}
    labels = list(sub)

    print("=== 1. 발전소별 지속 오차 (두 해로 배운 발전소별 잔차 중앙값을 남은 해에 적용) ===")
    for frac in (1.0, 0.5):
        out = []
        for h in labels:
            tr = pd.concat([sub[y] for y in labels if y != h])
            med = tr.groupby("site_id").res.median()
            d = sub[h]
            corr = d.site_id.map(med).fillna(0) * frac
            out.append(wl1_change(d.res, d.wt, corr))
        print(f"  보정 비율 {frac:.0%}: " + ", ".join(f"{h} {v:+.2f}%" for h, v in zip(labels, out)))
    # 같은 방향으로 계속 틀리는 발전소
    per = pd.DataFrame({h: sub[h].groupby("site_id").res.median() * 100 for h in labels})
    both = per.dropna()
    print(f"  발전소별 잔차 중앙값(%p)의 해 간 상관: "
          + ", ".join(f"{a}~{b} {both[a].corr(both[b]):+.2f}" for i, a in enumerate(labels) for b in labels[i + 1:]))

    print("\n=== 2. 흐림 예보(운량 A ≥ 90%)에서 두 변수 조합 5×5 구간 ===")
    pairs = [("wa_cloud_r17", "wa_humidity_r17"), ("wa_humidity_r17", "hour"), ("wa_humidity_r17", "elev"),
             ("wb_uv_idx_r17", "wa_humidity_r17"), ("wb_ceiling_r17", "wa_humidity_r17"),
             ("wb_precip_prob_r17", "wa_humidity_r17"), ("wb_cloud_r17", "wa_humidity_r17"),
             ("wa_temp_r17", "wa_humidity_r17"), ("month", "hour")]
    ov = {k: d[d.wa_cloud_r17 >= 90].reset_index(drop=True) for k, d in sub.items()}
    for a, b in pairs:
        out = []
        for h in labels:
            tr = pd.concat([ov[y] for y in labels if y != h])
            ea = np.unique(np.nanquantile(tr[a], np.linspace(0, 1, 6)))
            eb = np.unique(np.nanquantile(tr[b], np.linspace(0, 1, 6)))
            key = lambda d: (np.clip(np.searchsorted(ea, d[a], "right") - 1, 0, len(ea) - 2) * 10
                             + np.clip(np.searchsorted(eb, d[b], "right") - 1, 0, len(eb) - 2))
            med = tr.res.groupby(key(tr)).median()
            d = ov[h]
            corr = pd.Series(med.reindex(key(d)).fillna(0).to_numpy())
            out.append(wl1_change(d.res, d.wt, corr))
        print(f"  {a:20s} × {b:16s}: " + ", ".join(f"{h} {v:+.2f}%" for h, v in zip(labels, out))
              + f"  | 평균 {np.mean(out):+.2f}%")

    print("\n=== 3. 묶음 배율 α: 묶음 합계 예측 × α 의 벌금이 최소인 α (참조 구성) ===")
    comp = load_sites().set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    w = load_smp("train").set_index("ts").w
    alphas = np.round(np.arange(0.94, 1.061, 0.005), 3)
    for h, d in years.items():
        g = d.assign(group=d.site_id.map(comp)).groupby(["group", "ts"])[["pred_kwh", "energy"]].sum().reset_index()
        g["w"] = g.ts.map(w)
        pen = [float((g.w * (a * g.pred_kwh - g.energy).abs()).sum()) for a in alphas]
        best = alphas[int(np.argmin(pen))]
        p1 = pen[list(alphas).index(1.0)]
        print(f"  {h}: 최적 α {best:.3f}, 그때 벌금 변화 {(min(pen) / p1 - 1) * 100:+.2f}%  "
              f"(α 0.98: {(pen[list(alphas).index(0.98)] / p1 - 1) * 100:+.2f}%, 1.02: {(pen[list(alphas).index(1.02)] / p1 - 1) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
