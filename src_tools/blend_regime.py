"""조건을 만족하는 시각에서만 두 예측을 섞는다 (흐림 예보 전용 모델 등).

    python src_tools/blend_regime.py --base E7-00x2 --spec X15x2 --where "wa_cloud_r17 >= 90" --out X15bx2
    python src_tools/blend_regime.py ... --cut 2023-06-01      다른 해 (각 실험의 _cut2023 파일)
    python src_tools/blend_regime.py ... --mode test           실제 평가 기간

조건을 만족하는 행: (1 − w)·기준 + w·전용, 나머지 행: 기준 그대로. w 기본 0.5.
조건은 공통 테이블(base.parquet / base_test.parquet)의 열로 쓴다(예보 열만 — 실적 금지).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from blend import path_of
from oibc_common import ROOT, WORK


def main() -> None:
    ap = argparse.ArgumentParser(description="조건부 섞기")
    ap.add_argument("--base", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--where", required=True, help='pandas query. 예: "wa_cloud_r17 >= 90"')
    ap.add_argument("--w", type=float, default=0.5, help="조건 구간에서 전용 모델 비중")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--cut", default=None, metavar="YYYY-MM-DD")
    args = ap.parse_args()

    if re.search(r"\b(energy|cf|is_dead)\b", args.where):
        raise SystemExit("조건에 실적(energy, cf, is_dead)을 쓸 수 없음")
    b = pd.read_parquet(path_of(args.base, args.mode, args.cut))
    s = pd.read_parquet(path_of(args.spec, args.mode, args.cut))
    for d in (b, s):
        d["site_id"] = d.site_id.astype(str)
    cols = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", args.where)) - {"and", "or", "not", "in"})
    src = pd.read_parquet(WORK / ("base_test.parquet" if args.mode == "test" else "base.parquet"),
                          columns=["site_id", "ts"] + cols)
    src["site_id"] = src.site_id.astype(str)
    cond = src.query(args.where)[["site_id", "ts"]].assign(_c=True)
    m = b.merge(s.rename(columns={"pred_kwh": "spec"}), on=["site_id", "ts"], how="left", validate="1:1")
    if m.spec.isna().any():
        raise SystemExit("전용 모델 예측이 기준 예측의 행을 다 덮지 않음")
    m = m.merge(cond, on=["site_id", "ts"], how="left")
    c = m._c.notna().to_numpy()
    m["pred_kwh"] = np.where(c, (1 - args.w) * m.pred_kwh + args.w * m.spec, m.pred_kwh).clip(min=0)
    res = m[["site_id", "ts", "pred_kwh"]]

    out_name = f"{args.out}_cut{args.cut[:4]}" if args.cut else args.out
    path = path_of(args.out, args.mode, args.cut)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(path, index=False)
    meta = {"exp": out_name, "regime_blend": {"base": args.base, "spec": args.spec, "where": args.where, "w": args.w},
            "mode": args.mode, "cut": args.cut, "share_rows": float(c.mean()),
            "created": datetime.now().isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 {path.relative_to(ROOT)}  조건 행 비율 {c.mean():.1%}")


if __name__ == "__main__":
    main()
