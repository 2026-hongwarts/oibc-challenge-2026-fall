"""여러 실험의 예측을 가중 평균한다 (앙상블).

    python src_tools/blend.py --out E5-01x2 E4-08x2 E4-11x2                     가상 평가연도, 같은 비중
    python src_tools/blend.py --out E5-01x2 E4-08x2 E4-11x2 --weights 2 1       비중 2:1
    python src_tools/blend.py --out E5-01x2 E4-08x2 E4-11x2 --cut 2023-06-01   다른 해 (각 실험의 _cut2023 파일)
    python src_tools/blend.py --out E5-01x2 E4-08x2 E4-11x2 --mode test        실제 평가 기간 → outputs/preds_test/

각 실험이 같은 (site_id, ts) 행을 덮어야 한다. 비중은 합이 1이 되게 맞춘다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from oibc_common import PREDS, PREDS_TEST, ROOT, WORK


def path_of(name: str, mode: str, cut: str | None):
    if mode == "test":
        return PREDS_TEST / f"{name}.parquet"
    return PREDS / (f"{name}_cut{cut[:4]}.parquet" if cut else f"{name}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description="예측 가중 평균")
    ap.add_argument("names", nargs="+", help="섞을 실험 이름들")
    ap.add_argument("--out", required=True, help="결과 실험 이름")
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--cut", default=None, metavar="YYYY-MM-DD", help="다른 해 검증 파일끼리 섞기")
    args = ap.parse_args()

    w = np.array(args.weights if args.weights else [1.0] * len(args.names), dtype=float)
    if len(w) != len(args.names):
        raise SystemExit("--weights 개수가 실험 수와 다름")
    w = w / w.sum()
    base = None
    for name, wi in zip(args.names, w):
        p = pd.read_parquet(path_of(name, args.mode, args.cut))
        p["site_id"] = p.site_id.astype(str)
        p = p.set_index(["site_id", "ts"]).pred_kwh.rename(name)
        base = p.to_frame() if base is None else base.join(p, how="outer")
    if base.isna().any().any():
        raise SystemExit(f"실험들이 덮는 행이 서로 다름 (빈칸 {int(base.isna().sum().sum())}개)")
    out = (base.to_numpy() @ w).clip(min=0)
    res = base.index.to_frame(index=False).assign(pred_kwh=out)

    out_name = f"{args.out}_cut{args.cut[:4]}" if args.cut else args.out
    path = path_of(args.out, args.mode, args.cut)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(path, index=False)
    meta = {"exp": out_name, "blend": dict(zip(args.names, w.round(4).tolist())), "mode": args.mode,
            "cut": args.cut, "created": datetime.now().isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 {path.relative_to(ROOT)}  비중 {meta['blend']}")

    if args.mode == "valid" and args.cut:
        from evaluate import evaluate_preds, read_preds
        from train_site_model import cut_window
        start, end = cut_window(args.cut)
        a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy"])
        a = a[(a.ts >= start) & (a.ts < end)].reset_index(drop=True)
        a["site_id"] = a.site_id.astype(str)
        r = evaluate_preds(read_preds(path), actual=a)
        print(f"[{out_name}] 다른 해 검증 {start:%Y-%m}~  벌금 {r['penalty']:,.0f}원  11~15시 {r['peak']:,.0f}원")
    elif args.mode == "valid":
        from evaluate import report
        report(path)


if __name__ == "__main__":
    main()
