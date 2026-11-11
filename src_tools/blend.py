"""여러 실험의 예측을 가중 평균한다 (앙상블).

    python src_tools/blend.py --out E5-01x2 E4-08x2 E4-11x2                     가상 평가연도, 같은 비중
    python src_tools/blend.py --out E5-01x2 E4-08x2 E4-11x2 --weights 2 1       비중 2:1
    python src_tools/blend.py --out E5-01x2 E4-08x2 E4-11x2 --cut 2023-06-01   다른 해 (각 실험의 _cut2023 파일)
    python src_tools/blend.py --out E5-01x2 E4-08x2 E4-11x2 --mode test        실제 평가 기간 → outputs/preds_test/

각 실험이 같은 (site_id, ts) 행을 덮어야 한다. 비중은 합이 1이 되게 맞춘다.

입력 빠짐 처리(기본, --no-mask 로 끔. 11/11 파이프라인 점검에서 찾은 버그):
    외부 예보(GFS·GEFS·GFS 이전 실행·에어로졸)를 쓰는 구성원은, 그 시각에 그 예보가 없으면(마감 뒤 공개로 뺀 파일 등)
    그 시각만 빼고 나머지 구성원의 비중을 다시 맞춘다. 예: 평가 기간 GEFS 빈 4일에 S3 예측이 다른 모델의 9~17% 로 무너짐
    (학습 기간에 GEFS 빈칸이 없어 빈칸 처리를 배운 적이 없음). 입력이 있는지는 예보 시점에 알 수 있으므로 규정 §6 과 무관.
    구성원이 쓰는 외부 예보는 그 실험 json 의 add(피처 함수) 목록으로 정한다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from oibc_common import PREDS, PREDS_TEST, ROOT, TZ, WORK

# 피처 함수 → (예보 표, 그 예보가 있는지 볼 열). 표의 시각은 04 ~ 21시(KST).
SOURCES = {"add_gfs_kr": ("gfs_kr.parquet", "gfs_dswrf_kr"), "add_gfs_regions": ("gfs_kr.parquet", "gfs_dswrf_kr"),
           "add_gfs_layers_kr": ("gfs_layers.parquet", "gfs_lcdc_kr"),
           "add_gfs_layers_regions": ("gfs_layers.parquet", "gfs_lcdc_kr"),
           "add_gefs_kr": ("gefs_kr.parquet", "gefs_dswrfm_kr"), "add_gefs_regions": ("gefs_kr.parquet", "gefs_dswrfm_kr"),
           "add_gefs3h_regions": ("gefs_kr.parquet", "gefs_dswrfm_kr"),
           "add_gfs18_regions": ("gfs18_kr.parquet", "g18_dswrf_kr"), "add_aer_regions": ("aer_kr.parquet", "aer_aod_kr"),
           "add_gfs_pwat": ("gfs_pwat.parquet", "gfs_pwat_kr")}
_MISS: dict = {}


def missing_ts(fname: str, col: str) -> pd.DatetimeIndex:
    """그 예보 표에서 값이 없는 04 ~ 21시 시각."""
    if (fname, col) not in _MISS:
        t = pd.read_parquet(WORK / fname, columns=["ts", col])
        t["ts"] = pd.to_datetime(t.ts).dt.tz_convert(TZ).dt.as_unit("ns")
        d0, d1 = t.ts.min().normalize(), t.ts.max().normalize()
        full = pd.date_range(d0 + pd.Timedelta(hours=4), d1 + pd.Timedelta(hours=21), freq="h")
        full = full[(full.hour >= 4) & (full.hour <= 21)]
        _MISS[(fname, col)] = full.difference(pd.DatetimeIndex(t.ts[t[col].notna()]))
    return _MISS[(fname, col)]


def member_sources(name: str, mode: str, cut: str | None) -> set:
    j = path_of(name, mode, cut).with_suffix(".json")
    if not j.exists():
        return set()
    meta = json.loads(j.read_text(encoding="utf-8"))
    if "blend" in meta:                       # 섞은 결과를 다시 섞는 경우는 빠짐 처리 없음
        return set()
    return {SOURCES[a.split(":")[1]] for a in meta.get("add", []) if a.split(":")[-1] in SOURCES}


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
    ap.add_argument("--no-mask", action="store_true", help="입력 빠짐 처리를 끈다(예전 방식)")
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
    W = np.tile(w, (len(base), 1))
    masked = {}
    if not args.no_mask:
        ts = pd.DatetimeIndex(base.index.get_level_values("ts")).tz_convert(TZ)
        for j, name in enumerate(args.names):
            for fname, col in member_sources(name, args.mode, args.cut):
                hit = ts.isin(missing_ts(fname, col))
                if hit.any():
                    W[hit, j] = 0.0
                    masked[name] = masked.get(name, 0) + int(hit.sum())
        allz = W.sum(1) == 0
        W[allz] = w                                # 모두 빠지면 원래 비중(생기지 않아야 함)
    W = W / W.sum(1, keepdims=True)
    out = (base.to_numpy() * W).sum(1).clip(min=0)
    if masked:
        print("입력 빠짐으로 뺀 행: " + ", ".join(f"{k} {v:,}" for k, v in masked.items()))
    res = base.index.to_frame(index=False).assign(pred_kwh=out)

    out_name = f"{args.out}_cut{args.cut[:4]}" if args.cut else args.out
    path = path_of(args.out, args.mode, args.cut)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(path, index=False)
    meta = {"exp": out_name, "blend": dict(zip(args.names, w.round(4).tolist())), "masked_rows": masked, "mode": args.mode,
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
