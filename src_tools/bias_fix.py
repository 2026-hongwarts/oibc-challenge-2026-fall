"""체계적 편향 사후 보정 (11/11 파이프라인 점검 결과) — 재학습 없이 예측 파일에 구간별 배율을 곱한다.

    python src_tools/bias_fix.py --preds E12-10ctx2 --kind hour       → E12-10cthx2
    python src_tools/bias_fix.py --preds E12-10ctx2 --kind cloudgap   → E12-10ctgx2

근거(세 해 같은 방향의 치우침, 점검 보고):
    hour     : 15시 과대 +2.1 / +4.8 / +2.4%, 19 ~ 20시 과소
    cloudgap : A 운량이 GFS 운량보다 높을 때(A 가 더 흐리다고 할 때) 과소 −1.3 / −0.7 / −1.6%
방법(미리 정함):
    구간마다 배율 f = Σ실측 / Σ예측(발전소×시각 합, 고도 > 0 행). 검증 해마다 **그 해를 뺀 나머지 두 해**의 같은 후보 예측·실측으로
    f 를 구해 그 해에 곱한다(교차 해). 평가 기간 파일에는 세 해 모두로 구한 f. f 는 [0.9, 1.1] 로 자른다.
    hour 구간 = ts 의 시(6 ~ 20시, 그 밖은 1). cloudgap 구간 = (A 17시 발행 운량 − GFS 전체 평균 운량) 을
    [−∞, −40, −20, 0, 20, 40, ∞] 로 나눈 6칸(GFS 값이 없으면 1).
판정(미리 정함): 세 해 모두 원래 예측보다 벌금이 낮아야 채택.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from oibc_common import PREDS, PREDS_TEST, ROOT, TZ, WORK

CUTS = {"": "2024-06-01", "_cut2023": "2023-06-01", "_cut2022": "2022-06-01"}
EDGES = [-np.inf, -40, -20, 0, 20, 40, np.inf]


def keyed(p: pd.DataFrame, kind: str, side: pd.DataFrame) -> pd.Series:
    m = p.merge(side, on=["site_id", "ts"], how="left")
    if kind == "hour":
        h = m.ts.dt.hour
        return pd.Series(np.where((h >= 6) & (h <= 20), h, -1), index=p.index)
    g = pd.cut(m.wa_cloud_r17 - m.gfs_tcdc_kr, EDGES, labels=False)
    return pd.Series(np.where(g.isna(), -1, g), index=p.index)


def main() -> None:
    ap = argparse.ArgumentParser(description="체계적 편향 사후 보정(교차 해)")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--kind", choices=["hour", "cloudgap"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--neutral", action="store_true", help="배율을 그 두 해 전체 수준(Σ실측/Σ예측)으로 나눠 수준은 그대로 두고 모양만 고침")
    ap.add_argument("--test-src", default=None, help="평가 기간 원본 예측 이름(기본 --preds 와 같음). 예: 검증은 E12-10ctx2, 평가 기간은 E12-10cftx2")
    ap.add_argument("--test-out", default=None, help="평가 기간 결과 이름(기본 --out 과 같음)")
    args = ap.parse_args()
    tag = {"hour": "h", "cloudgap": "g"}[args.kind] + ("n" if args.neutral else "")
    out = args.out or (args.preds[:-2] + f"{tag}x2" if args.preds.endswith("x2") else args.preds + tag)

    a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "elev", "wa_cloud_r17"])
    a["site_id"] = a.site_id.astype(str)
    a["ts"] = pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    g = pd.read_parquet(WORK / "gfs_kr.parquet", columns=["ts", "gfs_tcdc_kr"])
    g["ts"] = pd.to_datetime(g.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    side = a[["site_id", "ts", "wa_cloud_r17"]].merge(g, on="ts", how="left")

    def load(path):
        p = pd.read_parquet(path)
        p["site_id"] = p.site_id.astype(str)
        p["ts"] = pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
        return p

    # 각 해의 (구간, 예측합, 실측합)
    stats = {}
    for suf, v in CUTS.items():
        p = load(PREDS / f"{args.preds}{suf}.parquet")
        s, e = pd.Timestamp(v, tz=TZ) + pd.Timedelta(hours=1), pd.Timestamp(v, tz=TZ) + pd.DateOffset(years=1) + pd.Timedelta(hours=1)
        m = p.merge(a[["site_id", "ts", "energy", "elev"]], on=["site_id", "ts"], how="inner")
        m = m[(m.ts >= s) & (m.ts < e) & m.energy.notna() & (m.elev > 0)]
        m["k"] = keyed(m[["site_id", "ts"]], args.kind, side).to_numpy()
        stats[suf] = m.groupby("k")[["pred_kwh", "energy"]].sum()

    def factors(use: list[str]) -> pd.Series:
        t = sum(stats[u] for u in use)
        f = (t.energy / t.pred_kwh)
        if args.neutral:                  # 수준은 추세 보정에 맡기고 구간 사이 모양만
            f = f / (t.energy.sum() / t.pred_kwh.sum())
        f = f.clip(0.9, 1.1)
        f.loc[-1] = 1.0
        return f

    report = {}
    for suf in CUTS:
        f = factors([u for u in CUTS if u != suf])
        p = load(PREDS / f"{args.preds}{suf}.parquet")
        k = keyed(p, args.kind, side)
        p["pred_kwh"] = p.pred_kwh * k.map(f).fillna(1.0).to_numpy()
        dst = PREDS / f"{out}{suf}.parquet"
        p.to_parquet(dst, index=False)
        report[suf or "2024-25"] = {int(i): round(float(x), 4) for i, x in f.items()}
        dst.with_suffix(".json").write_text(json.dumps({"exp": dst.stem, "bias_fix_of": f"{args.preds}{suf}", "kind": args.kind,
                                                        "factors": report[suf or "2024-25"], "created": datetime.now().isoformat(timespec="seconds")},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    src = PREDS_TEST / f"{args.test_src or args.preds}.parquet"
    if src.exists():
        f = factors(list(CUTS))
        p = load(src)
        side_t = None
        if args.kind == "cloudgap":            # 평가 기간 A 운량은 base_test 에서
            bt = pd.read_parquet(WORK / "base_test.parquet", columns=["site_id", "ts", "wa_cloud_r17"])
            bt["site_id"] = bt.site_id.astype(str)
            bt["ts"] = pd.to_datetime(bt.ts).dt.tz_convert(TZ).dt.as_unit("ns")
            side_t = bt.merge(g, on="ts", how="left")
        k = keyed(p, args.kind, side if side_t is None else side_t)
        p["pred_kwh"] = p.pred_kwh * k.map(f).fillna(1.0).to_numpy()
        dst = PREDS_TEST / f"{args.test_out or out}.parquet"
        p.to_parquet(dst, index=False)
        report["test(세 해)"] = {int(i): round(float(x), 4) for i, x in f.items()}
        dst.with_suffix(".json").write_text(json.dumps({"exp": dst.stem, "bias_fix_of": args.preds, "kind": args.kind,
                                                        "factors": report["test(세 해)"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    for k_, v_ in report.items():
        print(f"{k_}: {v_}")


if __name__ == "__main__":
    main()
