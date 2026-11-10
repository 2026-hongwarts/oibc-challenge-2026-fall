"""일출·일몰 시각 보정 (11/11 파이프라인 점검 3번) — 재학습 없이 예측 파일만 고친다.

    python src_tools/twilight_fix.py --preds E12-10cx2          → E12-10ctx2 (세 해 검증 + 평가 기간 파일이 있으면 그것도)

문제: 학습·예측은 구간 중앙(-30분) 태양 고도 > 0 인 행만 쓰고, 고도 ≤ 0 이면 예측을 0 으로 둔다.
    그런데 고도 −4 ~ 0° 행에도 발전이 있다(설비용량의 0.2 ~ 0.65%, 일출 직후·일몰 직전).
    발전량의 0.1% 지만 벌금의 약 0.4% 이고, 그 구간 편향이 −88 ~ −91%.
보정: 고도 −4 < elev ≤ 0 이고 예측이 0 인 행만, 그 해의 학습 기간(검증 시작 전 2년) 실측 발전율의 중앙값(고도 1° 구간별)
    × 설비용량으로 채운다. 학습 기간 실측만 쓰므로 규정 §6 을 지킨다. 다른 행은 그대로.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from oibc_common import PREDS, PREDS_TEST, ROOT, TZ, WORK

FOLDS = {"": ("2022-06-01", "2024-06-01"), "_cut2023": ("2021-06-01", "2023-06-01"),
         "_cut2022": ("2020-06-01", "2022-06-01"), "test": ("2023-06-01", "2025-06-01")}
BINS = np.arange(-4, 0.01, 1.0)


def table(a: pd.DataFrame, start: str, end: str) -> pd.Series:
    t = a[(a.ts >= pd.Timestamp(start, tz=TZ)) & (a.ts < pd.Timestamp(end, tz=TZ)) & a.energy.notna()
          & (a.elev > BINS[0]) & (a.elev <= 0)]
    cf = t.energy / t.capacity_kw
    return cf.groupby(pd.cut(t.elev, BINS), observed=True).median()


def main() -> None:
    ap = argparse.ArgumentParser(description="일출·일몰 시각 보정")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None, help="기본: 이름 끝 x2 앞에 t (E12-10cx2 → E12-10ctx2)")
    args = ap.parse_args()
    out = args.out or (args.preds[:-2] + "tx2" if args.preds.endswith("x2") else args.preds + "t")
    a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "capacity_kw", "elev"])
    a["site_id"] = a.site_id.astype(str)
    a["ts"] = pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    for suf, (s, e) in FOLDS.items():
        src = (PREDS_TEST / f"{args.preds}.parquet") if suf == "test" else (PREDS / f"{args.preds}{suf}.parquet")
        if not src.exists():
            continue
        med = table(a, s, e)
        p = pd.read_parquet(src)
        p["site_id"] = p.site_id.astype(str)
        p["ts"] = pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
        if suf == "test":                 # base.parquet 에는 평가 기간 행이 없음 → 태양 고도를 계산(구간 중앙, 같은 식)
            from oibc_common import load_sites, solar
            st = load_sites()[["site_id", "lat", "lon", "capacity_kw"]].assign(site_id=lambda d: d.site_id.astype(str))
            m = p.merge(st, on="site_id", how="left")
            m["elev"] = solar(m.ts.dt.tz_localize(None), m.lat.to_numpy(), m.lon.to_numpy())[0]
        else:
            m = p.merge(a[["site_id", "ts", "elev", "capacity_kw"]], on=["site_id", "ts"], how="left")
        hit = (m.elev > BINS[0]) & (m.elev <= 0) & (m.pred_kwh <= 0)
        cf = pd.cut(m.elev[hit], BINS).map(med).astype(float).fillna(0.0)
        m.loc[hit, "pred_kwh"] = cf.to_numpy() * m.capacity_kw[hit].to_numpy()
        dst = (PREDS_TEST / f"{out}.parquet") if suf == "test" else (PREDS / f"{out}{suf}.parquet")
        m[["site_id", "ts", "pred_kwh"]].to_parquet(dst, index=False)
        meta = {"exp": dst.stem, "twilight_fix_of": src.stem, "bins_median_cf": {str(k): round(float(v), 5) for k, v in med.items()},
                "rows_filled": int(hit.sum()), "train_window": [s, e], "created": datetime.now().isoformat(timespec="seconds")}
        dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"저장 {dst.relative_to(ROOT)}: 채운 행 {int(hit.sum()):,}, 추가 발전 {m.pred_kwh[hit].sum() / 1000:,.1f} MWh, 중앙값 {meta['bins_median_cf']}")


if __name__ == "__main__":
    main()
