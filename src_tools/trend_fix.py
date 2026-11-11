"""맑은 날 수준 추세 보정 (11/11 사용자 승인) — 재학습 없이 예측 파일에 배율 하나를 곱한다.

    python src_tools/trend_fix.py --preds E12-10cx2          → E12-10crx2 (세 해 + 평가 기간 파일이 있으면 그것도)

근거(파이프라인 점검): 같은 발전소의 맑은 시각 발전 수준이 해마다 낮아짐(23-24/22-23 = 0.970, 24-25/23-24 = 0.986),
    봄·가을에 몰려 출력제어로 의심. 모델은 학습 기간 평균 수준을 배우므로 다음 해를 과대예측(1.013 / 1.018 / 1.001).
방법(미리 정함, 학습 기간 실측만 — 규정 §6):
    학습 기간 T = 검증 시작 전 2년을 앞 1년 A, 뒤 1년 B 로 나눈다.
    발전소 × 달마다 '해가 높은 시각(고도 > 30°)의 발전율 / cos 천정각' 의 90% 분위수(맑은 날 능력)를 구하고,
    A 와 B 에 같은 달이 모두 있고 각 30행 이상인 (발전소, 달) 쌍의 B/A 비율 중앙값 = r.
    다음 해 예측 × r (모든 시각). 자료가 모자라 쌍이 50개 미만이면 r = 1.
판정(미리 정함): 세 해 모두 원래 예측보다 벌금이 낮아야 채택.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from oibc_common import PREDS, PREDS_TEST, ROOT, TZ, WORK

FOLDS = {"": "2024-06-01", "_cut2023": "2023-06-01", "_cut2022": "2022-06-01", "test": "2025-06-01"}


def trend_ratio(a: pd.DataFrame, v_start: str) -> tuple[float, int]:
    v = pd.Timestamp(v_start, tz=TZ)
    a0, a1, b1 = v - pd.DateOffset(years=2), v - pd.DateOffset(years=1), v
    t = a[(a.ts >= a0) & (a.ts < b1) & a.energy.notna() & (a.elev > 30)].copy()
    t["half"] = np.where(t.ts < a1, "A", "B")
    t["mon"] = t.ts.dt.month
    t["cap"] = t.energy / t.capacity_kw / t.cosz
    q = t.groupby(["site_id", "mon", "half"]).cap.agg(lambda s: s.quantile(0.9) if len(s) >= 30 else np.nan).unstack("half")
    q = q.dropna()
    q = q[(q.A > 0.05) & (q.B > 0.05)]
    if len(q) < 50:
        return 1.0, len(q)
    return float((q.B / q.A).median()), len(q)


def main() -> None:
    ap = argparse.ArgumentParser(description="맑은 날 수준 추세 보정")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=None, help="기본: 이름 끝 x2 앞에 r (E12-10cx2 → E12-10crx2)")
    args = ap.parse_args()
    out = args.out or (args.preds[:-2] + "rx2" if args.preds.endswith("x2") else args.preds + "r")
    a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "capacity_kw", "elev", "cosz"])
    a["ts"] = pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    for suf, v in FOLDS.items():
        src = (PREDS_TEST / f"{args.preds}.parquet") if suf == "test" else (PREDS / f"{args.preds}{suf}.parquet")
        if not src.exists():
            continue
        r, n = trend_ratio(a, v)
        p = pd.read_parquet(src)
        p["pred_kwh"] = p.pred_kwh * r
        dst = (PREDS_TEST / f"{out}.parquet") if suf == "test" else (PREDS / f"{out}{suf}.parquet")
        p.to_parquet(dst, index=False)
        meta = {"exp": dst.stem, "trend_fix_of": src.stem, "ratio": round(r, 5), "pairs": n, "valid_start": v,
                "created": datetime.now().isoformat(timespec="seconds")}
        dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{suf or '2024-25'}: r = {r:.4f} (발전소×달 쌍 {n}개) → 저장 {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
