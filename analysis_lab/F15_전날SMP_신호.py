"""F15 — 전날 SMP 가 다음 날 과대 예측(출력제어)의 신호인가 (2026-11-11, docs/피처_근거_분석.md §12).

규정 §8: 대상일 SMP 는 쓸 수 없지만 "전날 17:00 이전에 알 수 있는 SMP" 는 쓸 수 있다. 평가 기간 smp.csv 의 값도
날짜 순서대로(그 대상일 전날 17:00 까지의 시각만) 쓰면 허용된다(§6 "날짜 순서대로 진행하면서 그날까지 공개된 값만").
지표(대상일 D 마다, D-1 의 01 ~ 17시 SMP 만 사용):
    dip = D-1 한낮(11 ~ 15시) 평균 SMP ÷ D-1 01 ~ 17시 평균 SMP   (연료비 수준과 무관한 '한낮 공급 과잉' 정도)
    low = D-1 01 ~ 17시 최저 SMP ÷ 같은 평균
결과: 검증 세 해마다 dip 구간별로 D 한낮 묶음(전체 발전소 합) 예측÷실적, 그리고 맑은 날(A 운량 ≤ 30%)만.
기준 예측은 추세 보정까지 붙인 E12-10crx2(지금 최고 계열).
    python analysis_lab/F15_전날SMP_신호.py [예측 이름]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import numpy as np
import pandas as pd

from compare_years import actual
from oibc_common import PREDS, WORK, load_smp

ENS = sys.argv[1] if len(sys.argv) > 1 else "E12-10crx2"
FOLDS = [("2024-25", None, ""), ("2023-24", "2023-06-01", "_cut2023"), ("2022-23", "2022-06-01", "_cut2022")]


def smp_signal() -> pd.DataFrame:
    s = load_smp("train")
    s["day"] = (s.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
    s = s[(s.ts.dt.hour >= 1) & (s.ts.dt.hour <= 17)]                  # D-1 의 17시까지만 (마감 전에 끝난 시각)
    g = s.groupby("day")
    mid = s[(s.ts.dt.hour >= 11) & (s.ts.dt.hour <= 15)].groupby("day").smp.mean()
    out = pd.DataFrame({"dip": mid / g.smp.mean(), "low": g.smp.min() / g.smp.mean()})
    out.index = out.index + pd.Timedelta(days=1)                        # D-1 의 값 → 대상일 D 에 붙임
    return out


def main():
    sig = smp_signal()
    base = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "wa_cloud_r17"])
    base["site_id"] = base.site_id.astype(str)
    print(f"기준 {ENS}. 대상일 한낮(11 ~ 15시) 전체 발전소 합 예측÷실적, 전날 SMP 한낮 비율(dip) 구간별\n")
    for y, cut, tag in FOLDS:
        a = actual(cut)
        p = pd.read_parquet(PREDS / f"{ENS}{tag}.parquet")
        p["site_id"] = p.site_id.astype(str)
        p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
        m = a.merge(p, on=["site_id", "ts"]).merge(base, on=["site_id", "ts"], how="left")
        m = m[(m.ts.dt.hour >= 11) & (m.ts.dt.hour <= 15)]
        m["day"] = (m.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
        d = m.groupby("day").agg(f=("pred_kwh", "sum"), a=("energy", "sum"), cl=("wa_cloud_r17", "mean")).join(sig)
        d = d.dropna()
        d["r"] = d.f / d.a
        d["bin"] = pd.qcut(d.dip, 5, labels=["하위 20%(한낮 가장 쌈)", "20~40%", "40~60%", "60~80%", "상위 20%"])
        allv = d.groupby("bin", observed=True).apply(lambda g: pd.Series({"날수": len(g), "예측÷실적": g.f.sum() / g.a.sum(),
                                                                         "dip 평균": g.dip.mean()}))
        clr = d[d.cl <= 30].groupby("bin", observed=True).apply(lambda g: g.f.sum() / g.a.sum())
        allv["맑은 날 예측÷실적"] = clr
        rho = d[["dip", "r"]].corr(method="spearman").iloc[0, 1]
        rho_c = d[d.cl <= 30][["dip", "r"]].corr(method="spearman").iloc[0, 1]
        print(f"{y}: 날짜별 순위상관(dip, 예측÷실적) 전체 {rho:+.3f} / 맑은 날 {rho_c:+.3f}")
        print(allv.round(3).to_string())
        print()


if __name__ == "__main__":
    main()
