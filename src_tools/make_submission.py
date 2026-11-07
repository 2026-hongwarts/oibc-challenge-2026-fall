"""제출 파일 만들기 + 형식 검사.

    python src_tools/make_submission.py outputs/preds_test/E2-finalx2.parquet
    python src_tools/make_submission.py outputs/preds_test/E2-finalx2.parquet --composition comp.csv --name 1108_E2_comp

결과: outputs/submissions/<이름>/composition.csv, forecast.csv
    이름 기본값 = 예측 파일 이름. 구성 기본값 = sites.csv 의 current_group (A→G1, B→G2).
검사 (파일을 쓴 뒤 다시 읽어서 한다)
    composition.csv: 헤더 포함 254행, sites.csv 253곳 전부, group 은 G1·G2 만, 묶음별 설비용량 20,000kW 이상
    forecast.csv   : 헤더 포함 17,521행, G1·G2 × smp.csv 의 8,760시각(ts 문자열까지 같음), 타임존 포함, 0 이상, NaN 없음
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from oibc_common import CAP_MIN_KW, DATA, OUTPUTS, TZ, load_sites

GROUPS = ["G1", "G2"]


class SubmissionError(ValueError):
    pass


def load_composition(path: str | None, sites: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        return sites.assign(group=sites.current_group.map({"A": "G1", "B": "G2"}))[["site_id", "group"]]
    return pd.read_csv(path)[["site_id", "group"]]


def build(preds_path: str | Path, comp_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = load_sites()
    comp = load_composition(comp_path, sites)
    smp = pd.read_csv(DATA / "smp.csv")                      # ts 문자열 원본을 그대로 쓴다
    smp["t"] = pd.to_datetime(smp.ts).dt.tz_convert(TZ).dt.as_unit("ns")

    p = pd.read_parquet(preds_path)
    if not {"site_id", "ts", "pred_kwh"} <= set(p.columns):
        raise SubmissionError(f"예측 파일 열은 site_id, ts, pred_kwh 여야 함. 지금 {list(p.columns)}")
    p["site_id"] = p.site_id.astype(str)
    p["t"] = pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    full = pd.MultiIndex.from_product([sites.site_id, smp.t], names=["site_id", "t"])
    have = p.set_index(["site_id", "t"]).pred_kwh
    if have.index.duplicated().any():
        raise SubmissionError(f"(site_id, ts) 중복 {int(have.index.duplicated().sum())}개")
    have = have.reindex(full)
    if have.isna().any():
        raise SubmissionError(f"평가 기간 (발전소, 시각) 중 예측이 없거나 NaN 인 행 {int(have.isna().sum()):,}개")
    if (have < 0).any():
        raise SubmissionError(f"음수 예측 {int((have < 0).sum())}개")

    g = have.reset_index().merge(comp, on="site_id", how="left")
    if g.group.isna().any():
        raise SubmissionError("구성에 없는 발전소가 예측에 있음")
    s = g.groupby(["group", "t"]).pred_kwh.sum()
    s = s.reindex(pd.MultiIndex.from_product([GROUPS, smp.t], names=["group", "t"])).fillna(0.0)
    ts_str = dict(zip(smp.t, smp.ts))
    fc = s.reset_index()
    fc["ts"] = fc.t.map(ts_str)
    fc = fc.rename(columns={"pred_kwh": "forecast_kwh"})[["group", "ts", "forecast_kwh"]]
    fc["forecast_kwh"] = fc.forecast_kwh.round(4)
    return comp.sort_values("site_id").reset_index(drop=True), fc


def check(out_dir: Path) -> list[str]:
    """쓴 파일을 다시 읽어 대회 형식을 검사한다. 문제 목록을 돌려준다(비어 있으면 통과)."""
    errs = []
    sites = load_sites()
    smp = pd.read_csv(DATA / "smp.csv")
    c = pd.read_csv(out_dir / "composition.csv")
    f = pd.read_csv(out_dir / "forecast.csv")
    n_c = sum(1 for _ in open(out_dir / "composition.csv", encoding="utf-8"))
    n_f = sum(1 for _ in open(out_dir / "forecast.csv", encoding="utf-8"))
    if list(c.columns) != ["site_id", "group"]: errs.append(f"composition 헤더 {list(c.columns)}")
    if n_c != 254: errs.append(f"composition 행 수 {n_c} (헤더 포함 254 여야 함)")
    if sorted(c.site_id) != sorted(sites.site_id): errs.append("composition 의 site_id 가 sites.csv 253곳과 다름")
    if not set(c.group) <= set(GROUPS): errs.append(f"group 값 {sorted(set(c.group))}")
    cap = sites.set_index("site_id").capacity_kw.groupby(c.set_index("site_id").group).sum()
    if (cap.reindex(GROUPS).fillna(0) < CAP_MIN_KW).any(): errs.append(f"묶음 설비용량 {cap.round(1).to_dict()} (각 20,000kW 이상)")
    if list(f.columns) != ["group", "ts", "forecast_kwh"]: errs.append(f"forecast 헤더 {list(f.columns)}")
    if n_f != 17521: errs.append(f"forecast 행 수 {n_f} (헤더 포함 17,521 이어야 함)")
    for gname in GROUPS:
        ts_g = f.loc[f.group == gname, "ts"].tolist()
        if ts_g != smp.ts.tolist(): errs.append(f"{gname} 의 ts 가 smp.csv 8,760시각과 다름")
    if not f.ts.str.contains(r"\+09:00$").all(): errs.append("ts 에 타임존(+09:00)이 없는 행이 있음")
    if f.forecast_kwh.isna().any() or (f.forecast_kwh < 0).any(): errs.append("forecast_kwh 에 NaN 또는 음수")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description="제출 파일 만들기 + 형식 검사")
    ap.add_argument("preds", help="실제 평가 기간 예측 (outputs/preds_test/실험.parquet)")
    ap.add_argument("--composition", default=None, help="site_id,group CSV (기본: 참조 구성)")
    ap.add_argument("--name", default=None, help="출력 폴더 이름 (기본: 예측 파일 이름)")
    args = ap.parse_args()

    name = args.name or Path(args.preds).stem
    out_dir = OUTPUTS / "submissions" / name
    try:
        comp, fc = build(args.preds, args.composition)
    except SubmissionError as e:
        sys.exit(f"만들기 실패: {e}")
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_dir / "composition.csv", index=False)
    fc.to_csv(out_dir / "forecast.csv", index=False)

    errs = check(out_dir)
    sites = load_sites()
    cap = sites.set_index("site_id").capacity_kw.groupby(comp.set_index("site_id").group).sum()
    print(f"[{name}] → {out_dir}")
    print(f"  구성: G1 {int((comp.group == 'G1').sum())}곳 {cap['G1']:,.1f}kW / G2 {int((comp.group == 'G2').sum())}곳 {cap['G2']:,.1f}kW")
    tot = fc.groupby("group").forecast_kwh.sum() / 1e6
    print(f"  연간 예측: G1 {tot['G1']:.2f} GWh, G2 {tot['G2']:.2f} GWh, 합계 {tot.sum():.2f} GWh")
    ref = OUTPUTS / "submissions" / "baseline_original" / "forecast.csv"
    if ref.exists() and name != "baseline_original":
        b = pd.read_csv(ref).merge(fc, on=["group", "ts"], suffixes=("_b", ""))
        print(f"  baseline.py 원본 대비: 합계 {(b.forecast_kwh.sum() / b.forecast_kwh_b.sum() - 1) * 100:+.2f}%, "
              f"시각별 상관 {np.corrcoef(b.forecast_kwh, b.forecast_kwh_b)[0, 1]:.4f}  (참조 구성이 같을 때만 의미 있음)")
    if errs:
        print("  형식 검사 실패:")
        for e in errs:
            print(f"    - {e}")
        sys.exit(1)
    print("  형식 검사 통과: composition 254행, forecast 17,521행, ts·타임존·0 이상·용량 조건 모두 만족")


if __name__ == "__main__":
    main()
