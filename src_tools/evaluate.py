"""가상 평가연도 채점기 — 대회 점수식 그대로, 모두가 이것 하나로 채점한다.

    python src_tools/evaluate.py outputs/preds/E2-01.parquet                        기준 모델(outputs/preds/champion.txt) 대비로 출력
    python src_tools/evaluate.py outputs/preds/E2-01.parquet --ref outputs/preds/E4-00.parquet
    python src_tools/evaluate.py outputs/preds/E5-01.parquet --composition my_composition.csv

예측 파일 (parquet)
    발전소별: site_id, ts, pred_kwh   — outputs/work/base.parquet 의 valid 행(2024-06-01 01:00 ~ 2025-06-01 00:00)을 모두 덮어야 함
    묶음별  : group, ts, pred_kwh     — G1·G2 × 가상 평가연도 8,760시각
    ts 는 타임존이 있어야 하고, pred_kwh 는 0 이상, NaN 없음.

구성: 기본은 sites.csv 의 current_group(A→G1, B→G2). --composition 으로 site_id,group CSV 를 줄 수 있다.
가중치: 학습 구간 SMP × k(11~15시 3). 벌금 = 0.10 × Σ SMP·k·|예측 − 실적| (G1+G2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from oibc_common import CAP_MIN_KW, PENALTY_RATE, PREDS, TZ, WORK, day_of, load_sites, load_smp


class EvalError(ValueError):
    pass


def load_actual() -> pd.DataFrame:
    a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "split"])
    a = a[a.split == "valid"].drop(columns="split").reset_index(drop=True)
    a["site_id"] = a.site_id.astype(str)
    return a


def load_composition(path: str | None = None) -> pd.Series:
    sites = load_sites()
    if path is None:
        comp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    else:
        c = pd.read_csv(path)
        comp = c.set_index("site_id").group
        missing = set(sites.site_id) - set(comp.index)
        if missing or not set(comp.unique()) <= {"G1", "G2"}:
            raise EvalError(f"구성 파일 오류: 빠진 발전소 {len(missing)}개, 그룹 값 {sorted(comp.unique())}")
    cap = sites.set_index("site_id").capacity_kw.groupby(comp).sum()
    if (cap < CAP_MIN_KW).any():
        print(f"  [경고] 설비용량 20,000kW 미만 묶음: {cap.round(1).to_dict()}", file=sys.stderr)
    return comp.rename("group")


def read_preds(path: str | Path) -> pd.DataFrame:
    p = pd.read_parquet(path)
    key = "site_id" if "site_id" in p else "group" if "group" in p else None
    if key is None or not {"ts", "pred_kwh"} <= set(p.columns):
        raise EvalError(f"{path}: 열은 site_id(또는 group), ts, pred_kwh 여야 함. 지금 {list(p.columns)}")
    p = p[[key, "ts", "pred_kwh"]].copy()
    p["ts"] = pd.to_datetime(p.ts)
    if p.ts.dt.tz is None:
        raise EvalError(f"{path}: ts 에 타임존이 없음")
    p["ts"] = p.ts.dt.tz_convert(TZ).dt.as_unit("ns")
    p[key] = p[key].astype(str)
    if p.pred_kwh.isna().any() or (p.pred_kwh < 0).any():
        raise EvalError(f"{path}: pred_kwh 에 NaN {int(p.pred_kwh.isna().sum())}개, 음수 {int((p.pred_kwh < 0).sum())}개")
    dup = p.duplicated([key, "ts"]).sum()
    if dup:
        raise EvalError(f"{path}: ({key}, ts) 중복 {dup}개")
    return p


def group_frame(actual: pd.DataFrame, preds: pd.DataFrame, comp: pd.Series) -> pd.DataFrame:
    """묶음 × 시각: actual, fcst."""
    a = actual.assign(group=actual.site_id.map(comp))
    ga = a.groupby(["group", "ts"], as_index=False).energy.sum().rename(columns={"energy": "actual"})
    if "site_id" in preds:
        m = a.merge(preds, on=["site_id", "ts"], how="left")
        miss = m.pred_kwh.isna()
        if miss.any():
            ex = m.loc[miss, ["site_id", "ts"]].head(3).to_dict("records")
            raise EvalError(f"가상 평가연도 행 중 예측이 없는 행 {int(miss.sum()):,}개 (예: {ex})")
        gf = m.groupby(["group", "ts"], as_index=False).pred_kwh.sum()
    else:
        gf = preds
    g = ga.merge(gf, on=["group", "ts"], how="left").rename(columns={"pred_kwh": "fcst"})
    if g.fcst.isna().any():
        raise EvalError(f"묶음 예측이 없는 (group, ts) {int(g.fcst.isna().sum())}개")
    return g


def penalty_frame(g: pd.DataFrame, smp: pd.DataFrame) -> pd.DataFrame:
    m = g.merge(smp[["ts", "smp", "k", "w"]], on="ts", how="left")
    if m.w.isna().any():
        raise EvalError("SMP 가 없는 시각이 있음")
    m["pen"] = PENALTY_RATE * m.w * (m.fcst - m.actual).abs()
    m["rev"] = m.smp * m.actual
    m["month"] = day_of(m.ts).dt.strftime("%Y-%m").to_numpy()
    return m


def summarize(m: pd.DataFrame) -> dict:
    return {"penalty": float(m.pen.sum()), "revenue": float(m.rev.sum()),
            "peak": float(m.pen[m.k == 3].sum()), "by_month": m.groupby("month").pen.sum()}


def evaluate_preds(preds: pd.DataFrame, composition: str | None = None,
                   actual: pd.DataFrame | None = None) -> dict:
    actual = load_actual() if actual is None else actual
    m = penalty_frame(group_frame(actual, preds, load_composition(composition)), load_smp("train"))
    out = summarize(m)
    out["n_sites"] = actual.site_id.nunique()
    return out


def champion_path() -> Path | None:
    f = PREDS / "champion.txt"
    if not f.exists():
        return None
    name = f.read_text(encoding="utf-8").strip()
    if not name:
        return None
    path = PREDS / f"{name}.parquet"
    if not path.exists():
        print(f"  [안내] 기준 모델 예측 {path.name} 이 이 PC에 없어 비교 없이 채점합니다. "
              f"만들려면: python src_tools/train_site_model.py --base {name}", file=sys.stderr)
        return None
    return path


def report(path: str | Path, ref: str | Path | None = None, composition: str | None = None,
           use_champion: bool = True) -> dict:
    path = Path(path)
    if ref is None and use_champion:
        ref = champion_path()
        if ref is not None and ref.resolve() == path.resolve():
            ref = None
    actual = load_actual()
    r = evaluate_preds(read_preds(path), composition, actual)
    b = evaluate_preds(read_preds(ref), composition, actual) if ref else None
    d = lambda x, y: f"{(x / y - 1) * 100:+.2f}%" if y else ""

    name = path.stem
    comp_name = "참조 구성" if composition is None else Path(composition).name
    print(f"\n[{name}] 가상 평가연도 2024-06 ~ 2025-05 · {comp_name} · 발전소 {r['n_sites']}곳"
          + (f" · 비교 기준 {Path(ref).stem}" if b else ""))
    print(f"  벌금          {r['penalty']:>15,.0f}원   (수익의 {r['penalty'] / r['revenue'] * 100:.2f}%)"
          + (f"   기준 대비 {d(r['penalty'], b['penalty'])}" if b else ""))
    print(f"  11~15시 벌금  {r['peak']:>15,.0f}원   (벌금의 {r['peak'] / r['penalty'] * 100:.1f}%)"
          + (f"   기준 대비 {d(r['peak'], b['peak'])}" if b else ""))
    print("  월별 벌금" + ("            기준 대비" if b else ""))
    for mth, v in r["by_month"].items():
        extra = f"   {d(v, b['by_month'][mth])}" if b else ""
        print(f"    {mth}   {v:>13,.0f}{extra}")
    line = [name, f"{r['penalty']:.0f}"]
    if b:
        line += [d(r["penalty"], b["penalty"]), d(r["peak"], b["peak"])]
    print("  시트에 붙여넣기:  " + "\t".join(line))
    return {"result": r, "ref": b}


def main() -> None:
    ap = argparse.ArgumentParser(description="가상 평가연도 채점기")
    ap.add_argument("preds", help="예측 파일 (outputs/preds/실험번호.parquet)")
    ap.add_argument("--ref", default=None, help="비교 기준 예측 파일 (기본: outputs/preds/champion.txt 의 모델)")
    ap.add_argument("--no-ref", action="store_true", help="기준 대비 비교를 하지 않음")
    ap.add_argument("--composition", default=None, help="site_id,group CSV (기본: 참조 구성)")
    args = ap.parse_args()
    try:
        report(args.preds, args.ref, args.composition, use_champion=not args.no_ref)
    except EvalError as e:
        sys.exit(f"채점 실패: {e}")


if __name__ == "__main__":
    main()
