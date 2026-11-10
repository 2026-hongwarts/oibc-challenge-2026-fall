"""F10 — 검증 연도의 '처음 보는 발전소'를 떼어 내고 실험 효과를 다시 본다 (2026-11-11, docs/피처_근거_분석.md §6.3).

검증 연도(2024-25 / 2023-24 / 2022-23)에는 직전 2년 학습 기간에 없던 발전소가 17 ~ 42% 행을 차지하지만,
실제 평가 기간에는 0% 다(모든 발전소 이력 158일 이상). 발전소 정체성을 쓰는 피처(범주형 ID 등)는 처음 보는
발전소에서 크게 틀릴 수 있어, 검증 결과가 실제 평가 조건보다 나쁘게 나올 수 있다.

혼합 예측 = 학습 기간에 있던 발전소는 실험 예측, 처음 보는 발전소는 기준 예측. 참조 구성으로 채점(compare_years 와 같음).

    python analysis_lab/F10_본발전소_분리.py 기준 실험1 실험2 ...
    python analysis_lab/F10_본발전소_분리.py E9-00x2 V02x2 V03x2 E9-00s2x2     (E9-00s2x2 = 시드만 바꾼 대조군)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
import pandas as pd

from compare_years import actual
from evaluate import evaluate_preds, read_preds
from oibc_common import PREDS, WORK
from train_site_model import cut_window

YEARS = [("2024-25", None, "2024-06-01"), ("2023-24", "2023-06-01", "2023-06-01"), ("2022-23", "2022-06-01", "2022-06-01")]


def path(name: str, cut: str | None) -> Path:
    return PREDS / (f"{name}_cut{cut[:4]}.parquet" if cut else f"{name}.parquet")


def main() -> None:
    base_name, names = sys.argv[1], sys.argv[2:]
    base = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts"])
    base["site_id"] = base.site_id.astype(str)
    for y, cut, start in YEARS:
        s, _ = cut_window(start)
        seen = set(base.loc[(base.ts < s) & (base.ts >= s - pd.DateOffset(years=2)), "site_id"].unique())
        a = actual(cut)
        ref = read_preds(path(base_name, cut))
        unseen = ~a.site_id.isin(seen)
        row = [y, f"처음 보는 발전소 {a.loc[unseen].site_id.nunique()}곳, 행 {unseen.mean() * 100:.1f}%"]
        p0 = evaluate_preds(ref, actual=a)["penalty"]
        for n in names:
            v = read_preds(path(n, cut))
            full = evaluate_preds(v, actual=a)["penalty"] / p0 - 1
            mix = pd.concat([v[v.site_id.isin(seen)], ref[~ref.site_id.isin(seen)]], ignore_index=True)
            part = evaluate_preds(mix, actual=a)["penalty"] / p0 - 1
            row.append(f"{n}: 전체 {full * 100:+.2f}% / 본 발전소만 바꿈 {part * 100:+.2f}%")
        print(" | ".join(row), flush=True)


if __name__ == "__main__":
    main()
