"""F8 — 범주형으로 처리할 후보 컬럼의 고유값 비율과 범주별 표본 수.

    python analysis_lab/F8_범주형_후보.py

규칙: 고유값 비율(고유값 수 / 행 수)이 1 에 가까우면 행마다 값이 다른 ID 라 일반화할 정보가 없으므로 버린다.
함께 보는 것: 범주당 행 수(적으면 과적합), 예측 행 중 학습에 없던 범주의 비율(처음 보는 범주).
학습 행 = 태양고도 > 0. 가상 평가연도(학습 2022-06 ~ 2024-05)와 실제 평가(학습 2023-06 ~ 2025-05) 두 창을 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_tools"))
from train_site_model import load_frame  # noqa: E402

CANDS = ["site_id", "wa_station", "wb_station", "hour",
         # 참고: 정수 단계로 떨어지는 예보 값 (순서가 있는 숫자)
         "wb_uv_idx_r17", "wb_precip_prob_r17", "wa_cloud_r17", "wb_cloud_r17", "ts", "date"]


def main() -> None:
    for mode, label in [("valid", "가상 평가연도 창"), ("test", "실제 평가 창")]:
        df = load_frame(mode, 2)
        df = df[df.elev > 0]
        fit, pr = df[df.is_fit], df[~df.is_fit]
        print(f"\n[{label}] 학습 행 {len(fit):,} / 예측 행 {len(pr):,}")
        print(f"{'컬럼':20s} {'고유값':>8s} {'고유값 비율':>12s} {'범주당 행 최소':>12s} {'중앙':>10s} {'처음 보는 범주(예측 행 %)':>24s}")
        for c in CANDS:
            s = fit[c]
            n = s.nunique()
            cnt = s.value_counts()
            unseen = (~pr[c].isin(set(s.unique()))).mean() * 100
            print(f"{c:20s} {n:8,d} {n / len(s):12.6f} {cnt.min():12,d} {int(cnt.median()):10,d} {unseen:24.2f}")


if __name__ == "__main__":
    main()
