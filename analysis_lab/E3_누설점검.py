"""피처 누설 점검 — ③이 ② 코드(11/7 ~ 11/8 점검)를 확인하려고 만든 스크립트. 저장소 맨 위에서 실행.

    python analysis_lab/E3_누설점검.py

방법: 학습 스크립트와 같은 df(가상 평가연도, 학습 2년)를 만들고 피처 함수를 두 번 적용한다.
  1) 전체 데이터로  2) 대상일 D 이후 행을 모두 지운 데이터로
D 의 예측 행 피처 값이 두 경우에 같으면, 그 피처는 D 이후 정보를 쓰지 않는다.
학습 행은 보지 않는다(③ add_site_features 는 학습 행끼리 다른 해 값을 쓰도록 설계됨, 모두 컷오프 전).
"""
import sys

sys.path.insert(0, "src_tools")
import numpy as np
import pandas as pd

import features_weather as fw
from train_site_model import load_frame

HOOKS = [("features_weather:add_weather_features", fw.add_weather_features)]
try:                                    # ③ 전달본 (PR #4 병합 후 main 에 있음)
    import features_site as fs
    HOOKS.append(("features_site:add_site_features", fs.add_site_features))
except (ImportError, AttributeError):
    pass

DATES = ["2024-06-01", "2024-09-15", "2025-01-20", "2025-05-31"]


def apply(df: pd.DataFrame) -> pd.DataFrame:
    for _, f in HOOKS:
        df = f(df)
    return df


def main() -> None:
    df = load_frame("valid", 2)
    df.loc[~df.is_fit, ["energy", "cf"]] = np.nan      # 학습 스크립트와 같게
    df["is_dead"] = df.is_dead.eq(True) & df.is_fit
    before = set(df.columns)
    full = apply(df.copy())
    new = [c for c in full.columns if c not in before]
    print("점검 함수:", [n for n, _ in HOOKS])
    print("피처", len(new), "개:", new)
    ok = True
    for d in DATES:
        D = pd.Timestamp(d)
        keep = (df.date <= D).to_numpy()
        cut = apply(df[keep].reset_index(drop=True).copy())
        a = full.loc[keep].reset_index(drop=True)
        rows = (cut.date == D).to_numpy() & ~cut.is_fit.to_numpy()
        bad = []
        for c in new:
            x, y = a.loc[rows, c].to_numpy(float), cut.loc[rows, c].to_numpy(float)
            if (np.isnan(x) != np.isnan(y)).any() or np.nanmax(np.abs(x - y), initial=0) > 1e-6:
                bad.append(c)
        ok &= not bad
        print(f"{d}  예측 행 {int(rows.sum()):,}개  값이 바뀐 피처: {bad or '없음'}")
    print("결과:", "누설 없음" if ok else "누설 의심 — 위 피처 확인")


if __name__ == "__main__":
    main()
