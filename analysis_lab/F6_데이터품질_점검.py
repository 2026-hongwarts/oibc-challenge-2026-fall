"""F6 — 최종 피처(E9 계열 54개)의 결측·이상치 점검과 타깃 이상치 집계.

    python analysis_lab/F6_데이터품질_점검.py

학습 행(태양고도 > 0, 2022-06 ~ 2024-05), 가상 평가연도 행(2024-06 ~ 2025-05), 실제 평가 기간 행(2025-06 ~ 2026-05)으로 나눠
피처마다 결측률, 최솟값·최댓값, 물리 범위를 벗어난 비율을 낸다. 집계만 출력한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_tools"))
from features_ours import add_daily_v1  # noqa: E402
from features_v3 import add_hour_neighbors  # noqa: E402
from train_site_model import COMMON, WA_COLS, WB_COLS, load_frame  # noqa: E402

# 물리적으로 가능한 범위 (넉넉하게). 벗어나면 이상치로 센다
RANGE = {"cloud": (0, 100), "humidity": (0, 100), "rel_hum": (0, 100), "precip_prob": (0, 100),
         "rain": (0, 200), "total_liq": (0, 200), "vis": (0, 100), "uv_idx": (0, 15),
         "wind_speed": (0, 150), "ground_press": (850, 1070), "temp": (-40, 50), "dew_point": (-50, 40),
         "trange": (0, 40), "am_humid": (0, 100), "vis_min": (0, 100)}


def rng(col: str):
    for k, v in RANGE.items():
        if k in col:
            return v
    return None


def frame(mode: str) -> pd.DataFrame:
    df = load_frame(mode, 2)
    df.loc[~df.is_fit, ["energy", "cf"]] = np.nan
    df["is_dead"] = df.is_dead.eq(True) & df.is_fit
    return add_hour_neighbors(add_daily_v1(df))


def main() -> None:
    v = frame("valid")
    feats = [f"{c}_r{r}" for r in (10, 17) for c in WA_COLS + WB_COLS] + [c for c in COMMON if c not in ("doy_sin", "doy_cos")]
    feats += [c for c in v.columns if c.startswith("wx_")]
    parts = {"학습": v[v.is_fit & (v.elev > 0)], "가상평가": v[~v.is_fit & (v.elev > 0)]}
    t = frame("test")
    parts["실제평가"] = t[~t.is_fit & (t.elev > 0)]
    del t
    print(f"피처 {len(feats)}개, 행 수: " + ", ".join(f"{k} {len(p):,}" for k, p in parts.items()))

    rows = []
    for c in feats:
        r = {"피처": c}
        lo_hi = rng(c)
        for k, p in parts.items():
            s = p[c]
            r[f"{k} 결측%"] = s.isna().mean() * 100
            if lo_hi:
                r[f"{k} 범위밖%"] = ((s < lo_hi[0]) | (s > lo_hi[1])).mean() * 100
        r["최소"], r["최대"] = parts["학습"][c].min(), parts["학습"][c].max()
        r["평가 최소"], r["평가 최대"] = parts["실제평가"][c].min(), parts["실제평가"][c].max()
        rows.append(r)
    q = pd.DataFrame(rows).set_index("피처")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    flag = q[(q.filter(like="결측%").max(axis=1) > 0) | (q.filter(like="범위밖%").max(axis=1).fillna(0) > 0)]
    print("\n[1] 결측이나 범위 밖 값이 있는 피처 (나머지는 둘 다 0)")
    print(flag.round(4).to_string() if len(flag) else "  없음")
    print("\n[2] 전체 피처 최소·최대 (학습 / 실제 평가)")
    print(q[["최소", "최대", "평가 최소", "평가 최대"]].round(2).to_string())

    tr = parts["학습"]
    cf = tr.energy / tr.capacity_kw
    print("\n[3] 타깃 (학습 행, 태양고도 > 0)")
    print(f"  발전율 결측 {cf.isna().mean():.4%}, 음수 {(cf < 0).mean():.4%}, 1.0 초과 {(cf > 1.0).mean():.4%}, "
          f"1.2 초과 {(cf > 1.2).mean():.4%} (최대 {cf.max():.2f})")
    print(f"  정지 추정일 행 {tr.is_dead.mean():.3%}, 발전율 0 행 {(cf == 0).mean():.3%}")
    print(f"  발전소 수 {tr.site_id.nunique()}, 발전소-날 수 {tr.groupby(['site_id', 'date']).ngroups:,}")


if __name__ == "__main__":
    main()
