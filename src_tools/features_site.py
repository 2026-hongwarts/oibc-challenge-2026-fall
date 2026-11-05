"""③ 태양·발전소·데이터 피처. 실험 하나 = 함수 하나.

    python src_tools/train_site_model.py --exp E3-02x2 --base champion --seeds 2 --add features_site:add_clear_ref
    python src_tools/train_site_model.py --exp E3-03x2 --base champion --seeds 2 --add features_site:add_clearsky
    python src_tools/train_site_model.py --exp E3-04x2 --base champion --seeds 2 --add features_site:mark_partial_outage --drop-dead

발전량(energy, cf)으로 만드는 값은 학습 행(is_fit)에서만 계산한다. 예측 행의 cf 는 NaN 으로 들어온다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_clear_ref(df: pd.DataFrame) -> pd.DataFrame:
    """E3-02 맑은날 기준값: 발전소 × 월 × 시간별 cf 상위 5%. 표본 20개 미만이면 전체 발전소 (월, 시간) 값."""
    fit = df[df.is_fit & (df.elev > 5) & ~df.is_dead.eq(True)]      # 학습 행만 (누설 방지)
    g = fit.groupby(["site_id", "month", "hour"], observed=True).cf
    ref = g.quantile(0.95).where(g.size() >= 20)
    ref_all = fit.groupby(["month", "hour"]).cf.quantile(0.95)
    a = ref.reindex(pd.MultiIndex.from_arrays([df.site_id, df.month, df.hour])).to_numpy()
    b = ref_all.reindex(pd.MultiIndex.from_arrays([df.month, df.hour])).to_numpy()
    df["site_cf_p95"] = np.nan_to_num(np.where(np.isnan(a), b, a)).astype("float32")   # 밤은 0
    return df


def _p95_ref(src: pd.DataFrame, dst: pd.DataFrame, min_n: int) -> np.ndarray:
    """src 학습 행으로 (발전소, 월, 시간) cf p95 를 만들어 dst 행에 붙인다. 표본 부족·신규 발전소는 (월, 시간) 값."""
    g = src.groupby(["site_id", "month", "hour"], observed=True).cf
    ref = g.quantile(0.95).where(g.size() >= min_n)
    ref_all = src.groupby(["month", "hour"]).cf.quantile(0.95)
    a = ref.reindex(pd.MultiIndex.from_arrays([dst.site_id, dst.month, dst.hour])).to_numpy()
    b = ref_all.reindex(pd.MultiIndex.from_arrays([dst.month, dst.hour])).to_numpy()
    return np.nan_to_num(np.where(np.isnan(a), b, a))


def add_clear_ref_oof(df: pd.DataFrame) -> pd.DataFrame:
    """E3-02b 맑은날 기준값, 교차 계산. 학습 행은 다른 해(12개월 블록)로 만든 값을 받아 자기 실적이 피처에 안 섞인다.
    예측 행은 학습 기간 전체로 만든 값. E3-02 는 학습 행에 자기 실적이 섞여(+5.05%) 과대 예측이 커졌다."""
    ok = df.is_fit & (df.elev > 5) & ~df.is_dead.eq(True)
    last = df.ts[df.is_fit].max()
    recent = df.ts > last - pd.DateOffset(months=12)
    out = np.zeros(len(df))
    for f in (recent, ~recent):                                  # 학습 행: 다른 블록으로
        rows = (df.is_fit & f).to_numpy()
        out[rows] = _p95_ref(df[ok & ~f], df[rows], min_n=10)
    rows = (~df.is_fit).to_numpy()                               # 예측 행: 학습 기간 전체
    out[rows] = _p95_ref(df[ok], df[rows], min_n=20)
    df["site_cf_p95"] = out.astype("float32")
    return df


def _p95_ref_st(src: pd.DataFrame, dst: pd.DataFrame, min_n: int) -> np.ndarray:
    """_p95_ref 와 같고, 발전소 값이 없으면 같은 관측지점(wb_station) 발전소들 값의 평균 → 없으면 (월, 시간) 값."""
    g = src.groupby(["site_id", "month", "hour"], observed=True).cf
    ref = g.quantile(0.95).where(g.size() >= min_n).dropna()
    st = src.drop_duplicates("site_id").set_index("site_id").wb_station
    ref_st = ref.groupby([ref.index.get_level_values(0).map(st), ref.index.get_level_values(1),
                          ref.index.get_level_values(2)]).mean()
    ref_all = src.groupby(["month", "hour"]).cf.quantile(0.95)
    a = ref.reindex(pd.MultiIndex.from_arrays([dst.site_id, dst.month, dst.hour])).to_numpy()
    s = ref_st.reindex(pd.MultiIndex.from_arrays([dst.wb_station, dst.month, dst.hour])).to_numpy()
    b = ref_all.reindex(pd.MultiIndex.from_arrays([dst.month, dst.hour])).to_numpy()
    return np.nan_to_num(np.where(np.isnan(a), np.where(np.isnan(s), b, s), a))


def add_clear_ref_oof_st(df: pd.DataFrame) -> pd.DataFrame:
    """E3-06 add_clear_ref_oof + 관측지점 대체. 이력이 없거나 짧은 신규 48곳(2024-06 ~ 12월 시작)은
    전체 평균 대신 같은 wb_station 기존 발전소의 기준값을 빌린다. 실제 평가 기간에도 신규 발전소는
    이력이 5 ~ 12개월이라 빠진 (월, 시간)이 생긴다."""
    ok = df.is_fit & (df.elev > 5) & ~df.is_dead.eq(True)
    last = df.ts[df.is_fit].max()
    recent = df.ts > last - pd.DateOffset(months=12)
    out = np.zeros(len(df))
    for f in (recent, ~recent):
        rows = (df.is_fit & f).to_numpy()
        out[rows] = _p95_ref_st(df[ok & ~f], df[rows], min_n=10)
    rows = (~df.is_fit).to_numpy()
    out[rows] = _p95_ref_st(df[ok], df[rows], min_n=20)
    df["site_cf_p95"] = out.astype("float32")
    return df


def add_clearsky(df: pd.DataFrame) -> pd.DataFrame:
    """E3-03 청천 일사: pvlib Ineichen GHI. ts 는 구간 끝 시각이라 가운데(ts − 30분)로 계산."""
    import pvlib
    out = np.zeros(len(df), dtype="float32")
    t = df.ts - pd.Timedelta(minutes=30)
    for _, idx in df.groupby("site_id", observed=True).indices.items():
        loc = pvlib.location.Location(float(df.lat.iat[idx[0]]), float(df.lon.iat[idx[0]]), tz="Asia/Seoul")
        out[idx] = loc.get_clearsky(pd.DatetimeIndex(t.iloc[idx]), model="ineichen").ghi.to_numpy()
    df["sol_ghi_clear"] = out
    return df


def add_site_features(df: pd.DataFrame) -> pd.DataFrame:
    """③ 전달용: 채택된 피처만 (= E3-02b 맑은날 기준값 교차 계산, E1-01x2 대비 -2.35%).
    기각: E3-02(자기 실적 섞임 +5.05%), E3-03 청천 일사(-0.20%), E3-04 정지·부분 정지 제외(+0.51%),
    E3-06 관측지점 대체(E3-02b 대비 +0.43%). 학습 제외 규칙은 없음(정지일 포함 학습 유지)."""
    return add_clear_ref_oof(df)


def mark_partial_outage(df: pd.DataFrame) -> pd.DataFrame:
    """E3-04 부분 정지: 같은 시각 다른 발전소 중앙 cf ≥ 0.3 인데 혼자 ≤ 0.02 인 학습 행을 is_dead 로. --drop-dead 와 함께."""
    fit = df.is_fit & (df.elev > 20)
    med = df.cf.where(fit).groupby(df.ts).transform("median")
    df.loc[fit & (df.cf <= 0.02) & (med >= 0.3), "is_dead"] = True
    return df
