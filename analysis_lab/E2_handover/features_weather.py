"""② 예보 피처 — 전달본 (실험 E2-40). 과정·근거: analysis_lab/E2_예보피처.ipynb

    python src_tools/train_site_model.py --exp 실험번호 --base champion --seeds 2 --add features_weather:add_weather_features

하루 요약 피처 11개를 만든다. 모두 17시 발행분 예보(_r17)만 쓴다.
  그날 (같은 대상일 date 안에서)
    wx_day_cloud     해 뜬 시간(elev > 0) 평균 운량        A사 wa_cloud
    wx_day_trange    일교차 (하루 최고 − 최저 기온)          A사 wa_temp
    wx_day_rain      하루 강수 합                           B사 wb_total_liq
    wx_day_trange_b  일교차                                 B사 wb_temp
    wx_day_rain_a    하루 강수 합                           A사 wa_rain
  전날 (바로 앞 대상일의 위 값 — D−2일 17시 발행이라 D일 예측시점인 D−1일 17시보다 먼저 나와 있다)
    wx_prev_rain, wx_prev_rain_a, wx_prev_cloud, wx_prev_trange
  아침 안개 신호 (같은 대상일 06~10시)
    wx_am_vis_min    최저 시정                              B사 wb_vis
    wx_am_humid      평균 습도                              A사 wa_humidity

기준 모델 E1-01x2 대비 벌금 (같은 해·같은 시드 기준 모델 대비)
  가상 평가연도(2024-06 ~ 2025-05): 시드 0·1 −6.98%, 시드 2·3 −6.65%
  다른 해(2023-06 ~ 2024-05):      시드 0·1 −6.41%, 시드 2·3 −6.38%      → 측정 4개 평균 −6.61%
실측 기상(observed_*)과 다음 대상일 예보는 쓰지 않는다. 행 수와 순서는 바꾸지 않는다.
"""
import pandas as pd

TODAY = ["wx_day_rain", "wx_day_rain_a", "wx_day_cloud", "wx_day_trange"]
PREV = ["wx_prev_rain", "wx_prev_rain_a", "wx_prev_cloud", "wx_prev_trange"]


def add_weather_features(df):
    key = [df.site_id, df.date]

    # 그날 요약
    df["wx_day_cloud"] = df.wa_cloud_r17.where(df.elev > 0).groupby(key).transform("mean")
    t = df.wa_temp_r17.groupby(key)
    df["wx_day_trange"] = t.transform("max") - t.transform("min")
    df["wx_day_rain"] = df.wb_total_liq_r17.groupby(key).transform("sum")
    t = df.wb_temp_r17.groupby(key)
    df["wx_day_trange_b"] = t.transform("max") - t.transform("min")
    df["wx_day_rain_a"] = df.wa_rain_r17.groupby(key).transform("sum")

    # 전날 요약: 발전소 × 대상일 값을 하루 뒤 날짜로 옮겨 붙인다
    d = df.groupby(["site_id", "date"], observed=True)[TODAY].first().reset_index()
    d["date"] = d.date + pd.Timedelta(days=1)
    d = d.rename(columns=dict(zip(TODAY, PREV)))
    m = df[["site_id", "date"]].merge(d, on=["site_id", "date"], how="left")
    for c in PREV:
        df[c] = m[c].to_numpy()

    # 아침 안개 신호: 06~10시
    am = (df.hour >= 6) & (df.hour <= 10)
    df["wx_am_vis_min"] = df.wb_vis_r17.where(am).groupby(key).transform("min")
    df["wx_am_humid"] = df.wa_humidity_r17.where(am).groupby(key).transform("mean")
    return df
