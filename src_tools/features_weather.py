"""② 예보 피처 — 실험 하나 = 함수 하나.

전달 대상은 맨 아래 add_weather_features 하나다(= E2-40, 하루 요약 11개).
    python src_tools/train_site_model.py --exp 실험번호 --base champion --seeds 2 --add features_weather:add_weather_features
나머지 함수는 실험 기록용이다. analysis_lab/E2_예보피처.ipynb 와 outputs/preds/E2-*.json 이 참조하므로 지우지 않는다.
읽기 쉬운 단독 정리본: analysis_lab/E2_handover/features_weather.py (같은 결과)

규칙(README·train_site_model 맨 위 설명):
  - 쓰는 것: 예보 열 {변수}_r10, {변수}_r17 (실측 observed_* 는 base 에 없고 쓰지 않는다)
  - 같은 대상일(date) 다른 시각의 예보는 써도 되지만, 다음 대상일 예보는 예측시점(전일 17시) 이후에 나오므로 쓰지 않는다
  - 행 수와 순서는 바꾸지 않는다. 새 열 이름은 wx_ 로 시작한다
"""
import numpy as np
import pandas as pd


def _in_day(df: pd.DataFrame, col: str, hours: int) -> np.ndarray:
    """같은 발전소의 `hours`시간 뒤(앞이면 음수) 시각 값. 행이 없거나 다른 대상일로 넘어가면 NaN.
    df 가 학습 행(실적 있는 시각만)과 예측 행을 이어 붙인 것이라 shift 대신 (site_id, ts) 로 찾는다."""
    q = df[["site_id", "ts"]].copy()
    q["ts"] = q.ts + pd.Timedelta(hours=hours)
    look = df[["site_id", "ts", col]].drop_duplicates(["site_id", "ts"])
    v = q.merge(look, on=["site_id", "ts"], how="left")[col].to_numpy()
    nb_date = (q.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
    return np.where((nb_date == df.date).to_numpy(), v, np.nan)


def add_ceiling(df):                                           # E2-02 운고
    """구름 바닥이 높을수록 맑다. 기준 모델 입력에 운고가 없다."""
    df["wx_ceiling"] = df.wb_ceiling_r17
    return df


def add_round_summary(df):                                     # E2-03 라운드 요약
    """두 발행분(10시·17시)의 평균과 차이. 차이가 크면 예보가 흔들린다는 신호."""
    for name, col in [("cloud_a", "wa_cloud"), ("cloud_b", "wb_cloud"), ("humid_a", "wa_humidity")]:
        df[f"wx_{name}_avg"] = (df[f"{col}_r10"] + df[f"{col}_r17"]) / 2
        df[f"wx_{name}_gap"] = df[f"{col}_r17"] - df[f"{col}_r10"]
    return df


def add_day_summary(df):                                       # E2-04 하루 요약
    """발전소 × 대상일 요약: 해 뜬 시간 평균 운량, 일교차, 하루 강수 합. 전부 같은 대상일 안의 예보."""
    key = [df.site_id, df.date]
    df["wx_day_cloud"] = df.wa_cloud_r17.where(df.elev > 0).groupby(key).transform("mean")
    t = df.wa_temp_r17.groupby(key)
    df["wx_day_trange"] = t.transform("max") - t.transform("min")
    df["wx_day_rain"] = df.wb_total_liq_r17.groupby(key).transform("sum")
    return df


def add_neighbors(df):                                         # E2-05 앞뒤 시각
    """운량의 1시간 전·후와 3시간 평균. 구름이 지나가는 시점이 어긋나는 오차를 흡수."""
    prev, nxt = _in_day(df, "wa_cloud_r17", -1), _in_day(df, "wa_cloud_r17", 1)
    df["wx_cloud_prev"], df["wx_cloud_next"] = prev, nxt
    df["wx_cloud_3h"] = np.nanmean(np.vstack([prev, df.wa_cloud_r17.to_numpy(), nxt]), axis=0)
    return df


def add_uv_neighbors(df):                                      # E2-05b 자외선지수 앞뒤 시각
    """직전 시각 자외선지수. 우리 쪽 검증에서 중요도 1위였다(ts=13:00 발전량에는 12시 예보가 더 맞는다는 신호)."""
    prev, nxt = _in_day(df, "wb_uv_idx_r17", -1), _in_day(df, "wb_uv_idx_r17", 1)
    df["wx_uv_prev"], df["wx_uv_next"] = prev, nxt
    df["wx_uv_avg2"] = np.nanmean(np.vstack([prev, df.wb_uv_idx_r17.to_numpy()]), axis=0)
    return df


def add_all_sites(df):                                         # E2-06 전 지점 평균
    """같은 시각 모든 발전소 위치의 평균 운량. 지역 전체 날씨."""
    df["wx_all_cloud_a"] = df.wa_cloud_r17.groupby(df.ts).transform("mean")
    df["wx_all_cloud_b"] = df.wb_cloud_r17.groupby(df.ts).transform("mean")
    return df


def add_provider_gap(df):                                      # E2-07 제공사 차이
    """A 운량 − B 운량. 두 제공사가 엇갈리는 정도."""
    df["wx_cloud_diff"] = df.wa_cloud_r17 - df.wb_cloud_r17
    df["wx_cloud_absdiff"] = df.wx_cloud_diff.abs()
    return df


# ── 2라운드: 하루 요약(E2-04)을 쪼개고 넓혀 보기 ──────────────────────────────
def _day_mean(df, col):
    """해 뜬 시간(elev>0)만으로 낸 발전소 × 대상일 평균."""
    return df[col].where(df.elev > 0).groupby([df.site_id, df.date]).transform("mean")


def add_day_cloud(df):                                         # E2-04a 하루 평균 운량만
    df["wx_day_cloud"] = _day_mean(df, "wa_cloud_r17")
    return df


def add_day_trange(df):                                        # E2-04b 일교차만
    t = df.wa_temp_r17.groupby([df.site_id, df.date])
    df["wx_day_trange"] = t.transform("max") - t.transform("min")
    return df


def add_day_rain(df):                                          # E2-04c 하루 강수 합만
    df["wx_day_rain"] = df.wb_total_liq_r17.groupby([df.site_id, df.date]).transform("sum")
    return df


def add_day_cloud_trange(df):                                  # E2-11 하루 요약에서 강수 합 빼기
    return add_day_trange(add_day_cloud(df))


def add_day_summary_rounds(df):                                # E2-14 하루 요약을 두 발행분 평균으로
    key = [df.site_id, df.date]
    cloud = (df.wa_cloud_r10 + df.wa_cloud_r17) / 2
    temp = (df.wa_temp_r10 + df.wa_temp_r17) / 2
    rain = (df.wb_total_liq_r10 + df.wb_total_liq_r17) / 2
    df["wx_day_cloud"] = cloud.where(df.elev > 0).groupby(key).transform("mean")
    t = temp.groupby(key)
    df["wx_day_trange"] = t.transform("max") - t.transform("min")
    df["wx_day_rain"] = rain.groupby(key).transform("sum")
    return df


def add_day_summary_2prov(df):                                 # E2-15 하루 요약을 두 제공사 값으로
    df = add_day_summary(df)
    key = [df.site_id, df.date]
    df["wx_day_cloud_b"] = _day_mean(df, "wb_cloud_r17")
    t = df.wb_temp_r17.groupby(key)
    df["wx_day_trange_b"] = t.transform("max") - t.transform("min")
    df["wx_day_rain_a"] = df.wa_rain_r17.groupby(key).transform("sum")
    return df


def add_day_summary_2prov_nocloud(df):                         # E2-16 E2-15 에서 B사 하루 운량 빼기
    df = add_day_summary(df)
    key = [df.site_id, df.date]
    t = df.wb_temp_r17.groupby(key)
    df["wx_day_trange_b"] = t.transform("max") - t.transform("min")
    df["wx_day_rain_a"] = df.wa_rain_r17.groupby(key).transform("sum")
    return df


def add_day_summary_2prov_dry(df):                             # E2-17 E2-15 + 해 뜬 시간 평균 이슬점 차이
    df = add_day_summary_2prov(df)
    df["wx_day_dpd"] = (df.wb_temp_r17 - df.wb_dew_point_r17).where(df.elev > 0) \
        .groupby([df.site_id, df.date]).transform("mean")
    return df


def add_day_summary_v2(df):                                    # E2-08 하루 요약 + B사·습도·강수확률·자외선
    df = add_day_summary(df)
    df["wx_day_cloud_b"] = _day_mean(df, "wb_cloud_r17")
    df["wx_day_humid"] = _day_mean(df, "wa_humidity_r17")
    df["wx_day_pprob"] = _day_mean(df, "wb_precip_prob_r17")
    df["wx_day_uvmax"] = df.wb_uv_idx_r17.groupby([df.site_id, df.date]).transform("max")
    return df


def add_day_ampm(df):                                          # E2-09 하루 요약 + 오전·오후 운량
    df = add_day_summary(df)
    day = df.elev > 0
    key = [df.site_id, df.date]
    df["wx_am_cloud"] = df.wa_cloud_r17.where(day & (df.hour <= 12)).groupby(key).transform("mean")
    df["wx_pm_cloud"] = df.wa_cloud_r17.where(day & (df.hour > 12)).groupby(key).transform("mean")
    return df


def add_day_dev(df):                                           # E2-10 하루 요약 + 시각 운량의 하루 평균 대비 차이
    df = add_day_summary(df)
    df["wx_cloud_dev_a"] = df.wa_cloud_r17 - df.wx_day_cloud
    df["wx_cloud_dev_b"] = df.wb_cloud_r17 - _day_mean(df, "wb_cloud_r17")
    return df


# ── 6라운드: E2-16 위에 하나씩 ─────────────────────────────────────────────
def _daylight_range(df, col):
    v = df[col].where(df.elev > 0).groupby([df.site_id, df.date])
    return v.transform("max") - v.transform("min")


def add_e16_cloud_std(df):                                     # E2-18 + 해 뜬 시간 운량 표준편차
    df = add_day_summary_2prov_nocloud(df)
    df["wx_day_cloud_std"] = df.wa_cloud_r17.where(df.elev > 0).groupby([df.site_id, df.date]).transform("std")
    return df


def add_e16_rain_hours(df):                                    # E2-19 + 비 온 시간 수 (A·B)
    df = add_day_summary_2prov_nocloud(df)
    key = [df.site_id, df.date]
    df["wx_day_rainhrs_a"] = (df.wa_rain_r17 > 0.1).astype("float32").groupby(key).transform("sum")
    df["wx_day_rainhrs_b"] = (df.wb_total_liq_r17 > 0.1).astype("float32").groupby(key).transform("sum")
    return df


def add_e16_day_trange(df):                                    # E2-20 일교차를 해 뜬 시간으로만 (교체)
    df = add_day_summary_2prov_nocloud(df)
    df["wx_day_trange"] = _daylight_range(df, "wa_temp_r17")
    df["wx_day_trange_b"] = _daylight_range(df, "wb_temp_r17")
    return df


def add_e16_prev_rain(df):                                     # E2-21 + 전날 강수 합 (전 대상일 예보 → 사용 가능)
    df = add_day_summary_2prov_nocloud(df)
    d = df.groupby(["site_id", "date"], observed=True)[["wx_day_rain", "wx_day_rain_a"]].first().reset_index()
    d["date"] = d.date + pd.Timedelta(days=1)                  # 그날 값 → 다음 날의 '전날 값'
    d = d.rename(columns={"wx_day_rain": "wx_prev_rain", "wx_day_rain_a": "wx_prev_rain_a"})
    m = df[["site_id", "date"]].merge(d, on=["site_id", "date"], how="left")
    df["wx_prev_rain"], df["wx_prev_rain_a"] = m.wx_prev_rain.to_numpy(), m.wx_prev_rain_a.to_numpy()
    return df


def add_e16_fleet_day(df):                                     # E2-22 + 전 발전소 하루 평균 (운량·일교차·강수)
    df = add_day_summary_2prov_nocloud(df)
    cols = ["wx_day_cloud", "wx_day_trange", "wx_day_rain"]
    per_site = df.groupby(["site_id", "date"], observed=True)[cols].first().reset_index()
    fleet = per_site.groupby("date")[cols].mean().add_prefix("fleet_").reset_index()
    m = df[["date"]].merge(fleet, on="date", how="left")
    for c in cols:
        df[c.replace("wx_day_", "wx_fleet_")] = m[f"fleet_{c}"].to_numpy()
    return df


def add_e16_pprob(df):                                         # E2-23 + 해 뜬 시간 평균 강수확률 (B)
    df = add_day_summary_2prov_nocloud(df)
    df["wx_day_pprob"] = _day_mean(df, "wb_precip_prob_r17")
    return df


# ── 7라운드: 전날 정보 넓히기 (E2-21 위에) ───────────────────────────────────
def _lag_day(df, cols, days, names):
    """발전소 × 대상일 값(cols)을 `days`일 뒤 날짜로 옮겨 붙인다. 과거 대상일 예보라 예측시점 전에 나와 있다."""
    d = df.groupby(["site_id", "date"], observed=True)[cols].first().reset_index()
    d["date"] = d.date + pd.Timedelta(days=days)
    d = d.rename(columns=dict(zip(cols, names)))
    m = df[["site_id", "date"]].merge(d, on=["site_id", "date"], how="left")
    for n in names:
        df[n] = m[n].to_numpy()
    return df


def add_e21_prev_more(df):                                     # E2-24 + 전날 운량·일교차
    df = add_e16_prev_rain(df)
    return _lag_day(df, ["wx_day_cloud", "wx_day_trange"], 1, ["wx_prev_cloud", "wx_prev_trange"])


def add_e21_prev2_rain(df):                                    # E2-25 + 이틀 전 강수 합
    df = add_e16_prev_rain(df)
    return _lag_day(df, ["wx_day_rain", "wx_day_rain_a"], 2, ["wx_prev2_rain", "wx_prev2_rain_a"])


def add_e21_day_trange(df):                                    # E2-26 E2-21 + 일교차를 해 뜬 시간으로 (교체)
    df = add_e16_prev_rain(df)
    df["wx_day_trange"] = _daylight_range(df, "wa_temp_r17")
    df["wx_day_trange_b"] = _daylight_range(df, "wb_temp_r17")
    return df


# ── 8라운드: 관측지점 예보로 계산 (실적 행이 있는지와 무관하게) ─────────────────
def _station_daily(df, st, agg):
    """(관측지점, 시각)별 예보를 한 번씩만 모아 대상일 요약을 낸다. agg: {새이름: (열, 방법, 해뜬시간만)}.
    여러 발전소 행에서 모으므로, 한 발전소의 실적이 빠진 시각이 있어도 값이 비지 않는다."""
    cols = sorted({c for c, _, _ in agg.values()})
    h = df[[st, "ts", "date", "elev"] + cols].sort_values("elev", ascending=False).drop_duplicates([st, "ts"])
    out = {}
    for name, (c, how, day_only) in agg.items():
        v = h[c].where(h.elev > 0) if day_only else h[c]
        g = v.groupby([h[st], h.date], observed=True)
        out[name] = g.max() - g.min() if how == "range" else g.agg(how)
    return pd.DataFrame(out).reset_index()


def _attach(df, daily, key, lag_days=0, suffix=""):
    d = daily.copy()
    d["date"] = d.date + pd.Timedelta(days=lag_days)
    names = [c for c in d.columns if c not in (key, "date")]
    d = d.rename(columns={c: c + suffix for c in names})
    m = df[[key, "date"]].merge(d, on=[key, "date"], how="left")
    for c in names:
        df[c + suffix] = m[c + suffix].to_numpy()
    return df


def add_station_day(df):                                       # E2-27 = E2-24 와 같은 피처를 관측지점 기준으로
    a = _station_daily(df, "wa_station", {"wx_day_cloud": ("wa_cloud_r17", "mean", True),
                                          "wx_day_trange": ("wa_temp_r17", "range", False),
                                          "wx_day_rain_a": ("wa_rain_r17", "sum", False)})
    b = _station_daily(df, "wb_station", {"wx_day_trange_b": ("wb_temp_r17", "range", False),
                                          "wx_day_rain": ("wb_total_liq_r17", "sum", False)})
    df = _attach(df, a, "wa_station")
    df = _attach(df, b, "wb_station")
    df = _attach(df, a[["wa_station", "date", "wx_day_cloud", "wx_day_trange", "wx_day_rain_a"]], "wa_station", 1, "_prev")
    df = _attach(df, b[["wb_station", "date", "wx_day_rain"]], "wb_station", 1, "_prev")
    return df


def add_station_day_prevb(df):                                 # E2-28 E2-27 + 전날 B사 일교차
    df = add_station_day(df)
    b = _station_daily(df, "wb_station", {"wx_day_trange_b": ("wb_temp_r17", "range", False)})
    return _attach(df, b, "wb_station", 1, "_prev")


# ── 9라운드: E2-24 위에 새 하루 정보 ─────────────────────────────────────────
def add_e24_delta(df):                                         # E2-29 + 오늘 − 전날 (운량·일교차)
    df = add_e21_prev_more(df)
    df["wx_d_cloud"] = df.wx_day_cloud - df.wx_prev_cloud
    df["wx_d_trange"] = df.wx_day_trange - df.wx_prev_trange
    return df


def add_e24_humid(df):                                         # E2-30 + 해 뜬 시간 평균 습도 (오늘·전날)
    df = add_e21_prev_more(df)
    df["wx_day_humid"] = _day_mean(df, "wa_humidity_r17")
    return _lag_day(df, ["wx_day_humid"], 1, ["wx_prev_humid"])


def add_e24_press(df):                                         # E2-31 + 하루 평균 기압과 전날 대비 변화
    df = add_e21_prev_more(df)
    df["wx_day_press"] = df.wa_ground_press_r17.groupby([df.site_id, df.date]).transform("mean")
    df = _lag_day(df, ["wx_day_press"], 1, ["wx_prev_press"])
    df["wx_d_press"] = df.wx_day_press - df.wx_prev_press
    return df


def add_e24_wind(df):                                          # E2-32 + 하루 평균 풍속 (오늘·전날)
    df = add_e21_prev_more(df)
    df["wx_day_wind"] = df.wa_wind_speed_r17.groupby([df.site_id, df.date]).transform("mean")
    return _lag_day(df, ["wx_day_wind"], 1, ["wx_prev_wind"])


def add_e24_dayrain(df):                                       # E2-33 E2-24, 강수 합을 해 뜬 시간만으로 (교체)
    df = add_day_summary_2prov_nocloud(df)
    key = [df.site_id, df.date]
    df["wx_day_rain"] = df.wb_total_liq_r17.where(df.elev > 0, 0).groupby(key).transform("sum")
    df["wx_day_rain_a"] = df.wa_rain_r17.where(df.elev > 0, 0).groupby(key).transform("sum")
    return _lag_day(df, ["wx_day_rain", "wx_day_rain_a", "wx_day_cloud", "wx_day_trange"], 1,
                    ["wx_prev_rain", "wx_prev_rain_a", "wx_prev_cloud", "wx_prev_trange"])


# ── 10라운드: E2-24 에서 묶음별로 빼기 (덜어내도 같은지) ──────────────────────
def add_e24_no_rain_a(df):                                     # E2-34 A사 강수(오늘·전날) 빼기
    return add_e21_prev_more(df).drop(columns=["wx_day_rain_a", "wx_prev_rain_a"])


def add_e24_no_trange_b(df):                                   # E2-35 B사 일교차 빼기
    return add_e21_prev_more(df).drop(columns=["wx_day_trange_b"])


def add_e24_no_prev_cloud(df):                                 # E2-36 전날 운량 빼기
    return add_e21_prev_more(df).drop(columns=["wx_prev_cloud"])


def add_e24_no_prev_trange(df):                                # E2-37 전날 일교차 빼기
    return add_e21_prev_more(df).drop(columns=["wx_prev_trange"])


# ── 11라운드: E2-24 위에 새 가설 ─────────────────────────────────────────────
def add_e24_snow(df):                                          # E2-38 + 강설 합 (오늘·전날, A·B)
    df = add_e21_prev_more(df)
    key = [df.site_id, df.date]
    df["wx_day_snow_a"] = df.wa_snow_r17.groupby(key).transform("sum")
    df["wx_day_snow_b"] = df.wb_snow_r17.groupby(key).transform("sum")
    return _lag_day(df, ["wx_day_snow_a", "wx_day_snow_b"], 1, ["wx_prev_snow_a", "wx_prev_snow_b"])


def add_e24_overnight(df):                                     # E2-39 + 밤사이 강수 (전날 19시 ~ 오늘 06시, A·B)
    df = add_e21_prev_more(df)
    key = [df.site_id, df.date]
    late = (df.hour >= 19) | (df.hour == 0)                    # 대상일의 19~24시 (ts 19:00 ~ 다음날 00:00)
    early = (df.hour >= 1) & (df.hour <= 6)                    # 대상일의 00~06시
    for src, s in [("wb_total_liq_r17", "b"), ("wa_rain_r17", "a")]:
        df[f"_late_{s}"] = df[src].where(late, 0).groupby(key).transform("sum")
        df = _lag_day(df, [f"_late_{s}"], 1, [f"_prevlate_{s}"])
        df[f"wx_night_rain_{s}"] = df[f"_prevlate_{s}"] + df[src].where(early, 0).groupby(key).transform("sum")
    return df.drop(columns=["_late_b", "_prevlate_b", "_late_a", "_prevlate_a"])


def add_e24_fog(df):                                           # E2-40 + 아침 안개 신호 (06~10시 최저 시정 B, 평균 습도 A)
    df = add_e21_prev_more(df)
    key = [df.site_id, df.date]
    am = (df.hour >= 6) & (df.hour <= 10)
    df["wx_am_vis_min"] = df.wb_vis_r17.where(am).groupby(key).transform("min")
    df["wx_am_humid"] = df.wa_humidity_r17.where(am).groupby(key).transform("mean")
    return df


def add_e24_fog_vis(df):                                       # E2-43 아침 안개 신호 중 최저 시정만
    return add_e24_fog(df).drop(columns=["wx_am_humid"])


def add_e24_fog_humid(df):                                     # E2-44 아침 안개 신호 중 아침 습도만
    return add_e24_fog(df).drop(columns=["wx_am_vis_min"])


def add_e24_fog_ceiling(df):                                   # E2-45 E2-40 + 아침 최저 운고 (B)
    df = add_e24_fog(df)
    am = (df.hour >= 6) & (df.hour <= 10)
    df["wx_am_ceiling_min"] = df.wb_ceiling_r17.where(am).groupby([df.site_id, df.date]).transform("min")
    return df


def _fog(df, h0, h1):
    df = add_e21_prev_more(df)
    key = [df.site_id, df.date]
    am = (df.hour >= h0) & (df.hour <= h1)
    df["wx_am_vis_min"] = df.wb_vis_r17.where(am).groupby(key).transform("min")
    df["wx_am_humid"] = df.wa_humidity_r17.where(am).groupby(key).transform("mean")
    return df


def add_e24_fog_0509(df):                                      # E2-47 민감도 점검: 안개 창 05~09시 (선택용 아님)
    return _fog(df, 5, 9)


def add_e24_fog_0711(df):                                      # E2-48 민감도 점검: 안개 창 07~11시 (선택용 아님)
    return _fog(df, 7, 11)


def add_e40_no_prev_trange(df):                                # E2-46 E2-40 에서 전날 일교차 빼기 (단순화, 10개)
    return add_e24_fog(df).drop(columns=["wx_prev_trange"])


def add_e24_neighbors(df, k=10):                               # E2-41 + 주변 발전소 k곳의 하루 요약 평균 (자기 제외)
    import numpy as np
    df = add_e21_prev_more(df)
    sites = df.groupby("site_id", observed=True)[["lat", "lon"]].first()
    lat, lon = np.radians(sites.lat.to_numpy()), np.radians(sites.lon.to_numpy())
    h = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    nb = np.argsort(np.arcsin(np.sqrt(h)), axis=1, kind="stable")[:, 1:k + 1]
    cols = ["wx_day_cloud", "wx_day_trange", "wx_day_rain"]
    daily = df.groupby(["site_id", "date"], observed=True)[cols].first()
    for c in cols:
        p = daily[c].unstack("site_id").reindex(columns=sites.index)       # 날짜 × 발전소
        v = p.to_numpy()
        m = np.nanmean(np.stack([v[:, nb[i]] for i in range(len(sites))], axis=1), axis=2)
        nbv = pd.DataFrame(m, index=p.index, columns=sites.index).stack(dropna=False).rename(f"wx_nb_{c[7:]}")
        mm = df[["date", "site_id"]].merge(nbv.reset_index(), on=["date", "site_id"], how="left")
        df[f"wx_nb_{c[7:]}"] = mm[f"wx_nb_{c[7:]}"].to_numpy()
    return df


def add_e24_drop_wind_press(df):                               # E2-42 시각별 풍속·기압 예보 무력화 (빼기 효과)
    df = add_e21_prev_more(df)
    for c in ["wa_wind_speed_r10", "wa_wind_speed_r17", "wb_wind_speed_r10", "wb_wind_speed_r17",
              "wa_ground_press_r10", "wa_ground_press_r17"]:
        df[c] = np.nan
    return df


def add_weather_features(df):                                  # 11/6 전달본 = E2-40 (analysis_lab/E2_예보피처.ipynb)
    """하루 요약 11개.
    그날: A사 해 뜬 시간 평균 운량, A·B사 일교차, A·B사 하루 강수 합 (같은 대상일 예보)
    전날: A·B사 하루 강수 합, A사 평균 운량, A사 일교차 (전 대상일 예보 — 예측시점 전에 이미 발행됨)
    아침 안개 신호: 06~10시 최저 시정(B), 06~10시 평균 습도(A) (같은 대상일 예보)"""
    return add_e24_fog(df)
