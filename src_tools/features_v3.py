"""피처 v3 — 계획서 v4 의 미시험 항목(X01 ~ X16). 실험 하나 = 함수 하나.

기준 모델(E7-00x2)의 피처 함수 add_daily_v1 이 먼저 실행된 df 를 받는다(wx_day_*, wx_prev_* 열이 있음).
예측 행의 발전량은 학습 스크립트가 지워서 넘기므로, 발전량을 쓰는 함수도 학습 행만 보게 된다.
예보는 대상일과 그 이전 대상일 것만 쓴다(다음 대상일 예보는 제출 시점 이후 발행 → 쓰지 않음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features_ours import add_daily_v1
from oibc_common import DATA

TODAY = ["wx_day_cloud", "wx_day_trange", "wx_day_trange_b", "wx_day_rain", "wx_day_rain_a"]
PREV = ["wx_prev_rain", "wx_prev_rain_a", "wx_prev_cloud", "wx_prev_trange"]
AM = ["wx_am_vis_min", "wx_am_humid"]


def _day(df: pd.DataFrame, s: pd.Series, how: str, min_elev: float | None = 0.0) -> pd.Series:
    v = s if min_elev is None else s.where(df.elev > min_elev)
    return v.groupby([df.site_id, df.date]).transform(how)


def _shift_days(df: pd.DataFrame, cols: list[str], names: list[str], days: int = 1) -> pd.DataFrame:
    d = df.groupby(["site_id", "date"], observed=True)[cols].first().reset_index()
    d["date"] = d.date + pd.Timedelta(days=days)
    d = d.rename(columns=dict(zip(cols, names)))
    m = df[["site_id", "date"]].merge(d, on=["site_id", "date"], how="left")
    for n in names:
        df[n] = m[n].to_numpy()
    return df


def _at_offset(df: pd.DataFrame, col: str, hours: int) -> np.ndarray:
    q = df[["site_id", "ts"]].copy()
    q["ts"] = q.ts + pd.Timedelta(hours=hours)
    look = df[["site_id", "ts", col]].drop_duplicates(["site_id", "ts"])
    return q.merge(look, on=["site_id", "ts"], how="left")[col].to_numpy()


def _clear_ghi(df: pd.DataFrame) -> pd.Series:
    """Haurwitz 청천 일사, W/m²."""
    c = df.cosz.clip(lower=0)
    return pd.Series(np.where(c > 0.01, 1098.0 * c * np.exp(-0.057 / c.clip(lower=0.01)), 0.0), index=df.index)


# ── X01 원변수 추가 ─────────────────────────────────────────────────────
EXTRA_RAW = ["wb_ceiling", "wb_feel_temp", "wb_wet_bulb_temp", "wb_wind_gust_spd", "wb_wind_dir",
             "wb_rain_prob", "wb_snow", "wa_wind_dir", "wa_snow"]


def add_raw_extra(df):
    for v in EXTRA_RAW:
        for r in (10, 17):
            df[f"wx_raw_{v}_r{r}"] = df[f"{v}_r{r}"]
    return df


# ── X02 하루 요약 묶음 분해 (--reset-add 와 함께) ─────────────────────────
def daily_no_today(df):
    return add_daily_v1(df).drop(columns=TODAY)


def daily_no_prev(df):
    return add_daily_v1(df).drop(columns=PREV)


def daily_no_am(df):
    return add_daily_v1(df).drop(columns=AM)


# ── X03 1시간 전·후, 3시간 평균 ─────────────────────────────────────────
def add_hour_neighbors(df):
    nxt_other_day = (df.ts.dt.hour == 0).to_numpy()      # ts 00:00 의 다음 시각은 다음 대상일(나중 발행 예보)
    for v in ["wa_cloud_r17", "wa_humidity_r17"]:
        p, n = _at_offset(df, v, -1), _at_offset(df, v, 1)
        n = np.where(nxt_other_day, np.nan, n)
        df[f"wx_{v}_prev1"], df[f"wx_{v}_next1"] = p, n
        df[f"wx_{v}_avg3"] = np.nanmean(np.vstack([p, df[v].to_numpy(), n]), axis=0)
    return df


# ── X04 이틀 전 요약 + 오늘 − 전날 ─────────────────────────────────────
def add_prev2_delta(df):
    df = _shift_days(df, ["wx_day_rain", "wx_day_rain_a", "wx_day_cloud", "wx_day_trange"],
                     ["wx_prev2_rain", "wx_prev2_rain_a", "wx_prev2_cloud", "wx_prev2_trange"], days=2)
    df["wx_d_cloud"] = df.wx_day_cloud - df.wx_prev_cloud
    df["wx_d_trange"] = df.wx_day_trange - df.wx_prev_trange
    return df


# ── X05 밤사이 강수 + 강설 합 ──────────────────────────────────────────
def add_night_snow(df):
    late = (df.hour >= 19) | (df.hour == 0)                    # 그 대상일의 19 ~ 24시
    early = (df.hour >= 1) & (df.hour <= 6)                    # 그 대상일의 0 ~ 6시
    for src, s in [("wa_rain_r17", "a"), ("wb_total_liq_r17", "b")]:
        df[f"_late_{s}"] = _day(df, df[src].where(late, 0), "sum", None)
        df = _shift_days(df, [f"_late_{s}"], [f"_prevlate_{s}"])
        df[f"wx_night_rain_{s}"] = df[f"_prevlate_{s}"] + _day(df, df[src].where(early, 0), "sum", None)
        df = df.drop(columns=[f"_late_{s}", f"_prevlate_{s}"])
    df["wx_day_snow_a"] = _day(df, df.wa_snow_r17, "sum", None)
    df["wx_day_snow_b"] = _day(df, df.wb_snow_r17, "sum", None)
    return _shift_days(df, ["wx_day_snow_a", "wx_day_snow_b"], ["wx_prev_snow_a", "wx_prev_snow_b"])


# ── X06 하루 평균 습도·풍속 ─────────────────────────────────────────────
def add_day_humid_wind(df):
    df["wx_day_humid_a"] = _day(df, df.wa_humidity_r17, "mean")
    df["wx_day_humid_b"] = _day(df, df.wb_rel_hum_r17, "mean")
    df["wx_day_wind_a"] = _day(df, df.wa_wind_speed_r17, "mean")
    return _shift_days(df, ["wx_day_humid_a"], ["wx_prev_humid_a"])


# ── X07 전 지역 평균 (같은 시각, 같은 날) ────────────────────────────────
def add_fleet(df):
    st = df[["wa_station", "ts", "wa_cloud_r17", "wa_humidity_r17"]].drop_duplicates(["wa_station", "ts"])
    f = st.groupby("ts")[["wa_cloud_r17", "wa_humidity_r17"]].mean().add_prefix("wx_fleet_")
    m = df[["ts"]].merge(f, left_on="ts", right_index=True, how="left")
    for c in f.columns:
        df[c] = m[c].to_numpy()
    day = df[["site_id", "date", "wx_day_cloud", "wx_day_trange"]].drop_duplicates(["site_id", "date"])
    g = day.groupby("date")[["wx_day_cloud", "wx_day_trange"]].mean().add_prefix("wx_fleetday_")
    m = df[["date"]].merge(g, left_on="date", right_index=True, how="left")
    for c in g.columns:
        df[c] = m[c].to_numpy()
    return df


# ── X08 가까운 다른 예보 지점 3곳 ───────────────────────────────────────
def add_near_stations(df, k: int = 3):
    cen = df.groupby("wa_station", observed=True)[["lat", "lon"]].mean()
    xy = np.c_[cen.lat.to_numpy(), cen.lon.to_numpy() * np.cos(np.radians(36.0))]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nb = {s: cen.index[np.argsort(dist[i])[:k]].tolist() for i, s in enumerate(cen.index)}
    for v in ["wa_cloud_r17", "wa_humidity_r17"]:
        piv = df[["wa_station", "ts", v]].drop_duplicates(["wa_station", "ts"]).pivot(index="ts", columns="wa_station", values=v)
        near = pd.DataFrame({s: piv[[n for n in nb[s] if n in piv.columns]].mean(axis=1) for s in piv.columns})
        long = near.stack().rename(f"wx_near_{v}").reset_index()
        long.columns = ["ts", "wa_station", f"wx_near_{v}"]
        m = df[["wa_station", "ts"]].merge(long, on=["wa_station", "ts"], how="left")
        df[f"wx_near_{v}"] = m[f"wx_near_{v}"].to_numpy()
    return df


# ── X09 시각 단위 발행분·제공사 결합 ────────────────────────────────────
def add_round_prov_hourly(df):
    for v in ["wa_cloud", "wa_humidity"]:
        df[f"wx_{v}_ravg"] = (df[f"{v}_r10"] + df[f"{v}_r17"]) / 2
        df[f"wx_{v}_rdiff"] = df[f"{v}_r17"] - df[f"{v}_r10"]
    df["wx_prov_cloud_diff"] = df.wa_cloud_r17 - df.wb_cloud_r17
    df["wx_prov_humid_diff"] = df.wa_humidity_r17 - df.wb_rel_hum_r17
    return df


# ── X10 청천 일사·대기외 일사·대기질량 ───────────────────────────────────
def add_clearsky_phys(df):
    df["sol_ghi_clear"] = _clear_ghi(df)
    df["sol_toa"] = 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * df.doy / 365.0)) * df.cosz.clip(lower=0)
    df["sol_airmass"] = np.where(df.cosz > 0.05, 1.0 / df.cosz.clip(lower=0.05), 20.0)
    return df


# ── X11 구름 반영 일사(Kasten–Czeplak) + 패널 셀 온도·효율 ───────────────
def add_irr_celltemp(df):
    cs = _clear_ghi(df)
    df["sol_ghi_kc_a"] = cs * (1 - 0.75 * (df.wa_cloud_r17.clip(0, 100) / 100) ** 3.4)
    df["sol_ghi_kc_b"] = cs * (1 - 0.75 * (df.wb_cloud_r17.clip(0, 100) / 100) ** 3.4)
    df["sol_tcell"] = df.wa_temp_r17 + (45.0 - 20.0) / 800.0 * df.sol_ghi_kc_a      # NOCT 45°C
    df["sol_eff"] = 1 - 0.004 * (df.sol_tcell - 25.0)
    return df


# ── X12 정적 정보 ──────────────────────────────────────────────────────
def add_static_more(df):
    df["site_wa_dist"] = df.wa_dist
    df["site_wb_dist"] = df.wb_dist
    df["site_wa_code"] = df.wa_station.astype(str).str[2:].astype(int)
    df["site_wb_code"] = df.wb_station.astype(str).str[2:].astype(int)
    return df


# ── X13 부분 정지 표시 (--drop-dead 와 함께) ────────────────────────────
def mark_partial_outage(df):
    fit = df.is_fit & (df.elev > 20)
    med = df.cf.where(fit).groupby(df.ts).transform("median")
    df.loc[fit & (df.cf <= 0.02) & (med >= 0.3), "is_dead"] = True
    return df


# ── X16 예보 보정 (MOS) ────────────────────────────────────────────────
MOS = {"wa": [("wa_cloud", "wa_cloud"), ("wa_humidity", "wa_humidity"), ("wa_temp", "wa_temp")],
       "wb": [("wb_cloud", "wb_cloud"), ("wb_rel_hum", "wb_rel_hum"), ("wb_temp", "wb_temp")]}


def add_mos(df):
    """지점 × 시각별 (17시 발행 예보 − 실측) 중앙값을 학습 기간 실측으로 배워 빼 준다.
    실측은 학습 행의 마지막 시각까지만 쓴다(그 뒤 실측은 보지 않음)."""
    last = df.ts[df.is_fit].max()
    for p, pairs in MOS.items():
        st = f"{p}_station"
        obs = pd.read_parquet(DATA / "train" / f"observed_{p}.parquet", columns=[st, "ts"] + [o for _, o in pairs])
        obs["ts"] = pd.to_datetime(obs.ts).dt.as_unit("ns")
        obs = obs[obs.ts <= last]
        if "wb_cloud" in obs:
            obs.loc[obs.wb_cloud > 100, "wb_cloud"] = np.nan
        fc = df[[st, "ts"] + [f"{f}_r17" for f, _ in pairs]].drop_duplicates([st, "ts"])
        fc = fc[fc.ts <= last].merge(obs, on=[st, "ts"])
        fc["hour"] = fc.ts.dt.hour
        for f, o in pairs:
            fc[f"_e_{f}"] = fc[f"{f}_r17"] - fc[o]
        bias = fc.groupby([st, "hour"], observed=True)[[f"_e_{f}" for f, _ in pairs]].median()
        m = df[[st, "hour"]].merge(bias, left_on=[st, "hour"], right_index=True, how="left")
        for f, _ in pairs:
            df[f"wx_mos_{f}"] = df[f"{f}_r17"].to_numpy() - m[f"_e_{f}"].fillna(0).to_numpy()
    return df


def add_mos_daily(df):
    """MOS 보정 운량으로 하루 평균 운량(해 있는 시각)을 다시 만든다."""
    df = add_mos(df)
    df["wx_day_cloud_mos_a"] = _day(df, df.wx_mos_wa_cloud, "mean")
    df["wx_day_cloud_mos_b"] = _day(df, df.wx_mos_wb_cloud, "mean")
    return df


def add_mos_bin(df):
    """지점 × 예보 운량 구간(10% 단위)별 실측 운량 평균 = '이 지점에서 이 예보가 나오면 실제로는 얼마나 흐렸나'.
    학습 행의 마지막 시각까지의 실측만 쓴다."""
    last = df.ts[df.is_fit].max()
    for p in ["wa", "wb"]:
        st, f = f"{p}_station", f"{p}_cloud"
        obs = pd.read_parquet(DATA / "train" / f"observed_{p}.parquet", columns=[st, "ts", f])
        obs["ts"] = pd.to_datetime(obs.ts).dt.as_unit("ns")
        obs = obs[(obs.ts <= last) & (obs[f] <= 100)].rename(columns={f: "_obs"})
        fc = df[[st, "ts", f"{f}_r17"]].drop_duplicates([st, "ts"])
        fc = fc[fc.ts <= last].merge(obs, on=[st, "ts"])
        fc["_bin"] = (fc[f"{f}_r17"].clip(0, 100) / 10).round()
        tab = fc.groupby([st, "_bin"], observed=True)._obs.agg(["mean", "size"])
        tab = tab[tab["size"] >= 30]["mean"].rename("_m")
        q = df[[st]].assign(_bin=(df[f"{f}_r17"].clip(0, 100) / 10).round())
        m = q.merge(tab, left_on=[st, "_bin"], right_index=True, how="left")
        df[f"wx_mosbin_{f}"] = m._m.to_numpy()
    return df


# ── P2 (11/11 파이프라인 점검) B 운량·자외선 1시간 어긋남 ──────────────────────
# B 예보의 자외선 무게중심이 해보다 0.6 ~ 1.2시간 이르고(4개 해 × 12개월 모두), 운량은 ts−2, 자외선은 ts−1 값이
# 실측과 상관이 가장 높다(학습 기간에서 확인, 기온·습도는 정렬됨). → 같은 발전소(같은 B 지점)의 ts−1 값을 덧붙인다.
# 같은 발행분이거나 더 이른 시각 값이라 새 정보 누설 없음(규정 §6).
def add_wb_lag1(df):
    for v in ["wb_cloud_r10", "wb_cloud_r17", "wb_uv_idx_r10", "wb_uv_idx_r17"]:
        if v in df:
            df[f"wx_{v}_lag1"] = _at_offset(df, v, -1)
    return df
