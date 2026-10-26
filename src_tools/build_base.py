"""공통 테이블 만들기 — 모든 실험이 같은 데이터에서 출발하도록.

    python src_tools/build_base.py    # outputs/work/base.parquet, base_test.parquet 생성 (메모리 약 6GB)

base.parquet       학습 구간 발전량의 모든 행(2021-01 ~ 2025-06-01 00:00).
                   split = 'train'(< 2024-06-01 01:00) | 'valid'(가상 평가연도)
base_test.parquet  실제 평가 기간 253개소 × 8,760시각. split = 'test', 발전량 열 없음

열
  키        site_id, ts(KST, 우측닫힘), date(ts-1h 의 날짜), split
  타깃      energy(kWh), cf(= energy / capacity_kw), is_dead(정지 추정일)      ← base 만
  발전소    capacity_kw, lat, lon, wa_station, wb_station, wa_dist, wb_dist, start(첫 발전 시각)
  태양·시간 elev, azim, cosz, hour, month, doy, doy_sin, doy_cos
  예보      {변수}_r10, {변수}_r17 — 제공사 A 8개, B 20개.
            (지점, 시각)에서 한 라운드만 빠졌으면 다른 라운드 값으로 채웠다. 둘 다 없으면 NaN.
            wa_temp_max·wa_temp_min 은 하루전 예보에서 wa_temp 와 같아(README) 뺐다.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from oibc_common import (DATA, WORK, VALID_FROM, dead_days, day_of, fill_round_gaps,
                         load_forecast, load_sites, load_smp, solar)

WA_VARS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_ground_press", "wa_wind_speed",
           "wa_wind_dir", "wa_rain", "wa_snow"]
WB_VARS = ["wb_temp", "wb_feel_temp", "wb_wet_bulb_temp", "wb_dew_point", "wb_wind_speed",
           "wb_wind_dir", "wb_wind_gust_spd", "wb_rel_hum", "wb_vis", "wb_ceiling", "wb_uv_idx",
           "wb_precip_prob", "wb_rain_prob", "wb_snow_prob", "wb_ice_prob", "wb_total_liq",
           "wb_rain", "wb_snow", "wb_ice", "wb_cloud"]
META = ["site_id", "capacity_kw", "lat", "lon", "wa_station", "wb_station", "wa_dist", "wb_dist"]

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def add_time_solar(d: pd.DataFrame) -> pd.DataFrame:
    d["elev"], d["azim"], d["cosz"] = solar(d.ts, d.lat.to_numpy(), d.lon.to_numpy())
    d["date"] = day_of(d.ts).to_numpy()
    d["hour"] = d.ts.dt.hour.astype("int8")
    d["month"] = d.ts.dt.month.astype("int8")
    d["doy"] = d.ts.dt.dayofyear.astype("int16")
    d["doy_sin"] = np.sin(2 * np.pi * d.doy / 365.25).astype("float32")
    d["doy_cos"] = np.cos(2 * np.pi * d.doy / 365.25).astype("float32")
    return d


def wide_forecast(provider: str, split: str, index: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    """(지점, 시각)당 한 행: {변수}_r10, {변수}_r17."""
    st = f"{provider}_station"
    f = fill_round_gaps(load_forecast(provider, split, cols), provider, index)
    w = f.pivot(index=[st, "ts"], columns="round")
    w.columns = [f"{v}_r{r}" for v, r in w.columns]
    return w.astype("float32").reset_index()


def attach_forecasts(d: pd.DataFrame, split: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    for provider, cols in [("wa", WA_VARS), ("wb", WB_VARS)]:
        w = wide_forecast(provider, split, index, cols)
        d = d.merge(w, on=[f"{provider}_station", "ts"], how="left")
        log(f"  {split} 예보 {provider.upper()} 병합: 지점×시각 {len(w):,}행")
        del w
    return d


def finish(d: pd.DataFrame) -> pd.DataFrame:
    for c in ["site_id", "wa_station", "wb_station", "split"]:
        d[c] = d[c].astype("category")
    for c in ["capacity_kw", "lat", "lon", "wa_dist", "wb_dist"]:
        d[c] = d[c].astype("float32")
    return d.sort_values(["site_id", "ts"], ignore_index=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    sites = load_sites()
    gen = pd.read_parquet(DATA / "train" / "generation.parquet")
    gen["ts"] = pd.to_datetime(gen.ts).dt.as_unit("ns")
    meta = sites[META].merge(gen.groupby("site_id").ts.min().rename("start").reset_index(), on="site_id")

    # ── base: 학습 구간 ────────────────────────────────────────────────
    b = add_time_solar(gen.merge(meta, on="site_id", how="left"))
    b["energy"] = b.energy.astype("float32")
    b["cf"] = (b.energy / b.capacity_kw).astype("float32")
    dead = dead_days(b)
    b = b.merge(dead[["site_id", "date"]].assign(_d=1), on=["site_id", "date"], how="left")
    b["is_dead"] = b.pop("_d").notna()
    b["split"] = np.where(b.ts >= VALID_FROM, "valid", "train")
    log(f"base 발전량 {len(b):,}행, 정지 추정 site-day {len(dead):,}개")
    train_idx = pd.date_range(b.ts.min(), b.ts.max(), freq="h")
    b = finish(attach_forecasts(b, "train", train_idx))
    b.to_parquet(WORK / "base.parquet", index=False, compression="zstd")
    log(f"저장 outputs/work/base.parquet  {b.shape}  (train {int((b.split == 'train').sum()):,} / valid {int((b.split == 'valid').sum()):,})")
    fc = [c for c in b.columns if c.endswith(("_r10", "_r17"))]
    log(f"  예보 열 {len(fc)}개, 예보 결측 비율 최대 {b[fc].isna().mean().max() * 100:.2f}%")
    del b

    # ── base_test: 실제 평가 기간 격자 ─────────────────────────────────
    test_idx = pd.DatetimeIndex(load_smp("test").ts)
    t = pd.MultiIndex.from_product([sites.site_id, test_idx], names=["site_id", "ts"]).to_frame(index=False)
    t = add_time_solar(t.merge(meta, on="site_id", how="left"))
    t["split"] = "test"
    t = finish(attach_forecasts(t, "test", test_idx))
    t.to_parquet(WORK / "base_test.parquet", index=False, compression="zstd")
    fc = [c for c in t.columns if c.endswith(("_r10", "_r17"))]
    log(f"저장 outputs/work/base_test.parquet  {t.shape}, 예보 결측 행 {int(t[fc].isna().any(axis=1).sum())}")
    assert len(t) == len(sites) * len(test_idx)


if __name__ == "__main__":
    main()
