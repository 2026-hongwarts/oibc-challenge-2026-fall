"""F5 — 계획서 v4 의 11개 축 밖에 남은 피처 공간 탐색 (기준 모델 E7-00x2 의 세 해 잔차로).

    python analysis_lab/F5_빠진축_탐색.py

후보 축과 질문
  A 달력(요일·공휴일): 봄철 맑은 주말·휴일에 출력제어로 실적 < 예측인가?
  B 패널 오염: 마지막 비 이후 날수가 길수록 맑은 날 실적 < 예측인가?
  C 눈 덮임: 눈 온 뒤 며칠 동안 실적 < 예측인가?
  D 발전소 × 시각 모양(패널 방향·그늘): 두 해에서 배운 보정이 셋째 해에 통하는가?
  E 묶음 단위 2단계 보정(시각 × 전 지역 운량 구간): 두 해에서 배운 비율 보정이 셋째 해에 통하는가?
판단은 F3 와 같이 "해를 건너 통하는가". 비율 = Σ실적 / Σ예측 − 1 (11~15시처럼 벌금 큰 시각 위주).
예보로 만드는 값만 쓴다(실측·실적은 평가 기간에 없음).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_tools"))
from oibc_common import PREDS, WORK, load_sites, load_smp  # noqa: E402
from train_site_model import cut_window  # noqa: E402

YEARS = [("2024-25", "E7-00x2", "2024-06-01"), ("2023-24", "E7-00x2_cut2023", "2023-06-01"),
         ("2022-23", "E7-00x2_cut2022", "2022-06-01")]
HOL = pd.to_datetime([
    # 2022
    "2022-01-01", "2022-01-31", "2022-02-01", "2022-02-02", "2022-03-01", "2022-03-09", "2022-05-05", "2022-05-08",
    "2022-06-01", "2022-06-06", "2022-08-15", "2022-09-09", "2022-09-10", "2022-09-11", "2022-09-12", "2022-10-03",
    "2022-10-09", "2022-10-10", "2022-12-25",
    # 2023
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24", "2023-03-01", "2023-05-05", "2023-05-27",
    "2023-05-29", "2023-06-06", "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-02", "2023-10-03",
    "2023-10-09", "2023-12-25",
    # 2024
    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12", "2024-03-01", "2024-04-10", "2024-05-05",
    "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-10-01",
    "2024-10-03", "2024-10-09", "2024-12-25",
    # 2025 ~ 2026 (평가 기간 포함)
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-01", "2025-03-03", "2025-05-05",
    "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-05", "2025-10-06", "2025-10-07",
    "2025-10-08", "2025-10-09", "2025-12-25", "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-01",
    "2026-03-02", "2026-05-05", "2026-05-24", "2026-05-25"])
COLS = ["site_id", "ts", "energy", "capacity_kw", "elev", "hour", "date", "month", "is_dead", "wa_station",
        "wa_cloud_r17", "wb_cloud_r17", "wa_rain_r17", "wa_snow_r17", "wb_snow_r17", "wa_temp_r17"]


def load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = pd.read_parquet(WORK / "base.parquet", columns=COLS)
    base["site_id"] = base.site_id.astype(str)
    smp = load_smp("train")[["ts", "w"]]
    comp = load_sites().set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    out = {}
    for y, name, cut in YEARS:
        s, e = cut_window(cut)
        d = base[(base.ts >= s) & (base.ts < e)]
        p = pd.read_parquet(PREDS / f"{name}.parquet")
        p["site_id"] = p.site_id.astype(str)
        d = d.merge(p, on=["site_id", "ts"], how="inner").merge(smp, on="ts", how="left")
        d["g"] = d.site_id.map(comp)
        out[y] = d
    return base, out


def gpen(d: pd.DataFrame, pcol: str = "pred_kwh") -> float:
    g = d.groupby(["g", "ts"]).agg(a=("energy", "sum"), p=(pcol, "sum"), w=("w", "first"))
    return float(0.1 * (g.w * (g.p - g.a).abs()).sum())


def ratio(x: pd.DataFrame) -> float:
    return float(x.energy.sum() / x.pred_kwh.sum() - 1)


def station_day(base: pd.DataFrame) -> pd.DataFrame:
    """지점 × 날짜: 예보 강수 합, 강설 합 (모든 발행 r17). 연속 달력으로 채워 '마지막 비/눈 이후 날수'."""
    st = base.drop_duplicates(["wa_station", "ts"])
    sd = st.groupby(["wa_station", "date"], observed=True).agg(
        rain=("wa_rain_r17", "sum"), snow=("wa_snow_r17", "sum"), snow_b=("wb_snow_r17", "sum")).reset_index()
    rows = []
    for s, g in sd.groupby("wa_station", observed=True):
        g = g.set_index("date").reindex(pd.date_range(g.date.min(), g.date.max(), freq="D"))
        dates = pd.Series(g.index, index=g.index)
        for col, thr, nm in [("rain", 5.0, "dsr"), ("snow", 0.5, "dss")]:
            ev = g[col].fillna(0) >= thr
            last = dates.where(ev).ffill().shift(1)                   # 전날까지의 마지막 사건일
            g[nm] = (dates - last).dt.days
        g["snow3"] = (g.snow.fillna(0) + g.snow_b.fillna(0) / 10).rolling(3, min_periods=1).sum().shift(1)
        g["wa_station"] = s
        rows.append(g.reset_index(names="date"))
    return pd.concat(rows, ignore_index=True)[["wa_station", "date", "dsr", "dss", "snow3"]]


def main() -> None:
    base, Y = load()
    print("기준 벌금 (참조 구성):", {y: f"{gpen(d):,.0f}" for y, d in Y.items()})

    # ── A 달력 ──
    print("\n[A] 요일·공휴일 — 11~15시 전 지역 비율 (실적/예측 − 1)")
    print(f"{'해':8s} {'구간':22s} {'평일':>8s} {'휴일':>8s} {'휴일-평일':>9s} {'휴일 날수':>8s}")
    for y, d in Y.items():
        m = d[d.hour.between(11, 15) & ~d.is_dead.astype(bool)].copy()
        m["off"] = (m.date.dt.dayofweek >= 5) | m.date.isin(HOL)
        cloud = m.groupby("date").wa_cloud_r17.mean()
        for lab, sel in [("전체", m.index == m.index),
                         ("맑음(운량<30)", m.date.map(cloud).lt(30).to_numpy()),
                         ("봄 3~5월 맑음", (m.date.map(cloud).lt(30) & m.month.isin([3, 4, 5])).to_numpy())]:
            x = m[sel]
            on, off = ratio(x[~x.off]), ratio(x[x.off])
            print(f"{y:8s} {lab:22s} {on:+8.3f} {off:+8.3f} {off - on:+9.3f} {x[x.off].date.nunique():8d}")
        # 출력제어 흔적: 맑은 봄 한낮, 발전소-날 실적이 예측의 50% 미만인 비율
        x = m[(m.date.map(cloud).lt(30) & m.month.isin([3, 4, 5])).to_numpy()]
        sdd = x.groupby(["site_id", "date", "off"]).agg(a=("energy", "sum"), p=("pred_kwh", "sum")).reset_index()
        low = (sdd.a < 0.5 * sdd.p)
        print(f"{'':8s} 봄 맑은 한낮 '실적<예측의 50%' 발전소-날 비율: 평일 {low[~sdd.off].mean():.3%} / 휴일 {low[sdd.off].mean():.3%}")

    # ── B 패널 오염 · C 눈 ──
    sd = station_day(base)
    print("\n[B] 마지막 비(예보 하루 강수 ≥ 5 mm) 이후 날수 — 맑은 날(그 지점 한낮 운량<20) 11~15시 비율")
    bins = [0, 1, 3, 7, 14, 30, 999]
    labs = ["1", "2~3", "4~7", "8~14", "15~30", "31+"]
    print(f"{'해':8s} " + " ".join(f"{l:>7s}" for l in labs))
    for y, d in Y.items():
        m = d[d.hour.between(11, 15) & ~d.is_dead.astype(bool)].merge(sd, on=["wa_station", "date"], how="left")
        dc = m.groupby(["wa_station", "date"]).wa_cloud_r17.transform("mean")
        m = m[dc < 20]
        m["b"] = pd.cut(m.dsr, bins, labels=labs)
        r = m.groupby("b", observed=False).apply(lambda x: ratio(x) if len(x) else np.nan, include_groups=False)
        print(f"{y:8s} " + " ".join(f"{v:+7.3f}" for v in r.to_numpy()))
    print("\n[C] 마지막 눈(예보 A 강설 ≥ 0.5) 이후 날수 — 12~2월 11~15시 비율, 직전 3일 강설 합 > 0 인 날 따로")
    labs_c = ["1", "2~3", "4~7", "8+"]
    print(f"{'해':8s} " + " ".join(f"{l:>7s}" for l in labs_c) + f" {'3일눈>0':>8s} {'3일눈=0':>8s}")
    for y, d in Y.items():
        m = d[d.hour.between(11, 15) & ~d.is_dead.astype(bool) & d.month.isin([12, 1, 2])]
        m = m.merge(sd, on=["wa_station", "date"], how="left")
        m["b"] = pd.cut(m.dss, [0, 1, 3, 7, 9999], labels=labs_c)
        r = m.groupby("b", observed=False).apply(lambda x: ratio(x) if len(x) else np.nan, include_groups=False)
        s3 = m.snow3.fillna(0) > 0
        print(f"{y:8s} " + " ".join(f"{v:+7.3f}" for v in r.to_numpy()) + f" {ratio(m[s3]):+8.3f} {ratio(m[~s3]):+8.3f}")

    # ── D 발전소 × 시각 보정 / E 묶음 시각 × 운량 보정 (해 건너 적용) ──
    print("\n[D] 발전소 × 시각 중앙 잔차(설비용량 대비)를 다른 두 해에서 배워 셋째 해에 더함 — 벌금 변화")
    med = {}
    for y, d in Y.items():
        x = d[(d.elev > 0) & ~d.is_dead.astype(bool)]
        med[y] = ((x.energy - x.pred_kwh) / x.capacity_kw).groupby([x.site_id, x.hour]).median()
    for y, d in Y.items():
        others = [med[o] for o in Y if o != y]
        c = pd.concat(others, axis=1).mean(axis=1)                  # 두 해 평균 (한 해만 있으면 그 해)
        for shrink in (1.0, 0.5):
            corr = d.set_index(["site_id", "hour"]).index.map(c).to_numpy(dtype=float)
            corr = np.nan_to_num(corr) * shrink
            p2 = np.where(d.elev > 0, (d.pred_kwh + corr * d.capacity_kw).clip(lower=0), d.pred_kwh)
            print(f"  {y} 수축 {shrink}: {gpen(d.assign(p2=p2), 'p2') / gpen(d) * 100 - 100:+.2f}%")

    print("\n[E] 묶음 × 시각 × 전 지역 운량 구간(5)별 비율 보정을 다른 두 해에서 배워 셋째 해에 곱함 — 벌금 변화")
    tabs = {}
    for y, d in Y.items():
        g = d.groupby(["g", "ts"]).agg(a=("energy", "sum"), p=("pred_kwh", "sum"), c=("wa_cloud_r17", "mean"),
                                         hour=("hour", "first")).reset_index()
        g["cb"] = pd.cut(g.c, [-1, 20, 40, 60, 80, 101], labels=False)
        g = g[g.p > 0]
        tabs[y] = (g.a / g.p - 1).groupby([g.hour, g.cb]).median().clip(-0.3, 0.3)
    for y, d in Y.items():
        c = pd.concat([tabs[o] for o in Y if o != y], axis=1).mean(axis=1)
        cb = pd.cut(d.groupby(["g", "ts"]).wa_cloud_r17.transform("mean"), [-1, 20, 40, 60, 80, 101], labels=False)
        f = pd.MultiIndex.from_arrays([d.hour, cb]).map(c).to_numpy(dtype=float)
        for shrink in (1.0, 0.5):
            p2 = d.pred_kwh * (1 + np.nan_to_num(f) * shrink)
            print(f"  {y} 수축 {shrink}: {gpen(d.assign(p2=p2), 'p2') / gpen(d) * 100 - 100:+.2f}%")


if __name__ == "__main__":
    main()
