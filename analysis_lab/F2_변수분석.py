"""F2 변수 분석 — 모든 예보·실측·발전소 변수의 값 특징, 발전량과의 관계, 중복, 예보 정확도.

    python analysis_lab/F2_변수분석.py     # 결과 표는 화면과 outputs/eda/F2_*.csv

관계 분석 범위: 2023-06 ~ 2025-05, 태양고도 > 10°, 정지일 제외.
    CSI(맑은날 대비 발전율) = cf / (발전소, 월, 시간)별 cf 95% 분위수.
    ρ_s  : CSI 와의 스피어만 순위상관 (방향과 단조 관계의 세기)
    η²   : 변수를 20개 구간으로 나눴을 때 구간 평균이 CSI 분산을 설명하는 비율 (비선형 포함 설명력)
    추가 η²: 운량(A)·습도(A) 10×10 구간 평균을 뺀 나머지를 이 변수가 설명하는 비율 (새 정보의 양)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_tools"))
from oibc_common import DATA, OUTPUTS, PREDS, WORK, load_forecast, load_sites  # noqa: E402

OUT = OUTPUTS / "eda"
WA = ["wa_cloud", "wa_temp", "wa_temp_max", "wa_temp_min", "wa_humidity", "wa_ground_press",
      "wa_wind_speed", "wa_wind_dir", "wa_rain", "wa_snow"]
WB = ["wb_temp", "wb_feel_temp", "wb_wet_bulb_temp", "wb_dew_point", "wb_wind_speed", "wb_wind_dir",
      "wb_wind_gust_spd", "wb_rel_hum", "wb_vis", "wb_ceiling", "wb_uv_idx", "wb_precip_prob",
      "wb_rain_prob", "wb_snow_prob", "wb_ice_prob", "wb_total_liq", "wb_rain", "wb_snow", "wb_ice", "wb_cloud"]
BASE_WA = [c for c in WA if c not in ("wa_temp_max", "wa_temp_min")]      # base.parquet 에 있는 A 변수
OBS_WB = ["wb_temp", "wb_feel_temp", "wb_feel_temp_shade", "wb_rel_hum", "wb_dew_point", "wb_wind_dir",
          "wb_wind_speed", "wb_uv_idx", "wb_vis", "wb_cloud", "wb_ceiling", "wb_pressure", "wb_appr_temp",
          "wb_wind_chill_temp", "wb_precip_1h"]
OBS_BAD = {"wb_cloud": (None, 100), "wb_uv_idx": (None, 15), "wb_vis": (None, 50), "wb_wind_speed": (None, 200),
           "wb_pressure": (900, None), "wa_wind_speed": (None, 50)}           # README 이상값 → NaN


def describe(s: pd.Series) -> dict:
    v = s.dropna()
    u = np.unique(v.round(4).to_numpy())
    step = float(np.median(np.diff(u))) if len(u) > 1 else np.nan
    q = v.quantile([0.01, 0.5, 0.99]).to_numpy() if len(v) else [np.nan] * 3
    return {"NaN%": s.isna().mean() * 100, "min": v.min(), "p1": q[0], "중앙": q[1], "p99": q[2], "max": v.max(),
            "평균": v.mean(), "0 비율%": (v == 0).mean() * 100, "서로 다른 값": len(u), "값 간격": step}


def eta2(x: pd.Series, y: pd.Series, bins: int = 20) -> float:
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if x.nunique() <= bins:
        b = x
    else:
        b = pd.qcut(x, bins, labels=False, duplicates="drop")
    m = y.groupby(b).transform("mean")
    return float(1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def part_dictionary() -> None:
    section("A. 예보 변수 값 특징 (학습 구간 전체 시각, round 17) + 평가 구간 평균")
    rows = []
    for p, cols in [("wa", WA), ("wb", WB)]:
        tr = load_forecast(p, "train", cols)
        te = load_forecast(p, "test", cols)
        tr, te = tr[tr["round"] == 17], te[te["round"] == 17]
        for c in cols:
            d = describe(tr[c])
            d.update({"변수": c, "평가 구간 평균": te[c].mean(),
                      "학습→평가 평균 차(표준편차 단위)": (te[c].mean() - tr[c].mean()) / (tr[c].std() or np.nan)})
            rows.append(d)
    t = pd.DataFrame(rows).set_index("변수")
    print(t.round(3).to_string())
    t.to_csv(OUT / "F2_A_예보값.csv", encoding="utf-8-sig")
    fa = load_forecast("wa", "train", ["wa_temp", "wa_temp_max", "wa_temp_min"])
    print(f"wa_temp == wa_temp_max == wa_temp_min 인 행 비율: "
          f"{((fa.wa_temp == fa.wa_temp_max) & (fa.wa_temp == fa.wa_temp_min)).mean() * 100:.2f}%")


def load_rel_frame() -> pd.DataFrame:
    cols = (["site_id", "ts", "date", "month", "hour", "elev", "cf", "is_dead", "wa_station", "wb_station"]
            + [f"{c}_r{r}" for c in BASE_WA + WB for r in (10, 17)])
    b = pd.read_parquet(WORK / "base.parquet", columns=cols)
    b = b[(b.ts >= pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul")) & (b.elev > 10) & ~b.is_dead
          & b.cf.notna()].copy()
    ref = b.groupby(["site_id", "month", "hour"], observed=True).cf.quantile(0.95).rename("ref")
    b = b.join(ref, on=["site_id", "month", "hour"])
    b["csi"] = (b.cf / b.ref.clip(lower=0.02)).clip(0, 1.5)
    return b.reset_index(drop=True)


def part_relation(b: pd.DataFrame) -> None:
    section("B. 발전량(CSI)과의 관계 — round 17, 고도 > 10°")
    y = b.csi
    cb = pd.qcut(b.wa_cloud_r17, 10, labels=False, duplicates="drop")
    hb = pd.qcut(b.wa_humidity_r17, 10, labels=False, duplicates="drop")
    resid = y - y.groupby([cb, hb]).transform("mean")
    print(f"운량(A)×습도(A) 10×10 구간 평균이 설명하는 CSI 분산: {1 - resid.var() / y.var():.3f}")
    rows = []
    for c in BASE_WA + WB:
        x = b[f"{c}_r17"]
        rows.append({"변수": c, "ρ_s": x.corr(y, method="spearman"), "η²": eta2(x, y),
                     "추가 η²(운량·습도 뺀 뒤)": eta2(x, resid),
                     "ρ_s round10": b[f"{c}_r10"].corr(y, method="spearman")})
    t = pd.DataFrame(rows).set_index("변수").sort_values("η²", ascending=False)
    print(t.round(4).to_string())
    t.to_csv(OUT / "F2_B_관계.csv", encoding="utf-8-sig")

    section("B2. 시간대별 운량(A)·습도(A) 효과 — 11~15시와 나머지")
    for name, m in [("11~15시", b.hour.between(11, 15)), ("나머지", ~b.hour.between(11, 15))]:
        print(f"  {name}: η² 운량 {eta2(b.wa_cloud_r17[m], y[m]):.3f}, 습도 {eta2(b.wa_humidity_r17[m], y[m]):.3f}, "
              f"B 운량 {eta2(b.wb_cloud_r17[m], y[m]):.3f}, 운고 {eta2(b.wb_ceiling_r17[m], y[m]):.3f}")

    section("B3. 운량(A) 구간별 CSI (중앙값, IQR)")
    q = y.groupby(pd.cut(b.wa_cloud_r17, [0, 10, 30, 50, 70, 90, 100.1], right=False), observed=True)
    print(q.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]].round(3).to_string())


def part_redundancy(b: pd.DataFrame) -> None:
    section("C. 변수끼리의 중복 — round 17 스피어만 |ρ| ≥ 0.85 쌍")
    cols = [f"{c}_r17" for c in BASE_WA + WB]
    s = b[cols].sample(300_000, random_state=0)
    r = s.corr(method="spearman")
    pairs = [(a.removesuffix("_r17"), c.removesuffix("_r17"), r.loc[a, c])
             for i, a in enumerate(cols) for c in cols[i + 1:] if abs(r.loc[a, c]) >= 0.85]
    for a, c, v in sorted(pairs, key=lambda x: -abs(x[2])):
        print(f"  {a:18s} ~ {c:18s} {v:+.3f}")
    r.to_csv(OUT / "F2_C_상관행렬.csv", encoding="utf-8-sig")

    section("C2. 발행 라운드 10시 vs 17시 — 같은 변수의 차이 (고도 > 10°)")
    rows = []
    for c in BASE_WA + WB:
        a, z = b[f"{c}_r10"], b[f"{c}_r17"]
        rows.append({"변수": c, "ρ(r10, r17)": a.corr(z), "평균 |차|": (z - a).abs().mean(),
                     "차이 있는 행%": ((z - a).abs() > 1e-6).mean() * 100})
    t = pd.DataFrame(rows).set_index("변수")
    print(t.round(3).to_string())
    t.to_csv(OUT / "F2_C2_라운드.csv", encoding="utf-8-sig")


def part_skill(b: pd.DataFrame) -> None:
    section("D. 예보 정확도 — 실측과 비교 (8~17시, README 이상값 제외)")
    owa = pd.read_parquet(DATA / "train" / "observed_wa.parquet")
    owb = pd.read_parquet(DATA / "train" / "observed_wb.parquet")
    for o in (owa, owb):
        o["ts"] = pd.to_datetime(o.ts).dt.as_unit("ns")
        for c, (lo, hi) in OBS_BAD.items():
            if c in o:
                if lo is not None: o.loc[o[c] < lo, c] = np.nan
                if hi is not None: o.loc[o[c] > hi, c] = np.nan
    rows = []
    pairs = [("wa", owa, c, c) for c in WA] + [("wb", owb, c, c) for c in WB if c in OBS_WB] \
        + [("wb", owb, "wb_total_liq", "wb_precip_1h")]
    for p, o, fc, oc in pairs:
        f = load_forecast(p, "train", [fc])
        st = f"{p}_station"
        for rnd in (10, 17):
            m = f[(f["round"] == rnd) & f.ts.dt.hour.between(8, 17)][[st, "ts", fc]].merge(
                o[[st, "ts", oc]].rename(columns={oc: "obs"}), on=[st, "ts"])
            e = (m[fc] - m.obs).dropna()
            rows.append({"예보 변수": fc, "실측 변수": oc, "round": rnd, "상관": m[fc].corr(m.obs),
                         "MAE": e.abs().mean(), "평균 오차(예보−실측)": e.mean(), "n": len(e)})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    t.to_csv(OUT / "F2_D_예보정확도.csv", index=False, encoding="utf-8-sig")

    section("D2. 실측을 알았다면? — 실측 변수와 CSI 의 η² (예보 η² 와 비교)")
    g = b[["wa_station", "wb_station", "ts", "csi"]]
    ga = g.merge(owa[["wa_station", "ts", "wa_cloud", "wa_humidity", "wa_temp"]], on=["wa_station", "ts"])
    gb = g.merge(owb[["wb_station", "ts", "wb_cloud", "wb_uv_idx", "wb_vis", "wb_ceiling", "wb_rel_hum",
                      "wb_precip_1h"]], on=["wb_station", "ts"])
    for frame, cols in [(ga, ["wa_cloud", "wa_humidity", "wa_temp"]),
                        (gb, ["wb_cloud", "wb_uv_idx", "wb_vis", "wb_ceiling", "wb_rel_hum", "wb_precip_1h"])]:
        for c in cols:
            fc = "wb_total_liq" if c == "wb_precip_1h" else c
            print(f"  실측 {c:15s} η² {eta2(frame[c], frame.csi):.3f}   |  예보 {fc}_r17 η² {eta2(b[f'{fc}_r17'], b.csi):.3f}")


def part_sites() -> None:
    section("E. 발전소 변수 — 정적 정보와 기준 모델 오차의 관계 (가상 평가연도, 11~15시)")
    s = load_sites().set_index("site_id")
    name = (PREDS / "champion.txt").read_text(encoding="utf-8").strip()
    p = pd.read_parquet(PREDS / f"{name}.parquet")
    p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
    b = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "hour", "energy", "start", "split"])
    b = b[(b.split == "valid") & b.hour.between(11, 15)]
    b["site_id"] = b.site_id.astype(str)
    m = b.merge(p.assign(site_id=p.site_id.astype(str)), on=["site_id", "ts"])
    m["ae"] = (m.pred_kwh - m.energy).abs()
    m["err"] = m.pred_kwh - m.energy
    per = m.groupby("site_id").agg(ae=("ae", "sum"), err=("err", "sum"), a=("energy", "sum"), start=("start", "first"))
    per["nMAE%"] = per.ae / per.a * 100
    per["치우침%"] = per.err / per.a * 100
    per = per.join(s[["capacity_kw", "lat", "lon", "wa_dist", "wb_dist"]])
    per["이력(일, 2024-06 기준)"] = (pd.Timestamp("2024-06-01", tz="Asia/Seoul") - per.start).dt.days
    print(per[["nMAE%", "치우침%"]].describe().round(2).to_string())
    for c in ["capacity_kw", "lat", "lon", "wa_dist", "wb_dist", "이력(일, 2024-06 기준)"]:
        print(f"  ρ_s(nMAE, {c}) = {per['nMAE%'].corr(per[c], method='spearman'):+.3f}")
    print("\n  오차가 가장 큰 발전소 8곳:")
    print(per.sort_values("nMAE%", ascending=False).head(8)[["nMAE%", "치우침%", "capacity_kw", "wa_dist", "wb_dist",
                                                             "이력(일, 2024-06 기준)"]].round(1).to_string())
    per.to_csv(OUT / "F2_E_발전소오차.csv", encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    part_dictionary()
    b = load_rel_frame()
    print(f"\n관계 분석 행: {len(b):,}")
    part_relation(b)
    part_redundancy(b)
    part_skill(b)
    part_sites()


if __name__ == "__main__":
    main()
