"""F3 잔차 스크리닝 — 현재 기준 모델이 아직 못 쓰는 정보를 다시 학습 없이 찾는다.

    python analysis_lab/F3_잔차스크리닝.py            # 기준 모델(champion), 세 해
    python analysis_lab/F3_잔차스크리닝.py --name E4-08x2

잔차 = (예측 − 실적) / 설비용량 (발전율 단위, +면 과대예측). 범위: 태양고도 > 10°, 정지일 제외.
후보 피처마다 세 해(2024-25 가상 평가연도, 2023-24, 2022-23)에서
    η²잔차 : 후보를 20구간으로 나눴을 때 구간 평균이 잔차 분산을 설명하는 비율 (모델이 놓친 정보의 양)
    흐림 η²: 같은 값을 예보 운량(A) ≥ 90% 시각에서만
을 재고, 세 해 중 가장 작은 값(일관성)으로 줄을 세운다. 한 해에서만 큰 후보는 그해 우연일 수 있다.
결과: 화면, outputs/eda/F3_잔차스크리닝.csv, F3_구간치우침.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_tools"))
from oibc_common import OUTPUTS, PREDS, WORK  # noqa: E402

YEARS = [("2024-25", "", None), ("2023-24", "_cut2023", "2023-06-01"), ("2022-23", "_cut2022", "2022-06-01")]
RAW = ["wa_cloud", "wa_temp", "wa_humidity", "wa_ground_press", "wa_wind_speed", "wa_wind_dir", "wa_rain",
       "wb_temp", "wb_feel_temp", "wb_wet_bulb_temp", "wb_dew_point", "wb_wind_speed", "wb_wind_dir",
       "wb_wind_gust_spd", "wb_rel_hum", "wb_vis", "wb_ceiling", "wb_uv_idx", "wb_precip_prob", "wb_total_liq",
       "wb_cloud"]


def eta2(x: pd.Series, y: pd.Series, bins: int = 20) -> float:
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 1000:
        return np.nan
    b = x if x.nunique() <= bins else pd.qcut(x, bins, labels=False, duplicates="drop")
    m = y.groupby(b).transform("mean")
    return float(1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def add_candidates(d: pd.DataFrame) -> dict[str, pd.Series]:
    """후보 피처. 모두 예보·태양기하·발전소 정적 정보로만 만든다(예측 시점에 알 수 있음)."""
    c = {}
    key = [d.site_id, d.date]
    # 1. B 자외선 예보 ÷ 맑은 날 기대 자외선(Madronich 근사 UVI ≈ 12.5·cosz^2.42) = B사가 본 일사 감쇠
    uv_clear = 12.5 * d.cosz.clip(lower=0) ** 2.42
    c["uv_감쇠비(17)"] = (d.wb_uv_idx_r17 / uv_clear.clip(lower=0.5)).clip(0, 2)
    c["uv_감쇠비(10)"] = (d.wb_uv_idx_r10 / uv_clear.clip(lower=0.5)).clip(0, 2)
    # 2. 구름 바닥 높이 근사: 상승응결고도 ≈ 125 m × (기온 − 이슬점)
    c["LCL(A기온−B이슬점)"] = 125 * (d.wa_temp_r17 - d.wb_dew_point_r17)
    c["운고−LCL"] = d.wb_ceiling_r17 - c["LCL(A기온−B이슬점)"]
    c["운고 구간"] = pd.cut(d.wb_ceiling_r17, [-1, 500, 1000, 2000, 4000, 9000, 1e9], labels=False)
    # 3. 예보 흔들림 (시각)
    c["운량 발행차 |r17−r10| A"] = (d.wa_cloud_r17 - d.wa_cloud_r10).abs()
    c["운량 제공사차 A−B"] = d.wa_cloud_r17 - d.wb_cloud_r17
    c["습도 제공사차 A−B"] = d.wa_humidity_r17 - d.wb_rel_hum_r17
    # 4. 예보 흔들림 (하루)
    day = d.elev > 0
    c["하루 운량 표준편차 A"] = d.wa_cloud_r17.where(day).groupby(key).transform("std")
    c["하루 발행차 평균 A"] = (d.wa_cloud_r17 - d.wa_cloud_r10).abs().where(day).groupby(key).transform("mean")
    c["하루 제공사 운량차 평균"] = (d.wa_cloud_r17 - d.wb_cloud_r17).where(day).groupby(key).transform("mean")
    c["하루 최소 운고 B"] = d.wb_ceiling_r17.where(day).groupby(key).transform("min")
    c["하루 평균 자외선 감쇠비"] = c["uv_감쇠비(17)"].where(d.elev > 15).groupby(key).transform("mean")
    c["하루 최대 강수확률 B"] = d.wb_precip_prob_r17.where(day).groupby(key).transform("max")
    # 5. 흐림 강도 조합
    c["운량×습도/100 A"] = d.wa_cloud_r17 * d.wa_humidity_r17 / 100
    c["이슬점차(B기온−B이슬점)"] = d.wb_temp_r17 - d.wb_dew_point_r17
    # 6. 발전소·시간
    c["발전소 나이(일)"] = (d.ts - d.start).dt.days
    c["월"] = d.month
    c["시각"] = d.hour
    c["태양고도"] = d.elev
    c["용량"] = d.capacity_kw
    c["위도"] = d.lat
    c["경도"] = d.lon
    for v in RAW:
        c[f"[원변수] {v}"] = d[f"{v}_r17"]
    return c


def transfer_screen(frames: dict[str, tuple[pd.Series, pd.Series, dict[str, pd.Series]]]) -> pd.DataFrame:
    """해 건너 보정 시험: 두 해로 '후보 구간별 잔차 중앙값'을 배우고 남은 해에서 그만큼 빼면
    가중 절대 오차(설비용량 × k)가 몇 % 줄어드는지. 음수일수록 좋다. L1 모델에 맞는 지표다."""
    years = list(frames)
    names = list(next(iter(frames.values()))[2])
    rows = []
    for c in ["(구간 없음: 전체 중앙값 보정)"] + names:
        rec = {"후보": c}
        for h in years:
            tr = [y for y in years if y != h]
            r_tr = pd.concat([frames[y][0] for y in tr])
            r_h, wt_h, cand_h = frames[h]
            if c.startswith("("):
                corr = pd.Series(r_tr.median(), index=r_h.index)
            else:
                v_tr = pd.concat([frames[y][2][c] for y in tr])
                v_h = cand_h[c]
                ok = v_tr.notna()
                edges = np.unique(np.nanquantile(v_tr[ok], np.linspace(0, 1, 21)))
                if len(edges) < 3:
                    rec[h] = np.nan
                    continue
                b_tr = np.clip(np.searchsorted(edges, v_tr[ok], side="right") - 1, 0, len(edges) - 2)
                med = r_tr[ok].groupby(b_tr).median()
                b_h = np.clip(np.searchsorted(edges, v_h.fillna(edges[0]), side="right") - 1, 0, len(edges) - 2)
                corr = pd.Series(med.reindex(b_h).fillna(0).to_numpy(), index=r_h.index).where(v_h.notna(), 0.0)
            before = (wt_h * r_h.abs()).sum()
            after = (wt_h * (r_h - corr).abs()).sum()
            rec[h] = (after / before - 1) * 100
        rows.append(rec)
    t = pd.DataFrame(rows).set_index("후보")
    t["평균"] = t[years].mean(axis=1)
    t["최악"] = t[years].max(axis=1)
    return t.sort_values("평균")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--transfer-only", action="store_true", help="해 건너 보정 시험만")
    args = ap.parse_args()
    name = args.name or (PREDS / "champion.txt").read_text(encoding="utf-8").strip()
    cols = (["site_id", "ts", "date", "month", "hour", "elev", "cosz", "energy", "capacity_kw", "is_dead", "start",
             "lat", "lon"] + [f"{v}_r{r}" for v in RAW for r in (10, 17)])
    base = pd.read_parquet(WORK / "base.parquet", columns=cols)
    base["site_id"] = base.site_id.astype(str)
    rows, bias_tabs, frames = [], [], {}
    for label, suffix, cut in YEARS:
        p = pd.read_parquet(PREDS / f"{name}{suffix}.parquet")
        p["site_id"] = p.site_id.astype(str)
        p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
        d = base.merge(p, on=["site_id", "ts"])
        d = d.sort_values(["site_id", "ts"]).reset_index(drop=True)
        cand = add_candidates(d)
        m = (d.elev > 10) & ~d.is_dead
        res = ((d.pred_kwh - d.energy) / d.capacity_kw)[m]
        over = m & (d.wa_cloud_r17 >= 90)
        res_o = ((d.pred_kwh - d.energy) / d.capacity_kw)[over]
        print(f"[{label}] 행 {int(m.sum()):,} (흐림 {int(over.sum()):,}), 잔차 평균 {res.mean():+.4f}, MAE {res.abs().mean():.4f}",
              flush=True)
        wt = (d.capacity_kw * np.where(d.hour.between(11, 15), 3, 1))[m]
        frames[label] = (res.reset_index(drop=True), wt.reset_index(drop=True),
                         {k: v[m].reset_index(drop=True) for k, v in cand.items()})
        if args.transfer_only:
            continue
        for k, v in cand.items():
            rows.append({"후보": k, "해": label, "η²잔차": eta2(v[m], res), "흐림 η²": eta2(v[over], res_o)})
        # 예보 운량 × 습도 구간별 치우침
        cb = pd.cut(d.wa_cloud_r17[m], [0, 30, 70, 90, 101], right=False)
        hb = pd.cut(d.wa_humidity_r17[m], [0, 50, 70, 85, 101], right=False)
        t = res.groupby([cb, hb], observed=True).mean().unstack() * 100
        t.index = t.index.astype(str); t.columns = t.columns.astype(str)
        bias_tabs.append(t.assign(해=label))
    pd.set_option("display.width", 220)
    tt = transfer_screen(frames)
    print("\n=== 해 건너 보정 시험: 두 해로 배운 구간별 중앙값 보정을 남은 해에 적용했을 때 가중 절대 오차 변화 % ===")
    print("    (음수 = 남은 해에서도 오차가 준다 = 모델이 놓친, 해가 바뀌어도 통하는 정보)")
    print(tt.round(2).to_string())
    OUTPUTS.joinpath("eda").mkdir(parents=True, exist_ok=True)
    tt.to_csv(OUTPUTS / "eda" / "F3_해건너보정.csv", encoding="utf-8-sig")
    if args.transfer_only:
        return
    r = pd.DataFrame(rows)
    piv = r.pivot_table(index="후보", columns="해", values=["η²잔차", "흐림 η²"])
    piv[("최소", "η²잔차")] = piv["η²잔차"].min(axis=1)
    piv[("최소", "흐림 η²")] = piv["흐림 η²"].min(axis=1)
    piv = piv.sort_values(("최소", "η²잔차"), ascending=False)
    pd.set_option("display.width", 220)
    print("\n=== 후보별 잔차 설명력 (세 해, 최소값 순) ×1000 ===")
    print((piv * 1000).round(2).to_string())
    OUTPUTS.joinpath("eda").mkdir(parents=True, exist_ok=True)
    piv.to_csv(OUTPUTS / "eda" / "F3_잔차스크리닝.csv", encoding="utf-8-sig")
    print("\n=== 예보 운량(행) × 습도(열) 구간별 평균 잔차 (발전율 %p, +면 과대예측) ===")
    for t in bias_tabs:
        print(f"\n{t.해.iloc[0]}")
        print(t.drop(columns="해").round(2).to_string())
    pd.concat(bias_tabs).to_csv(OUTPUTS / "eda" / "F3_구간치우침.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
