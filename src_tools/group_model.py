"""묶음 직접 모델 (11/11 판 뒤집기 검토 ①·②) — 채점 대상인 묶음 합계를 바로 예측하는 구성원.

    python src_tools/group_model.py                → 세 해 + 평가 기간 예측(outputs/preds_group/GMx2*.parquet) + 앙상블 판정
    python src_tools/group_model.py --submit A      → 고른 판으로 제출 파일(일출·일몰 + 추세 보정)

근거: 지금 구성원은 모두 발전소별 L1(중앙값) 모델이고, 그 합은 묶음 합계의 중앙값이 아니다. 묶음 합계를 직접 맞히는 모델은
    오차 구조가 달라 다양성이 생긴다(지역 합계 예측의 '직접 방식', IEA PVPS T16). 구성원 E11-11 ↔ G5bT1 의 묶음 오차 상관이
    0.985 라 한 자리를 바꿀 여지도 있다(②).
방법(미리 정함):
    구성 = outputs/compositions/E12-10cx2.csv 고정(학습 기간 오차로 정한 것, 규정 §6).
    묶음 × 시각 표: 실적이 있는 발전소의 설비용량 가중 평균 — 지역 예보 A·B(운량·습도·기온·강수·기압·자외선·시정), 태양 cos 천정각·고도,
      + 모든 발전소에 같은 GFS 지역 86개·층별 운량 전체 3개·GEFS 전체 4개(§7), 시각, 묶음 표시. 타깃 = Σ실적 / Σ설비용량(발전율).
    학습 = 검증 시작 전 2년(평가 기간은 2023-06 ~ 2025-05), 가중 고도 > 0 시각. LightGBM L1, 시드 0·1 평균.
    예측 = 발전율 × 그 시각 실적이 있는 발전소 설비용량 합(평가 기간은 전체), 가중 고도 ≤ −2° 이면 0.
판정(미리 정함): 같은 구성으로 E12-10c 대비 세 해 모두 개선. 후보
    A = E12-10c 다섯 + GM (GM 1/6)   B = G5bT1 자리를 GM 으로(1/5)   C = E11-11 자리를 GM 으로(1/5)
    여럿 통과하면 가장 나쁜 해가 가장 덜 나쁜 것.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluate import group_frame, penalty_frame
from oibc_common import OUTPUTS, PREDS, PREDS_TEST, ROOT, TZ, WORK, load_smp

OUT = OUTPUTS / "preds_group"
COMP = OUTPUTS / "compositions" / "E12-10cx2.csv"
FOLDS = {"": "2024-06-01", "_cut2023": "2023-06-01", "_cut2022": "2022-06-01", "test": "2025-06-01"}
SITE_F = ["wa_cloud_r10", "wa_cloud_r17", "wa_humidity_r17", "wa_temp_r17", "wa_rain_r17", "wa_ground_press_r17",
          "wb_cloud_r10", "wb_cloud_r17", "wb_uv_idx_r17", "wb_rel_hum_r17", "wb_temp_r17", "wb_vis_r17", "cosz", "elev"]
PARAMS = dict(objective="regression_l1", n_estimators=2000, learning_rate=0.02, num_leaves=63, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1)
BASE5 = ["E11-12x2", "E11-11x2", "G5bT1x2", "S3x2", "V07T1x2"]


def comp() -> pd.Series:
    c = pd.read_csv(COMP)
    return c.set_index(c.site_id.astype(str)).group


def ext_table() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src_tools"))
    import features_gfs as F
    g = F._gfs()[["ts"] + [c for c in F._gfs().columns if c.startswith("gfs_")]]
    la = F._load("gfs_layers.parquet")
    g = g.merge(la[["ts"] + [c for c in la.columns if c.endswith("_kr")]], on="ts", how="left")
    ge = F._load("gefs_kr.parquet")
    return g.merge(ge[["ts"] + [c for c in ge.columns if c.startswith("gefs_") and c.endswith("_kr")]], on="ts", how="left")


def group_table(rows: pd.DataFrame, cmap: pd.Series, has_target: bool) -> pd.DataFrame:
    d = rows.copy()
    d["group"] = d.site_id.astype(str).map(cmap)
    w = d.capacity_kw
    agg = {"cap": ("capacity_kw", "sum")}
    for c in SITE_F:
        d[f"_w_{c}"] = d[c] * w
        d[f"_c_{c}"] = np.where(d[c].notna(), w, 0.0)
    num = d.groupby(["group", "ts"])[[f"_w_{c}" for c in SITE_F]].sum()
    den = d.groupby(["group", "ts"])[[f"_c_{c}" for c in SITE_F]].sum()
    t = pd.DataFrame({c: num[f"_w_{c}"].to_numpy() / np.where(den[f"_c_{c}"].to_numpy() > 0, den[f"_c_{c}"].to_numpy(), np.nan)
                      for c in SITE_F}, index=num.index)
    t["cap"] = d.groupby(["group", "ts"]).capacity_kw.sum()
    if has_target:
        t["energy"] = d.groupby(["group", "ts"]).energy.sum()
    t = t.reset_index()
    t["g"] = (t.group == "G2").astype(int)
    t["hour"] = t.ts.dt.hour
    return t


def load_rows(start, end, test: bool) -> pd.DataFrame:
    cols = ["site_id", "ts", "capacity_kw"] + SITE_F + ([] if test else ["energy"])
    f = WORK / ("base_test.parquet" if test else "base.parquet")
    a = pd.read_parquet(f, columns=cols)
    a["ts"] = pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    a = a[(a.ts >= start) & (a.ts < end)]
    if not test:
        a = a[a.energy.notna()]
    return a


def fold(suf: str, v: str, cmap, ext) -> pd.DataFrame:
    vs = pd.Timestamp(v, tz=TZ)
    tr = group_table(load_rows(vs - pd.DateOffset(years=2), vs, False), cmap, True).merge(ext, on="ts", how="left")
    pr_rows = load_rows(vs, vs + pd.DateOffset(years=1) + pd.Timedelta(hours=1), suf == "test")
    pr = group_table(pr_rows, cmap, suf != "test").merge(ext, on="ts", how="left")
    feats = [c for c in tr.columns if c not in ("group", "ts", "energy", "cap")]
    tr = tr[(tr.elev > 0) & tr.energy.notna() & (tr.cap > 0)]
    y = (tr.energy / tr.cap).clip(0, 1.2)
    preds = []
    for s in (0, 1):
        m = lgb.LGBMRegressor(**{**PARAMS, "random_state": s}).fit(tr[feats], y)
        preds.append(m.predict(pr[feats]))
    cf = np.clip(np.mean(preds, axis=0), 0, 1.2)
    cf = np.where(pr.elev.to_numpy() > -2, cf, 0.0)
    out = pr[["group", "ts"]].assign(pred_kwh=cf * pr.cap.to_numpy())
    print(f"  {suf or '2024-25'}: 학습 {len(tr):,}행(묶음×시각), 예측 {len(out):,}행, 피처 {len(feats)}개", flush=True)
    return out


def site_group(name: str, suf: str, cmap, actual: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(PREDS / f"{name}{suf}.parquet")
    p["site_id"] = p.site_id.astype(str)
    p["ts"] = pd.to_datetime(p.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    g = group_frame(actual, p, cmap)
    g["ts"] = pd.to_datetime(g.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="묶음 직접 모델")
    ap.add_argument("--submit", default=None, choices=["A", "B", "C"], help="고른 판으로 제출 파일")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cmap = comp()
    if args.submit is None:
        ext = ext_table()
        for suf, v in FOLDS.items():
            f = OUT / f"GMx2{'' if suf == 'test' else suf}{'_test' if suf == 'test' else ''}.parquet"
            if f.exists():
                continue
            fold(suf, v, cmap, ext).to_parquet(f, index=False)
        # 판정
        py = sys.executable
        for out, mem in [("E12-10cxGx2", ["E11-12x2", "E11-11x2", "S3x2", "V07T1x2"]),
                         ("E12-10cxEx2", ["E11-12x2", "G5bT1x2", "S3x2", "V07T1x2"])]:
            for c in ["", "--cut 2023-06-01", "--cut 2022-06-01"]:
                subprocess.run([py, "src_tools/blend.py", "--out", out, *mem, *c.split()], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        smp = load_smp("train")
        sys.path.insert(0, str(ROOT / "src_tools"))
        from compare_years import actual as actual_of
        res = {k: [] for k in ["E12-10c", "A", "B", "C"]}
        for suf, cut in [("", None), ("_cut2023", "2023-06-01"), ("_cut2022", "2022-06-01")]:
            act = actual_of(cut)
            act = act.assign(site_id=act.site_id.astype(str))
            gm = pd.read_parquet(OUT / f"GMx2{suf}.parquet")
            base = site_group("E12-10cx2", suf, cmap, act)
            res["E12-10c"].append(penalty_frame(base, smp).pen.sum())
            for k, (name, w) in {"A": ("E12-10cx2", 1 / 6), "B": ("E12-10cxGx2", 1 / 5), "C": ("E12-10cxEx2", 1 / 5)}.items():
                s = site_group(name, suf, cmap, act).merge(gm.rename(columns={"pred_kwh": "gm"}), on=["group", "ts"], how="left")
                s["fcst"] = (1 - w) * s.fcst + w * s.gm.fillna(s.fcst)
                res[k].append(penalty_frame(s.drop(columns="gm"), smp).pen.sum())
        print("\n같은 구성(E12-10cx2.csv)으로 E12-10c 대비 벌금 변화 (2024-25 / 2023-24 / 2022-23)")
        b = res["E12-10c"]
        best = None
        for k in ["A", "B", "C"]:
            r = [y / x - 1 for x, y in zip(b, res[k])]
            ok = all(v < 0 for v in r)
            print(f"  {k}: " + " / ".join(f"{v * 100:+.2f}%" for v in r) + f"  → {'통과' if ok else '불통과'}")
            if ok and (best is None or max(r) < best[1]):
                best = (k, max(r))
        gm1 = {suf: pd.read_parquet(OUT / f"GMx2{suf}.parquet") for suf in ["", "_cut2023", "_cut2022"]}
        print(f"  (참고) 묶음 직접 모델 단독 vs E12-10c: " + " / ".join(
            f"{(penalty_frame(site_group('E12-10cx2', suf, cmap, actual_of(cut).assign(site_id=lambda d: d.site_id.astype(str))).drop(columns='fcst').merge(g.rename(columns={'pred_kwh': 'fcst'}), on=['group', 'ts']), smp).pen.sum() / x - 1) * 100:+.2f}%"
            for (suf, cut), x, g in zip([("", None), ("_cut2023", "2023-06-01"), ("_cut2022", "2022-06-01")], b, gm1.values())))
        print(f"선택: {best[0] if best else '없음(모두 불통과)'}")
        json.dump({"penalties": res, "best": best[0] if best else None, "created": datetime.now().isoformat(timespec="seconds")},
                  open(OUT / "GM_judge.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return

    # 제출 파일: 사이트 앙상블(빠짐 처리 → 일출·일몰) 의 묶음 합 × (1−w) + 묶음 모델 × w → 추세 보정 r
    k = args.submit
    mem = {"A": BASE5, "B": ["E11-12x2", "E11-11x2", "S3x2", "V07T1x2"], "C": ["E11-12x2", "G5bT1x2", "S3x2", "V07T1x2"]}[k]
    w = 1 / 6 if k == "A" else 1 / 5
    py = sys.executable
    name = f"E16{k}x2"
    subprocess.run([py, "src_tools/blend.py", "--out", f"{name}f", *mem, "--mode", "test"], cwd=ROOT, check=True)
    subprocess.run([py, "src_tools/twilight_fix.py", "--preds", f"{name}f", "--out", f"{name}ft"], cwd=ROOT, check=True)
    p = pd.read_parquet(PREDS_TEST / f"{name}ft.parquet")
    p["site_id"] = p.site_id.astype(str)
    p["group"] = p.site_id.map(cmap)
    s = p.groupby(["group", "ts"]).pred_kwh.sum().reset_index()
    gm = pd.read_parquet(OUT / "GMx2_test.parquet")
    s["ts"] = pd.to_datetime(s.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    gm["ts"] = pd.to_datetime(gm.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    s = s.merge(gm.rename(columns={"pred_kwh": "gm"}), on=["group", "ts"], how="left")
    s["pred_kwh"] = (1 - w) * s.pred_kwh + w * s.gm.fillna(s.pred_kwh)
    sys.path.insert(0, str(ROOT / "src_tools"))
    from trend_fix import trend_ratio
    a = pd.read_parquet(WORK / "base.parquet", columns=["site_id", "ts", "energy", "capacity_kw", "elev", "cosz"])
    a["ts"] = pd.to_datetime(a.ts).dt.tz_convert(TZ).dt.as_unit("ns")
    r, _ = trend_ratio(a, "2025-06-01")
    s["pred_kwh"] = s.pred_kwh * r
    rep = cmap.reset_index().groupby("group").site_id.first()          # 묶음마다 대표 발전소 한 곳에 묶음 합을 실어 make_submission 을 그대로 씀
    pseudo = s.assign(site_id=s.group.map(rep))[["site_id", "ts", "pred_kwh"]]
    pseudo.to_parquet(PREDS_TEST / f"{name}ftrx2.parquet", index=False)
    print(f"추세 보정 r = {r:.4f}, 묶음 모델 비중 {w:.3f}")
    subprocess.run([py, "src_tools/make_submission.py", str(PREDS_TEST / f"{name}ftrx2.parquet"), "--composition", str(COMP),
                    "--name", f"{name}ftrx2_comp"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
