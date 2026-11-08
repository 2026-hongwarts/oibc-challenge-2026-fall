"""F9 — 최종 피처(E9 계열 54개) 분석. 학습 기간 세 fold 만 쓴다(평가 기간으로는 아무것도 판단하지 않음, 규정 §6).

    python analysis_lab/F9_피처분석.py

fold 마다 LightGBM(E9-00 설정, 시드 0)을 직전 2년으로 학습하고 다음 1년을 예측한 뒤
  ① 벌금 기준 순열 중요도: 검증 연도에서 피처 묶음(같은 변수의 두 발행분, 앞뒤 시각 세 개)을 같은 순서로 섞었을 때 대회 벌금 증가율
  ② SHAP(LightGBM pred_contrib): 검증 연도 표본 2만 행의 평균 |기여|(발전율 단위)와 방향(피처 값과 기여의 순위 상관)
  ③ 나무 분할 이득 비율
  ④ 한낮(11~15시) 발전율과의 순위 상관을 학습 1년차 / 2년차 / 검증 연도 따로 → 부호가 뒤집히는지
  ⑤ 피처끼리 순위 상관 |ρ| ≥ 0.95 쌍 (첫 fold 학습 행)
  ⑥ 상위 피처의 10분위별 한낮 발전율 평균 (첫 fold 학습 행)
결과: outputs/work/f9_*.csv 와 화면 요약.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src_tools"))
from features_ours import add_daily_v1  # noqa: E402
from features_v3 import add_hour_neighbors  # noqa: E402
from oibc_common import WORK, load_sites, load_smp  # noqa: E402
from train_site_model import COMMON, PARAMS, WA_COLS, WB_COLS, load_frame  # noqa: E402

FOLDS = [("2024-25", None), ("2023-24", "2023-06-01"), ("2022-23", "2022-06-01")]
FEATS_RAW = [f"{c}_r{r}" for r in (10, 17) for c in WA_COLS + WB_COLS]
FEATS_COMMON = [c for c in COMMON if c not in ("doy_sin", "doy_cos")]
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


def groups(feats: list[str]) -> dict[str, list[str]]:
    g = {c: [f"{c}_r10", f"{c}_r17"] for c in WA_COLS + WB_COLS}
    g.update({c: [c] for c in FEATS_COMMON})
    for c in feats:
        if c.startswith("wx_day") or c.startswith("wx_prev") or c.startswith("wx_am"):
            g[c] = [c]
    g["앞뒤시각_A운량"] = [c for c in feats if c.startswith("wx_wa_cloud_r17_")]
    g["앞뒤시각_A습도"] = [c for c in feats if c.startswith("wx_wa_humidity_r17_")]
    return g


def build(cut: str | None) -> tuple[pd.DataFrame, list[str]]:
    df = load_frame("valid", 2, cut)
    df["_act"] = df.energy
    df.loc[~df.is_fit, ["energy", "cf"]] = np.nan
    df["is_dead"] = df.is_dead.eq(True) & df.is_fit
    before = set(df.columns)
    df = add_hour_neighbors(add_daily_v1(df))
    added = [c for c in df.columns if c not in before]
    return df, FEATS_RAW + FEATS_COMMON + added


def penalty_fn(pr: pd.DataFrame, day: np.ndarray):
    comp = load_sites().set_index("site_id").current_group.map({"A": 0, "B": 1})
    g = pr.site_id.map(comp).to_numpy().astype(int)
    tcode, tu = pd.factorize(pr.ts)
    key = tcode * 2 + g
    n = 2 * len(tu)
    A = np.bincount(key, weights=pr._act.fillna(0).to_numpy(), minlength=n)
    w = pd.Series(tu).map(load_smp("train").set_index("ts").w).fillna(0).to_numpy()
    W = np.repeat(w, 2)
    cap = pr.capacity_kw.to_numpy()

    def pen(pred_day: np.ndarray) -> float:
        p = np.zeros(len(pr))
        p[day] = np.clip(pred_day, 0, None) * cap[day]
        P = np.bincount(key, weights=p, minlength=n)
        return float(0.1 * np.sum(W * np.abs(P - A)))
    return pen


def main() -> None:
    perm_rows, shap_rows, gain_rows, corr_rows = [], [], [], []
    for fy, cut in FOLDS:
        df, feats = build(cut)
        tr = df[df.is_fit & (df.elev > 0)]
        pr = df[~df.is_fit].reset_index(drop=True)
        day = (pr.elev > 0).to_numpy()
        y = (tr.energy / tr.capacity_kw).clip(0, 1.2).astype("float32")
        m = lgb.LGBMRegressor(**{**PARAMS, "objective": "regression_l1", "random_state": 0})
        m.fit(tr[feats].astype("float32"), y)
        X = pr.loc[day, feats].astype("float32").reset_index(drop=True)
        pen = penalty_fn(pr, day)
        base = pen(m.predict(X))
        log(f"{fy}: 학습 {len(tr):,}행, 검증 {int(day.sum()):,}행, 벌금 {base:,.0f}원")

        # ① 순열 중요도 (벌금)
        rng = np.random.default_rng(0)
        for gname, cols in groups(feats).items():
            idx = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:
                Xp[c] = X[c].to_numpy()[idx]
            perm_rows.append({"fold": fy, "묶음": gname, "벌금증가%": (pen(m.predict(Xp)) / base - 1) * 100})
        log(f"{fy}: 순열 중요도 완료")

        # ② SHAP
        s = X.sample(min(20_000, len(X)), random_state=0)          # TreeSHAP 은 행 × 나무 × 잎 × 깊이² 라 2만 행으로
        sv = m.booster_.predict(s, pred_contrib=True)[:, :-1]
        for i, c in enumerate(feats):
            d = pd.Series(sv[:, i]).corr(s[c].reset_index(drop=True), method="spearman")
            shap_rows.append({"fold": fy, "피처": c, "평균|SHAP|": float(np.abs(sv[:, i]).mean()), "방향ρ": d})
        # ③ gain
        gn = m.booster_.feature_importance("gain")
        for c, v in zip(feats, gn / gn.sum() * 100):
            gain_rows.append({"fold": fy, "피처": c, "gain%": v})
        # ④ 한낮 발전율과의 순위 상관 (학습 1년차 / 2년차 / 검증 연도)
        mid_tr = tr[tr.hour.between(11, 15)]
        half = mid_tr.ts.min() + pd.DateOffset(years=1)
        val = pr[day & pr.hour.between(11, 15).to_numpy()]
        periods = {"학습1년차": mid_tr[mid_tr.ts < half], "학습2년차": mid_tr[mid_tr.ts >= half], "검증연도": val}
        for pname, p in periods.items():
            p = p.sample(min(150_000, len(p)), random_state=0)
            tgt = (p._act / p.capacity_kw) if pname == "검증연도" else (p.energy / p.capacity_kw)
            for c in feats:
                corr_rows.append({"fold": fy, "기간": pname, "피처": c,
                                  "ρ": pd.Series(p[c].to_numpy()).corr(pd.Series(tgt.to_numpy()), method="spearman")})
        log(f"{fy}: SHAP·gain·상관 완료")

        if cut is None:   # ⑤ 중복, ⑥ 구간별 관계는 첫 fold 학습 행으로
            smp = tr[feats].sample(200_000, random_state=0)
            cm = smp.rank().corr()
            pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if abs(cm.loc[a, b]) >= 0.95]
            pd.DataFrame(pairs, columns=["피처1", "피처2", "ρ"]).to_csv(WORK / "f9_redundant.csv", index=False)
            first = (mid_tr, feats)
        del df, tr, pr, X

    perm = pd.DataFrame(perm_rows).pivot(index="묶음", columns="fold", values="벌금증가%")[[f for f, _ in FOLDS]]
    perm["평균"] = perm.mean(axis=1)
    perm["최소"] = perm[[f for f, _ in FOLDS]].min(axis=1)
    perm = perm.sort_values("평균", ascending=False)
    shap = pd.DataFrame(shap_rows).groupby("피처").agg(평균SHAP=("평균|SHAP|", "mean"), 방향=("방향ρ", "mean"))
    gain = pd.DataFrame(gain_rows).groupby("피처")["gain%"].mean()
    corr = pd.DataFrame(corr_rows).pivot_table(index="피처", columns=["fold", "기간"], values="ρ")
    perm.to_csv(WORK / "f9_perm.csv")
    shap.join(gain).to_csv(WORK / "f9_shap_gain.csv")
    corr.to_csv(WORK / "f9_corr.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print("\n① 벌금 기준 순열 중요도 (검증 연도에서 묶음을 섞었을 때 벌금 증가 %) — 평균 순")
    print(perm.round(2).to_string())
    print("\n②③ SHAP 평균 |기여|(발전율 단위, 세 fold 평균)·방향(+면 값이 클수록 발전율↑), gain 비율 — SHAP 순")
    print(shap.join(gain).sort_values("평균SHAP", ascending=False).round(4).to_string())
    sign = np.sign(corr)
    flips = corr[(sign.nunique(axis=1) > 1) & (corr.abs().max(axis=1) > 0.05)]
    print("\n④ 한낮 발전율과의 순위 상관 부호가 기간에 따라 바뀌는 피처 (|ρ| 최대 > 0.05 인 것만)")
    print(flips.round(3).to_string() if len(flips) else "  없음")
    print("\n④-요약 한낮 발전율과의 순위 상관 (9개 기간의 최소 / 평균 / 최대)")
    print(pd.DataFrame({"최소": corr.min(axis=1), "평균": corr.mean(axis=1), "최대": corr.max(axis=1)})
          .sort_values("평균").round(3).to_string())
    red = pd.read_csv(WORK / "f9_redundant.csv")
    print(f"\n⑤ 순위 상관 |ρ| ≥ 0.95 인 피처 쌍 {len(red)}개")
    print(red.round(3).to_string(index=False))
    mid, feats = first
    top = [c for c in perm.index[:8]]
    print("\n⑥ 상위 묶음 대표 피처의 10분위별 한낮 발전율 평균 (첫 fold 학습 행)")
    g = groups(feats)
    for gname in top:
        c = g[gname][-1]
        q = pd.qcut(mid[c], 10, duplicates="drop")
        r = (mid.energy / mid.capacity_kw).groupby(q, observed=True).mean()
        print(f"  {c:28s} " + " ".join(f"{v:.2f}" for v in r.to_numpy()))
    log("끝")


if __name__ == "__main__":
    main()
