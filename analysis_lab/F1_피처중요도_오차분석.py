"""F1 피처 중요도 + 오차 분석 (기준 모델 기준, 가상 평가연도 2024-06 ~ 2025-05).

    python analysis_lab/F1_피처중요도_오차분석.py              # 둘 다
    python analysis_lab/F1_피처중요도_오차분석.py --only error  # 오차 분석만 (학습 없음, 몇 초)

피처 중요도: 기준 모델 설정으로 시드 1개 모델을 학습한 뒤, 예측 행에서 피처(또는 피처 묶음)를 섞었을 때
    대회 점수식 벌금이 몇 % 늘어나는지 본다. 늘어나는 폭이 클수록 중요하고, 0 근처나 음수면 쓸모없거나 해롭다.
오차 분석: outputs/preds/<기준 모델>.parquet 의 예측으로 월·시간대별 과대/과소, 예보 운량 구간별 벌금,
    벌금이 큰 날을 본다.
결과는 outputs/eda/F1_*.csv 에도 저장한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_tools"))
import lightgbm as lgb  # noqa: E402

import train_site_model as tsm  # noqa: E402
from evaluate import evaluate_preds, load_actual  # noqa: E402
from oibc_common import OUTPUTS, PREDS, load_sites, load_smp  # noqa: E402

OUT = OUTPUTS / "eda"


def champion() -> tuple[str, dict]:
    return tsm.load_base_config("champion")


def importance(name: str, cfg: dict) -> pd.DataFrame:
    df = tsm.load_frame("valid", cfg["years"])
    df.loc[~df.is_fit, ["energy", "cf"]] = np.nan
    df["is_dead"] = df.is_dead.eq(True) & df.is_fit
    df, added = tsm.apply_hooks(df, cfg["add"])
    tr = df[df.is_fit & (df.elev > 0)]
    if cfg["drop_dead"]:
        tr = tr[~tr.is_dead]
    pr = df[~df.is_fit].reset_index(drop=True)
    sets = tsm.model_feature_sets(cfg["rounds"])
    if len(sets) != 1:
        raise SystemExit("rounds=avg(모델 2개)는 이 분석에서 지원하지 않음")
    feats = sets[0] + tsm.COMMON + added
    params = {**tsm.PARAMS, **cfg["params"], "objective": tsm.OBJECTIVE[cfg["objective"]], "random_state": 0}
    y = (tr.energy / tr.capacity_kw).clip(0, 1.2)
    model = lgb.LGBMRegressor(**params).fit(tr[feats].astype("float32"), y)
    tsm.log(f"모델 학습 완료 ({len(feats)}개 피처)")

    actual = load_actual()
    X = pr[feats].astype("float32")
    day = pr.elev.to_numpy() > 0
    cap = pr.capacity_kw.to_numpy()

    def penalty(Xp):
        p = np.where(day, np.clip(model.predict(Xp), 0, None), 0) * cap
        return evaluate_preds(pr[["site_id", "ts"]].assign(pred_kwh=p), actual=actual)["penalty"]

    base = penalty(X)
    tsm.log(f"기준 벌금(시드 1개) {base:,.0f}원")
    rng = np.random.default_rng(0)
    families = {
        "[묶음] 10시 발행 예보 전체": [c for c in feats if c.endswith("_r10")],
        "[묶음] 17시 발행 예보 전체": [c for c in feats if c.endswith("_r17")],
        "[묶음] 하루 요약 wx_*": [c for c in feats if c.startswith("wx_")],
        "[묶음] 발전소 위치·용량": ["lat", "lon", "capacity_kw"],
        "[묶음] 태양·시간": ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour"],
    }
    rows = []
    for label, cols in [(c, [c]) for c in feats] + list(families.items()):
        Xp = X.copy()
        idx = rng.permutation(len(Xp))
        Xp[cols] = Xp[cols].to_numpy()[idx]
        d = (penalty(Xp) / base - 1) * 100
        rows.append({"피처": label, "섞었을 때 벌금 변화 %": d})
        tsm.log(f"{label:28s} {d:+7.2f}%")
    gain = pd.Series(model.booster_.feature_importance("gain"), index=feats)
    res = pd.DataFrame(rows)
    res["gain 비중 %"] = res["피처"].map(gain / gain.sum() * 100)
    return res.sort_values("섞었을 때 벌금 변화 %", ascending=False)


def error_analysis(name: str) -> None:
    p = pd.read_parquet(PREDS / f"{name}.parquet")
    p["ts"] = pd.to_datetime(p.ts).dt.tz_convert("Asia/Seoul").dt.as_unit("ns")
    p["site_id"] = p.site_id.astype(str)
    cols = ["site_id", "ts", "date", "month", "hour", "elev", "energy", "capacity_kw", "wa_cloud_r17", "split"]
    b = pd.read_parquet(tsm.WORK / "base.parquet", columns=cols)
    b = b[b.split == "valid"].drop(columns="split")
    b["site_id"] = b.site_id.astype(str)
    m = b.merge(p, on=["site_id", "ts"])
    comp = load_sites().set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    m["group"] = m.site_id.map(comp)
    smp = load_smp("train")[["ts", "w"]]
    g = m.groupby(["group", "ts"]).agg(a=("energy", "sum"), f=("pred_kwh", "sum"), cap=("capacity_kw", "sum"),
                                       cloud=("wa_cloud_r17", "mean"), month=("month", "first"),
                                       hour=("hour", "first"), date=("date", "first")).reset_index()
    g = g.merge(smp, on="ts")
    g["pen"] = 0.1 * g.w * (g.f - g.a).abs()
    g["err"] = g.f - g.a
    tot = g.pen.sum()

    print("\n=== 월별: 벌금 비중과 치우침(예측 − 실적, 11~15시) ===")
    pk = g[g.hour.between(11, 15)]
    mo = g.groupby("month").pen.sum().div(tot).mul(100).to_frame("벌금 %")
    mo["치우침 %"] = pk.groupby("month").err.sum() / pk.groupby("month").a.sum() * 100
    mo["과대예측 시각 %"] = pk.groupby("month").err.apply(lambda e: (e > 0).mean() * 100)
    print(mo.round(1).to_string())

    print("\n=== 시간대별 치우침 (예측 − 실적) / 실적 ===")
    hr = g[g.a > 0].groupby("hour").apply(lambda x: pd.Series({"벌금 %": x.pen.sum() / tot * 100,
                                                              "치우침 %": x.err.sum() / x.a.sum() * 100}))
    print(hr[hr["벌금 %"] > 0.5].round(1).to_string())

    print("\n=== 예보 운량(17시, 묶음 평균) 구간별 — 11~15시 ===")
    pk = pk.assign(bin=pd.cut(pk.cloud, [0, 20, 40, 60, 80, 90, 101], right=False))
    cb = pk.groupby("bin", observed=True).agg(n=("pen", "size"), pen=("pen", "sum"), err=("err", "sum"), a=("a", "sum"))
    cb["벌금 %"] = cb.pen / tot * 100
    cb["치우침 %"] = cb.err / cb.a * 100
    print(cb[["n", "벌금 %", "치우침 %"]].round(1).to_string())

    print("\n=== 벌금이 가장 큰 날 15개 ===")
    d = g.groupby("date").agg(pen=("pen", "sum"), a=("a", "sum"), f=("f", "sum"), cloud=("cloud", "mean"))
    d["벌금 %"] = d.pen / tot * 100
    d["예측/실적"] = d.f / d.a
    top = d.sort_values("pen", ascending=False).head(15)
    print(top[["벌금 %", "예측/실적", "cloud"]].round(2).to_string())
    print(f"\n상위 15일(전체의 4%)이 벌금의 {top['벌금 %'].sum():.1f}%")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "F1_일별벌금.csv", encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["importance", "error"], default=None)
    args = ap.parse_args()
    name, cfg = champion()
    print(f"기준 모델 {name}: {cfg}")
    if args.only != "importance":
        error_analysis(name)
    if args.only != "error":
        res = importance(name, cfg)
        OUT.mkdir(parents=True, exist_ok=True)
        res.to_csv(OUT / "F1_피처중요도.csv", index=False, encoding="utf-8-sig")
        print("\n=== 피처 중요도 (섞었을 때 벌금 변화, 큰 순) ===")
        print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
