"""설정값 탐색(Optuna TPE, 시드 고정) — 세 해(2024-25 / 2023-24 / 2022-23) 대회 벌금 기준.

    python src_tools/tune.py --base E10-00x2 --trials 40          LightGBM (설정은 --base 의 json 에서)
    python src_tools/tune.py --base E10-11x2 --trials 30          XGBoost
    → outputs/work/tune_{base}.csv (시도마다 한 줄, 중간에 끊겨도 남음)

- 세 해의 학습·예측 표(피처 함수 적용 결과)를 한 번만 만들어 두고 설정값만 바꿔 학습한다(시드 1개).
- 점수 = 해마다 (그 시도 벌금 / 기본 설정 벌금) 의 평균. 세 해 모두 1 보다 작은지도 기록.
- 같은 세 해로 고르므로 낙관적이다 → 고른 설정은 다른 시드(2·3)로 train_site_model 에서 다시 확인한다.
- 벌금은 참조 구성의 묶음 × 시각 합계로 계산(학습 기간 SMP × k, 0.1 배).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from oibc_common import WORK, load_sites, load_smp
from train_site_model import (COMMON, apply_hooks, clear_ref, load_base_config, load_frame, make_model,
                              model_feature_sets, to_X)

CUTS = [None, "2023-06-01", "2022-06-01"]
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)


def space(kind: str, t) -> dict:
    """t: optuna.trial.Trial"""
    if kind == "lgb":
        return dict(num_leaves=t.suggest_categorical("num_leaves", [31, 63, 127, 255]),
                    min_child_samples=t.suggest_int("min_child_samples", 50, 1500, log=True),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.1, log=True),
                    n_estimators=t.suggest_categorical("n_estimators", [400, 600, 900, 1300]),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                    subsample=t.suggest_float("subsample", 0.6, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 0.01, 30, log=True))
    return dict(max_depth=t.suggest_int("max_depth", 5, 10),
                min_child_weight=t.suggest_float("min_child_weight", 10, 500, log=True),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.1, log=True),
                n_estimators=t.suggest_categorical("n_estimators", [500, 800, 1200]),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 30, log=True))


def build(cfg: dict, cut: str | None) -> dict:
    df = load_frame("valid", cfg["years"], cut)
    act = df.energy.copy()
    df.loc[~df.is_fit, ["energy", "cf"]] = np.nan
    df["is_dead"] = df.is_dead.eq(True) & df.is_fit
    df, added = apply_hooks(df, cfg["add"])
    tr = df[df.is_fit & (df.elev > 0)]
    if cfg["drop_dead"]:
        tr = tr[~tr.is_dead.astype(bool)]
    pr_all = df[~df.is_fit]
    day = (pr_all.elev > 0).to_numpy()
    cols = model_feature_sets(cfg["rounds"])[0]
    feats = [c for c in cols + COMMON + added
             if not any(c == d or c in (f"{d}_r10", f"{d}_r17") for d in cfg["drop"])]
    cats = cfg.get("cat", [])
    feats += [c for c in cats if c not in feats]
    y = (tr.energy / tr.capacity_kw).clip(0, 1.2).astype("float32").to_numpy()
    w = None
    csi = cfg.get("target", "cf") == "csi"
    if csi:
        ref = clear_ref(tr)
        y = np.clip(y / ref, 0, 3.0).astype("float32")
        w = ref
    comp = load_sites().set_index("site_id").current_group.map({"A": 0, "B": 1})
    g = pr_all.site_id.map(comp).to_numpy().astype(int)
    tcode, tu = pd.factorize(pr_all.ts)
    key = tcode * 2 + g
    A = np.bincount(key, weights=act[~df.is_fit].fillna(0).to_numpy(), minlength=2 * len(tu))
    W = np.repeat(pd.Series(tu).map(load_smp("train").set_index("ts").w).fillna(0).to_numpy(), 2)
    return dict(X_tr=to_X(tr, feats, cats), y=y, w=w, X_pr=to_X(pr_all[day], feats, cats),
                ref_pr=clear_ref(pr_all[day]) if csi else None, cap=pr_all.capacity_kw.to_numpy()[day],
                day=day, key=key, A=A, W=W, n=len(pr_all), nfeat=len(feats))


def penalty(d: dict, model) -> float:
    p = np.clip(model.predict(d["X_pr"]), 0, None)
    if d["ref_pr"] is not None:
        p = np.clip(p * d["ref_pr"], 0, 1.2)
    full = np.zeros(d["n"])
    full[d["day"]] = p * d["cap"]
    P = np.bincount(d["key"], weights=full, minlength=len(d["A"]))
    return float(0.1 * np.sum(d["W"] * np.abs(P - d["A"])))


def main() -> None:
    ap = argparse.ArgumentParser(description="세 해 벌금 기준 설정값 무작위 탐색")
    ap.add_argument("--base", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    name, cfg = load_base_config(args.base)
    kind = cfg["model"]
    out = WORK / f"tune_{name}.csv"
    folds = {}
    for c in CUTS:
        folds[c] = build(cfg, c)
        log(f"{c or '2024-25'} 표 준비 (학습 {len(folds[c]['y']):,}행, 피처 {folds[c]['nfeat']}개)")

    def trial(params: dict) -> list[float]:
        res = []
        for c in CUTS:
            d = folds[c]
            m = make_model(kind, cfg["objective"], params, 0, has_cat=bool(cfg.get("cat"))).fit(d["X_tr"], d["y"], sample_weight=d["w"])
            res.append(penalty(d, m))
        return res

    base = trial(cfg["params"])
    log(f"기본 설정 벌금 {[f'{v:,.0f}' for v in base]}")
    rows = [dict(trial=0, **{f"pen_{i}": v for i, v in enumerate(base)}, score=1.0, all3=False, params=json.dumps(cfg["params"]))]
    if out.exists():                                   # 중간에 끊겼던 실행의 결과표를 이어 씀
        prev = pd.read_csv(out)
        rows = rows + prev[prev.trial > 0].to_dict("records")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(t) -> float:
        p = space(kind, t)
        r = trial(p)
        ratios = [a / b for a, b in zip(r, base)]
        row = dict(trial=t.number + 1, **{f"pen_{i}": v for i, v in enumerate(r)}, score=float(np.mean(ratios)),
                   all3=all(x < 1 for x in ratios), params=json.dumps(p))
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        log(f"시도 {row['trial']}: " + " / ".join(f"{(x - 1) * 100:+.2f}%" for x in ratios)
            + f"  평균 {(row['score'] - 1) * 100:+.2f}%" + ("  세 해 모두 개선" if row["all3"] else "") + f"  {p}")
        return row["score"]

    # 시도마다 SQLite 에 저장 → 컴퓨터가 꺼져도 다시 실행하면 끝난 시도는 이어받고 남은 횟수만 시도
    storage = f"sqlite:///outputs/work/tune_{name}.db"
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed),
                                study_name=name, storage=storage, load_if_exists=True)
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    log(f"저장된 완료 시도 {done}개 → 남은 {max(0, args.trials - done)}개 시도")
    if args.trials > done:
        study.optimize(objective, n_trials=args.trials - done)
    best = pd.DataFrame(rows).sort_values("score").head(5)
    log("상위 5개:\n" + best[["trial", "score", "all3", "params"]].to_string(index=False))


if __name__ == "__main__":
    main()
