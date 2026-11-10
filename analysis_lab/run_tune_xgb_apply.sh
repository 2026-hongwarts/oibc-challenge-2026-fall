#!/usr/bin/env bash
# XGBoost 탐색(run_tune.sh, E11-11x2 20회)이 끝나면: 1위 설정(T2) 고르기 → XGBoost 두 구성원에 적용 → 섞기 → 구성 → 비교 → 시드 2·3 앙상블 재현.
# 고르는 규칙(미리 정함, LightGBM 과 같음): 세 해 모두 개선된 시도 중 가장 나쁜 해가 가장 덜 나쁜 것(maximin).
# 판정(미리 정함): E11-10T2 가 E11-10T1 대비 세 해 모두 개선 + XGBoost 시드 2·3 앙상블에서도 세 해 모두 개선이면 채택.
# 끝난 학습은 건너뜀 → 다시 실행하면 이어서.
#   bash analysis_lab/run_tune_xgb_apply.sh >> outputs/work/run_tune_xgb_apply.log 2>&1
#   E11-10T2x2 : E11-12T2 + E11-11T2 + G5bT1 + S2T1 + V07T1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
until grep -aq "탐색 끝" outputs/work/tune_lgb.log; do sleep 60; done
if [ ! -f outputs/work/tune_xgb_best.json ]; then
  $PY - <<'EOF'
import json
import pandas as pd
t = pd.read_csv("outputs/work/tune_E11-11x2.csv")
t = t[(t.trial > 0) & t.all3.astype(bool)].copy()
t["worst"] = t[["pen_0", "pen_1", "pen_2"]].div(pd.read_csv("outputs/work/tune_E11-11x2.csv").query("trial == 0")[["pen_0", "pen_1", "pen_2"]].iloc[0]).max(axis=1)
b = t.sort_values("worst").iloc[0]
print(f"XGBoost 고른 시도 {int(b.trial)}: 가장 나쁜 해 {(b.worst - 1) * 100:+.2f}%, 평균 {(b.score - 1) * 100:+.2f}%  {b.params}")
open("outputs/work/tune_xgb_best.json", "w", encoding="utf-8").write(b.params)
EOF
fi
P=$(cat outputs/work/tune_xgb_best.json)
log "XGBoost 새 설정 $P"
run() {
  local name=$1 base=$2; shift 2
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base "$base" "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
}
blend() {
  local out=$1; shift
  for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
    $PY src_tools/blend.py --out "$out" "$@" $c > /dev/null 2>&1 || log "  !! 섞기 실패 $out $c"
  done
}
# 1) 시드 0·1
run E11-11T2x2 E11-11x2 --params "$P"
run E11-12T2x2 E11-12x2 --params "$P"
$PY src_tools/compare_years.py E11-11x2 E11-11T2x2 | tail -1
$PY src_tools/compare_years.py E11-12x2 E11-12T2x2 | tail -1
blend E11-10T2x2 E11-12T2x2 E11-11T2x2 G5bT1x2 S2T1x2 V07T1x2
$PY src_tools/compare_years.py E11-10T1x2 E11-10T2x2 | tail -1
[ -f outputs/compositions/E11-10T2x2.csv ] || $PY src_tools/optimize_composition.py --preds E11-10T2x2 2>&1 | grep -a -E "홀수|짝수|다른 해|최종|저장"
$PY src_tools/compare_own_comp.py E10-08x2 E11-10x2 E11-10T1x2 E11-10T2x2
log "XGBoost 적용 끝"
# 2) 앙상블 단위 재현: XGBoost 둘만 시드 2·3 (LightGBM 셋은 T1 시드 0·1 그대로)
run E11-11s2x2 E11-11x2 --seed 2
run E11-11T2s2x2 E11-11x2 --seed 2 --params "$P"
run E11-12s2x2 E11-12x2 --seed 2
run E11-12T2s2x2 E11-12x2 --seed 2 --params "$P"
$PY src_tools/compare_years.py E11-11s2x2 E11-11T2s2x2 | tail -1
$PY src_tools/compare_years.py E11-12s2x2 E11-12T2s2x2 | tail -1
blend E11-10T1xs2x2 E11-12s2x2 E11-11s2x2 G5bT1x2 S2T1x2 V07T1x2
blend E11-10T2s2x2 E11-12T2s2x2 E11-11T2s2x2 G5bT1x2 S2T1x2 V07T1x2
$PY src_tools/compare_years.py E11-10T1xs2x2 E11-10T2s2x2 | tail -1
log "XGBoost 재현 끝"
