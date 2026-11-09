#!/usr/bin/env bash
# LightGBM 탐색 1위 설정(시도 26, maximin 기준) 재현 확인 — 시드 0·1 과 2·3, 세 해. 끝난 것은 건너뜀.
#   G5bT1x2   vs G5bx2    (같은 시드 0·1)
#   G5bT1s2x2 vs G5bs2x2  (같은 시드 2·3)
#   bash analysis_lab/run_tune_check.sh > outputs/work/run_tune_check.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
P=$(cat outputs/work/tune_lgb_best.json)
run() {
  local name=$1; shift
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base G5bx2 --params "$P" "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
}
run G5bT1x2
run G5bT1s2x2 --seed 2
$PY src_tools/compare_years.py G5bx2 G5bT1x2 | tail -1
$PY src_tools/compare_years.py G5bs2x2 G5bT1s2x2 | tail -1
log "확인 끝"
