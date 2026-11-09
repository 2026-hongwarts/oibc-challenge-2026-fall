#!/usr/bin/env bash
# run_tune_apply.sh 결과 뒤: V07T1 은 2023-24 +1.01% 로 관문 실패 → V07 은 옛 설정 유지한 E11-10T1b,
# 그리고 통과한 S2T1 을 시드 2·3 으로 재현 확인(S2s2x2 vs S2T1s2x2). 끝난 것은 건너뜀.
#   E11-10T1bx2: E11-12 + E11-11 + G5bT1 + S2T1 + V07
#   bash analysis_lab/run_tune_apply2.sh > outputs/work/run_tune_apply2.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
P=$(cat outputs/work/tune_lgb_best.json)
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E11-10T1bx2 E11-12x2 E11-11x2 G5bT1x2 S2T1x2 V07x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E11-10x2 E11-10T1x2 E11-10T1bx2 | tail -2
[ -f outputs/compositions/E11-10T1bx2.csv ] || $PY src_tools/optimize_composition.py --preds E11-10T1bx2 2>&1 | grep -a -E "홀수|짝수|다른 해|최종|저장"
$PY src_tools/compare_own_comp.py E10-08x2 E11-10x2 E11-10T1x2 E11-10T1bx2
log "T1b 끝"
run() {
  local name=$1; shift
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base S2x2 --seed 2 "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
}
run S2s2x2
run S2T1s2x2 --params "$P"
$PY src_tools/compare_years.py S2s2x2 S2T1s2x2 | tail -1
log "S2 재현 끝"
