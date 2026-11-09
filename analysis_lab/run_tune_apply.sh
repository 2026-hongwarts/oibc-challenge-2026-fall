#!/usr/bin/env bash
# 통과한 LightGBM 설정(시도 26, T1)을 E11-10 의 LightGBM 구성원에 적용 → 섞기 → 구성 → 마지막 제출(E10-08) 대비.
#   E11-10T1ax2: E11-12 + E11-11 + G5bT1 + S2   + V07    (G5b 만 새 설정)
#   E11-10T1x2 : E11-12 + E11-11 + G5bT1 + S2T1 + V07T1  (LightGBM 셋 모두 새 설정)
# 끝난 학습은 건너뜀 → 컴퓨터가 꺼져도 다시 실행하면 이어서.
#   bash analysis_lab/run_tune_apply.sh > outputs/work/run_tune_apply.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
P=$(cat outputs/work/tune_lgb_best.json)
run() {
  local name=$1 base=$2
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base "$base" --params "$P" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
}
blend() {
  local out=$1; shift
  for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
    $PY src_tools/blend.py --out "$out" "$@" $c > /dev/null 2>&1 || log "  !! 섞기 실패 $out $c"
  done
}
blend E11-10T1ax2 E11-12x2 E11-11x2 G5bT1x2 S2x2 V07x2
$PY src_tools/compare_years.py E11-10x2 E11-10T1ax2 | tail -1
log "T1a 섞기 끝"
run S2T1x2 S2x2
run V07T1x2 V07x2
$PY src_tools/compare_years.py S2x2 S2T1x2 | tail -1
$PY src_tools/compare_years.py V07x2 V07T1x2 | tail -1
blend E11-10T1x2 E11-12x2 E11-11x2 G5bT1x2 S2T1x2 V07T1x2
$PY src_tools/compare_years.py E11-10x2 E11-10T1ax2 E11-10T1x2 | tail -2
for n in E11-10T1ax2 E11-10T1x2; do
  [ -f "outputs/compositions/${n}.csv" ] || $PY src_tools/optimize_composition.py --preds "$n" 2>&1 | grep -a -E "홀수|짝수|다른 해|최종|저장"
done
$PY src_tools/compare_own_comp.py E10-08x2 E11-10x2 E11-10T1ax2 E11-10T1x2
log "적용 끝"
