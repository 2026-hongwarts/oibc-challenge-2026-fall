#!/usr/bin/env bash
# SMP 가중 손실 (설명회 슬 27 "SMP · k로 가중한 손실", 11/10 사용자 승인: 학습 기간 SMP 만 → 규정 §8 지킴)
# 미리 정한 판정: 같은 장치 기준 대비 세 해 모두 개선이면 통과.
#   GPU: E13-11wgx2 = E11-11gx2 + --weight smp_k   (기준 E11-11gx2)
#   CPU: G8wx2      = G5bT1x2  + --weight smp_k    (기준 G5bT1x2, S3 재현이 끝난 뒤)
#   bash analysis_lab/run_smpw.sh >> outputs/work/run_smpw.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
run() {
  local name=$1 base=$2; shift 2
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base "$base" "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
  $PY src_tools/compare_years.py "$base" "$name" | tail -1
}
run E13-11wgx2 E11-11gx2 --weight smp_k
log "SMP 가중 XGB(GPU) 끝"
until grep -aq "S3 재현 끝" outputs/work/run_s3_repro.log 2>/dev/null; do sleep 60; done
run G8wx2 G5bT1x2 --weight smp_k
log "SMP 가중 끝"
