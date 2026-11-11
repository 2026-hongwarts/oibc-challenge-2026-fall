#!/usr/bin/env bash
# 11/11 낮 순서(메모리 32GB — 학습은 한 번에 하나): 부분 정지 빼기(run_clean) → 에어로졸(GPU) → 기압 자료(CPU) → MLP 두 번째(GPU)
#   bash analysis_lab/run_chain_1111.sh >> outputs/work/run_chain_1111.log 2>&1   (각 단계는 끝난 것을 건너뜀)
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
until grep -aq "정리 끝" outputs/work/run_clean.log 2>/dev/null; do sleep 60; done
log "에어로졸"; bash analysis_lab/run_fe_gpu.sh >> outputs/work/run_fe_gpu.log 2>&1
log "기압 자료"; bash analysis_lab/run_nopress.sh >> outputs/work/run_nopress.log 2>&1
log "MLP 두 번째"; bash analysis_lab/run_mlp2.sh >> outputs/work/run_mlp2.log 2>&1
log "순서 끝"
