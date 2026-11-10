#!/usr/bin/env bash
# G7 이 2022-23 에서 실패(+0.93%) → 앞서 말한 대비책: GEFS 상자 평균 4개(평균·표준편차 × 일사·운량)만 넣은 작은 판을 GPU 로.
# GEFS 평균은 GFS 와 상관 0.996(거의 같은 정보) → 172개를 넣으면 희석될 수 있다는 가설. 판정: E11-11gx2 대비 세 해 모두 개선.
#   bash analysis_lab/run_gefs_k_gpu.sh >> outputs/work/run_gefs_k_gpu.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/E12-11kgx2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "E12-11kgx2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp E12-11kgx2 --base E11-11gx2 --add features_gfs:add_gefs_kr ${cut:+--cut $cut} \
    > "outputs/work/xbatch/E12-11kgx2_${cut:0:4}.log" 2>&1 || log "  !! 실패 $cut"
done
$PY src_tools/compare_years.py E11-11gx2 E12-11gx2 E12-11kgx2 | tail -2
log "GEFS 작은 판 끝"
