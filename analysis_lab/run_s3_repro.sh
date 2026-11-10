#!/usr/bin/env bash
# S3(GEFS 전용 LightGBM) 시드 재현 — 관문 2(다른 시드 2·3).
#   S3s2x2 vs S2T1s2x2 (같은 시드 2·3), 그리고 앙상블 E12-10cs2x2(S3s2 를 넣은 것) vs E11-10T1 의 S2T1s2 판(E11-10T1s2Sx2)
#   bash analysis_lab/run_s3_repro.sh >> outputs/work/run_s3_repro.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/S3s2x2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "S3s2x2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp S3s2x2 --base S3x2 --seed 2 ${cut:+--cut $cut} \
    > "outputs/work/xbatch/S3s2x2_${cut:0:4}.log" 2>&1 || log "  !! 실패 $cut"
done
$PY src_tools/compare_years.py S2T1s2x2 S3s2x2 | tail -1
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E11-10T1s2Sx2 E11-12x2 E11-11x2 G5bT1x2 S2T1s2x2 V07T1x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
  $PY src_tools/blend.py --out E12-10cs2x2 E11-12x2 E11-11x2 G5bT1x2 S3s2x2 V07T1x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E11-10T1s2Sx2 E12-10cs2x2 | tail -1
log "S3 재현 끝"
