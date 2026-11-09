#!/usr/bin/env bash
# 3단계 준비 (2단계 run_stage2.sh 뒤에 실행, 이미 있으면 건너뜀)
#  1) E10-12bx2: XGBoost CSI + G2 만 (V07 빼기 없음) — V07 은 LightGBM 중요도로 정해서 XGBoost 에는 해로울 수 있음
#  2) E10 모델들의 실제 평가 기간 예측(2023-06 ~ 2025-05 학습 → 2025-06 ~ 2026-05)
#   bash analysis_lab/run_stage3_test.sh > outputs/work/run_stage3_test.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }

for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/E10-12bx2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "E10-12bx2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp E10-12bx2 --base E9-12x2 --add features_gfs:add_gfs_regions ${cut:+--cut $cut} \
    > "outputs/work/xbatch/E10-12bx2_${cut:0:4}.log" 2>&1 || log "  !! 실패 E10-12bx2 $cut"
done
$PY src_tools/compare_years.py E9-12x2 E10-12x2 E10-12bx2

for name in E10-00x2 E10-12bx2 E10-12x2 E10-02x2 E10-11x2; do
  [ -f "outputs/preds_test/${name}.parquet" ] && continue
  [ -f "outputs/preds/${name}.json" ] || { log "$name 설정 없음, 건너뜀"; continue; }
  log "$name 평가 기간 학습"
  $PY src_tools/train_site_model.py --base "$name" --mode test > "outputs/work/xbatch/${name}_test.log" 2>&1 || log "  !! 실패 $name"
done
log "평가 기간 학습 끝"
