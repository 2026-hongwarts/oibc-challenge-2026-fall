#!/usr/bin/env bash
# GEFS 를 XGBoost 두 모델에 넣기 — GPU(--device cuda)로, CPU 에서 도는 run_gefs.sh(LightGBM)와 동시에.
# 미리 정한 판정:
#   기준은 같은 GPU 로 다시 학습한 E11-11gx2(XGB 발전율)·E11-12gx2(XGB CSI) 세 해.
#   E12-11gx2 = E11-11gx2 + GEFS 1° 지역(172개), E12-12gx2 = E11-12gx2 + GEFS 1° 지역 → 각 기준 대비 세 해 모두 개선이면 통과.
#   E11-10T1gx2 = 현재 최고(E11-10T1)에서 XGB 둘만 GPU 판 → E11-10T1x2 와 비교(GPU 로 바꿔도 앙상블이 같은지 확인).
# 메모리: G7x2 의 2024-25 가 끝난 뒤 시작(그 뒤 LightGBM 학습은 전체 표를 일찍 반납하는 새 코드로 돈다).
# 끝난 것은 건너뜀.   bash analysis_lab/run_gefs_xgb.sh >> outputs/work/run_gefs_xgb.log 2>&1
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
}
blend() {
  local out=$1; shift
  for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
    $PY src_tools/blend.py --out "$out" "$@" $c > /dev/null 2>&1 || log "  !! 섞기 실패 $out $c"
  done
}
until [ -f outputs/preds/G7x2.parquet ]; do sleep 60; done
log "GPU 기준 다시 학습"
run E11-11gx2 E11-11x2 --device cuda
run E11-12gx2 E11-12x2 --device cuda
$PY src_tools/compare_years.py E11-11x2 E11-11gx2 | tail -1
$PY src_tools/compare_years.py E11-12x2 E11-12gx2 | tail -1
log "GEFS 넣은 XGBoost"
run E12-11gx2 E11-11gx2 --add features_gfs:add_gefs_regions
run E12-12gx2 E11-12gx2 --add features_gfs:add_gefs_regions
$PY src_tools/compare_years.py E11-11gx2 E12-11gx2 | tail -1
$PY src_tools/compare_years.py E11-12gx2 E12-12gx2 | tail -1
blend E11-10T1gx2 E11-12gx2 E11-11gx2 G5bT1x2 S2T1x2 V07T1x2
$PY src_tools/compare_years.py E11-10T1x2 E11-10T1gx2 | tail -1
log "GEFS XGBoost 끝"
