#!/usr/bin/env bash
# GEFS 전용 XGBoost (S3 의 XGB 판, GPU) — 출처별 전용 모델 확장(11/10 에 통한 유일한 방향).
# 미리 정한 판정: E13-10x2 = E12-10c 다섯 모델 + S3Xg (여섯 같은 비중) 를 E12-10cx2 와 같은 참조 구성(sites.csv)으로 비교(compare_years) → 세 해 모두 개선이면 통과.
#   (참고로 S3Xg 단독도 S3x2 와 비교만 해 둔다)
#   bash analysis_lab/run_s3x_gpu.sh >> outputs/work/run_s3x_gpu.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/S3Xgx2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "S3Xgx2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp S3Xgx2 --base S3x2 --model xgb --device cuda ${cut:+--cut $cut} \
    > "outputs/work/xbatch/S3Xgx2_${cut:0:4}.log" 2>&1 || log "  !! 실패 $cut"
done
$PY src_tools/compare_years.py S3x2 S3Xgx2 | tail -1
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E13-10x2 E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2 S3Xgx2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E12-10cx2 E13-10x2 | tail -1
log "S3X 끝"
