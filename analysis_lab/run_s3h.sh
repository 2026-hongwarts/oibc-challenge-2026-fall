#!/usr/bin/env bash
# 파이프라인 점검 5번: GEFS 표준편차 3시간 환산(features_gfs:add_gefs3h_regions) → S3 다시.
# 미리 정한 판정: S3hx2 vs S3x2 세 해 모두 개선 → 통과하면 E14Hx2 = E12-10c 의 S3 자리를 S3h 로 vs E12-10cx2 세 해 모두 개선
#   → 통과하면 평가 기간 학습 → 빠짐 처리 → 일출·일몰 → 추세 → 제출 파일 E14Hftrx2_comp (구성 E12-10cx2.csv)
#   bash analysis_lab/run_s3h.sh >> outputs/work/run_s3h.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/S3hx2${cut:+_cut${cut:0:4}}.parquet"
  [ -f "$f" ] && continue
  log "S3hx2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp S3hx2 --base S3x2 --reset-add --add features_gfs:add_gefs3h_regions ${cut:+--cut $cut} \
    > "outputs/work/xbatch/S3hx2_${cut:0:4}.log" 2>&1 || log "  !! 실패 $cut"
done
R=$($PY src_tools/compare_years.py S3x2 S3hx2 | tail -1); echo "$R"
if ! echo "$R" | grep -q "통과"; then log "S3h 불통과 → 끝"; log "S3h 끝"; exit 0; fi
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E14Hx2 E11-12x2 E11-11x2 G5bT1x2 S3hx2 V07T1x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
R2=$($PY src_tools/compare_years.py E12-10cx2 E14Hx2 | tail -1); echo "$R2"
if ! echo "$R2" | grep -q "통과"; then log "앙상블 불통과 → 끝"; log "S3h 끝"; exit 0; fi
[ -f outputs/preds_test/S3hx2.parquet ] || $PY src_tools/train_site_model.py --base S3hx2 --mode test > outputs/work/xbatch/S3hx2_test.log 2>&1
$PY src_tools/blend.py --out E14Hfx2 E11-12x2 E11-11x2 G5bT1x2 S3hx2 V07T1x2 --mode test 2>&1 | grep -a "빠짐\|저장"
$PY src_tools/twilight_fix.py --preds E14Hfx2 --out E14Hftx2 2>&1 | grep -a "preds_test"
$PY src_tools/trend_fix.py --preds E14Hftx2 --out E14Hftrx2 2>&1 | grep -a "test"
$PY src_tools/make_submission.py outputs/preds_test/E14Hftrx2.parquet --composition outputs/compositions/E12-10cx2.csv --name E14Hftrx2_comp 2>&1 | tail -2
$PY outputs/work/validate_submission.py --composition outputs/submissions/E14Hftrx2_comp/composition.csv \
  --forecast outputs/submissions/E14Hftrx2_comp/forecast.csv --data 01_participant_dataset | head -1
log "S3h 끝"
