#!/usr/bin/env bash
# 파이프라인 점검 4번: 부분 정지·저하 기간(발전소×달 평소 대비 < 0.85)을 학습에서 빼기 (src_tools/features_clean.py)
# 미리 정한 판정: G9cx2 = G5bT1x2 + 정리 vs G5bT1x2, 세 해 모두 개선 → 통과하면 다른 LightGBM 구성원(S3, V07T1)에도 적용해 앙상블로.
#   bash analysis_lab/run_clean.sh >> outputs/work/run_clean.log 2>&1   (끝난 것은 건너뜀)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
CLEAN=(--add features_clean:add_low_month --drop low_month --fit-where "low_month == 0")
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
run G9cx2 G5bT1x2 "${CLEAN[@]}"
grep -a "add_low_month\|fit-where" outputs/work/xbatch/G9cx2_.log | head -2
R=$($PY src_tools/compare_years.py G5bT1x2 G9cx2 | tail -1); echo "$R"
if ! echo "$R" | grep -q "통과"; then log "정리 G5bT1 불통과 → 끝"; log "정리 끝"; exit 0; fi
run S3cx2 S3x2 "${CLEAN[@]}"
run V07cx2 V07T1x2 "${CLEAN[@]}"
$PY src_tools/compare_years.py S3x2 S3cx2 | tail -1
$PY src_tools/compare_years.py V07T1x2 V07cx2 | tail -1
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E14Cx2 E11-12x2 E11-11x2 G9cx2 S3cx2 V07cx2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E12-10cx2 E14Cx2 | tail -1
log "정리 끝"
