#!/usr/bin/env bash
# 파이프라인 점검 2번: B 운량·자외선 1시간 어긋남 → ts−1 값 덧붙이기(features_v3:add_wb_lag1).
# 미리 정한 판정(세 해 모두 개선):
#   1) GPU 걸러내기: E13-11bgx2 = E11-11gx2 + add_wb_lag1  vs E11-11gx2
#   2) 통과하면 CPU: V07Bx2 = V07T1x2 + add_wb_lag1 vs V07T1x2   (밤샘 자동 진행이 끝난 뒤 — 메모리)
#   3) 앙상블: E14Bx2 = [E11-12x2, E13-11bgx2, G5bT1x2, S3x2, V07Bx2(통과 시) 또는 V07T1x2]
#      vs 같은 장치 기준 E12-10cGx2 = [E11-12x2, E11-11gx2, G5bT1x2, S3x2, V07T1x2]
#   bash analysis_lab/run_p2.sh >> outputs/work/run_p2.log 2>&1   (끝난 것은 건너뜀)
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
until grep -aq "MLP 끝" outputs/work/run_mlp.log 2>/dev/null; do sleep 60; done
run E13-11bgx2 E11-11gx2 --add features_v3:add_wb_lag1
R=$($PY src_tools/compare_years.py E11-11gx2 E13-11bgx2 | tail -1); echo "$R"
if ! echo "$R" | grep -q "통과"; then log "B 어긋남 GPU 걸러내기 불통과 → 끝"; log "P2 끝"; exit 0; fi
while [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'overnight' }).Count")" != "0" ]; do sleep 60; done
run V07Bx2 V07T1x2 --add features_v3:add_wb_lag1
R2=$($PY src_tools/compare_years.py V07T1x2 V07Bx2 | tail -1); echo "$R2"
V=V07T1x2; echo "$R2" | grep -q "통과" && V=V07Bx2
blend E12-10cGx2 E11-12x2 E11-11gx2 G5bT1x2 S3x2 V07T1x2
blend E14Bx2 E11-12x2 E13-11bgx2 G5bT1x2 S3x2 "$V"
$PY src_tools/compare_years.py E12-10cGx2 E14Bx2 | tail -1
log "P2 끝"
