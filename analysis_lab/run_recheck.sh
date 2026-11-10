#!/usr/bin/env bash
# 임의 판단 다시 따지기 (docs/임의_판단_점검.md D절)
#   D1: GEFS 전용 XGBoost 를 XGB 탐색 설정(시도 13)으로 — S3Xg 는 기본 설정이라 불공정했음.
#       판정(미리 정함): E13-10Tx2 = E12-10c + S3XTgx2 (여섯 같은 비중) vs E12-10cx2, 세 해 모두 개선.
#   D2: S3 학습 기간 3년(정보용 — 2022-23 은 학습 자료가 같아 규칙으로 판정 불가). CPU 가 빌 때(밤샘 자동 진행 뒤).
#   bash analysis_lab/run_recheck.sh >> outputs/work/run_recheck.log 2>&1   (끝난 것은 건너뜀)
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
until [ -f outputs/preds/E13-11bgx2_cut2022.parquet ] || grep -aq "P2 끝" outputs/work/run_p2.log 2>/dev/null; do sleep 60; done   # GPU 는 하나씩
P=$(cat outputs/work/tune_xgb_best.json)
run S3XTgx2 S3x2 --model xgb --device cuda --params "$P"
$PY src_tools/compare_years.py S3x2 S3XTgx2 | tail -1
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E13-10Tx2 E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2 S3XTgx2 $c > /dev/null 2>&1 || log "  !! 섞기 실패 $c"
done
$PY src_tools/compare_years.py E12-10cx2 E13-10Tx2 | tail -1
log "D1 끝"
while [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'overnight' }).Count")" != "0" ]; do sleep 60; done
run S3Y3x2 S3x2 --years 3
$PY src_tools/compare_years.py S3x2 S3Y3x2 | tail -1
log "D2 끝"
