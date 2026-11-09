#!/usr/bin/env bash
# 앙상블 단위 재현: LightGBM 구성원만 시드 2·3 으로 바꿔서(XGBoost 둘은 그대로) 새 설정의 효과가 다시 나오는지.
#   E11-10s2x2   : E11-12 + E11-11 + G5bs2   + S2s2   + V07s2     (옛 설정)
#   E11-10T1s2x2 : E11-12 + E11-11 + G5bT1s2 + S2T1s2 + V07T1s2   (새 설정)
#   E11-10T1bs2x2: E11-12 + E11-11 + G5bT1s2 + S2T1s2 + V07s2     (V07 만 옛 설정)
# 끝난 학습은 건너뜀.   bash analysis_lab/run_tune_apply3.sh > outputs/work/run_tune_apply3.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
P=$(cat outputs/work/tune_lgb_best.json)
run() {
  local name=$1; shift
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base V07x2 --seed 2 "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
}
run V07s2x2
run V07T1s2x2 --params "$P"
$PY src_tools/compare_years.py V07s2x2 V07T1s2x2 | tail -1
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E11-10s2x2 E11-12x2 E11-11x2 G5bs2x2 S2s2x2 V07s2x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패"
  $PY src_tools/blend.py --out E11-10T1s2x2 E11-12x2 E11-11x2 G5bT1s2x2 S2T1s2x2 V07T1s2x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패"
  $PY src_tools/blend.py --out E11-10T1bs2x2 E11-12x2 E11-11x2 G5bT1s2x2 S2T1s2x2 V07s2x2 $c > /dev/null 2>&1 || log "  !! 섞기 실패"
done
$PY src_tools/compare_years.py E11-10s2x2 E11-10T1s2x2 E11-10T1bs2x2 | tail -2
log "앙상블 재현 끝"
