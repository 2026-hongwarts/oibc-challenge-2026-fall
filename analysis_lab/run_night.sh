#!/usr/bin/env bash
# 11/10 밤사이: G5b(구름 높이별 운량 1° 지역) 통과 후 — run_more.sh 뒤에 실행, 끝난 것은 건너뜀
#  1) G5b 재현 (시드 2·3, 비교 E10-00s2x2)
#  2) G5b 를 넣은 XGBoost: E11-11x2 (발전율), E11-12x2 (CSI)
#  3) 세 모델 앙상블 E11-08x2 = E11-12 + E11-11 + G5bx2(LightGBM) 1/3씩 → E10-08x2 와 세 해 비교
#  4) 설정값 탐색: LightGBM (G5bx2 설정) 40회 → XGBoost (E11-11x2 설정) 25회
#   bash analysis_lab/run_night.sh > outputs/work/run_night.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
L=(--add features_gfs:add_gfs_layers_regions)
run() {   # run 이름 base 추가옵션...
  local name=$1 base=$2; shift 2
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base "$base" "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
}

until grep -a -q "추가 끝" outputs/work/run_more.log 2>/dev/null; do sleep 30; done

run G5bs2x2 E10-00x2 "${L[@]}" --seed 2
$PY src_tools/compare_years.py E10-00s2x2 G5bs2x2 | tail -1
run E11-11x2 E10-11x2 "${L[@]}"
run E11-12x2 E10-12x2 "${L[@]}"
$PY src_tools/compare_years.py E10-11x2 E11-11x2 | tail -1
$PY src_tools/compare_years.py E10-12x2 E11-12x2 | tail -1
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E11-08x2 E11-12x2 E11-11x2 G5bx2 $c > /dev/null 2>&1
done
$PY src_tools/compare_years.py E10-08x2 E11-08x2
log "앙상블 끝"

$PY -u src_tools/tune.py --base G5bx2 --trials 40 > outputs/work/tune_lgb.log 2>&1
log "LightGBM 탐색 끝"
$PY -u src_tools/tune.py --base E11-11x2 --trials 25 > outputs/work/tune_xgb.log 2>&1
log "밤 작업 끝"
