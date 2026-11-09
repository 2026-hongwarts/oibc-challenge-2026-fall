#!/usr/bin/env bash
# GFS 확장 (기준 E10-00x2 = LightGBM 발전율, V07 + G2) — 다시 실행하면 끝난 것은 건너뜀
#   G3  : + 0.5° 칸 일사량·운량 주성분 15개씩 (30개)
#   G5a : + 하층·중층·상층 운량 전체 평균 (3개, 1°)
#   G5b : + 하층·중층·상층 운량 1° 지역 42곳 + 평균 (129개)
#   bash analysis_lab/run_gfs_ext.sh > outputs/work/run_gfs_ext.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
G=${G:-/c/Users/109so/miniconda3/envs/gfs/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
run() {   # run 이름 추가옵션...
  local name=$1; shift
  for cut in "" 2023-06-01 2022-06-01; do
    local f="outputs/preds/${name}${cut:+_cut${cut:0:4}}.parquet"
    [ -f "$f" ] && continue
    log "$name ${cut:-2024-25}"
    $PY src_tools/train_site_model.py --exp "$name" --base E10-00x2 "$@" ${cut:+--cut $cut} \
      > "outputs/work/xbatch/${name}_${cut:0:4}.log" 2>&1 || log "  !! 실패 $name $cut"
  done
  $PY src_tools/compare_years.py E10-00x2 "$name" | tail -1
}

until [ -f outputs/work/gfs_cells.parquet ] && grep -q "칸" outputs/work/gfs_decode_cells.log 2>/dev/null; do sleep 20; done
run G3x2 --add features_gfs:add_gfs_cells_pca

until grep -q "1825/1825일" outputs/work/gfs_fetch_1p00.log 2>/dev/null; do sleep 30; done
if [ ! -f outputs/work/gfs_layers.parquet ]; then
  log "구름 높이별 운량 해석"
  $G -u src_tools/gfs_decode.py --raw outputs/work/gfs_raw_1p00 --vars LCDC MCDC HCDC --out outputs/work/gfs_layers.parquet \
    > outputs/work/gfs_decode_layers.log 2>&1
  tail -1 outputs/work/gfs_decode_layers.log
fi
run G5ax2 --add features_gfs:add_gfs_layers_kr
run G5bx2 --add features_gfs:add_gfs_layers_regions
log "GFS 확장 끝"
