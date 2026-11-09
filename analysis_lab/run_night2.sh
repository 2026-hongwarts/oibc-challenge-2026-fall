#!/usr/bin/env bash
# E10-10(출처별 모델 추가) 통과 후 — run_night.sh 의 "앙상블 끝" 뒤에 실행 (설정값 탐색과 동시에 돌아도 됨, 가벼움)
#  S2bx2: GFS 만(일사량·운량 지역 + 구름 높이별 운량 지역) 쓰는 LightGBM
#  E11-10x2 : E11-12 + E11-11 + G5bx2 + S2x2  + V07x2  같은 비중
#  E11-10bx2: E11-12 + E11-11 + G5bx2 + S2bx2 + V07x2  같은 비중
#   bash analysis_lab/run_night2.sh > outputs/work/run_night2.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
LOCAL=(wa_cloud wa_temp wa_humidity wa_rain wa_wind_speed wa_ground_press wb_cloud wb_temp wb_rel_hum wb_uv_idx
       wb_precip_prob wb_total_liq wb_vis wb_wind_speed wb_dew_point)
D=(); for v in "${LOCAL[@]}"; do D+=(--drop "$v"); done

until grep -a -q "앙상블 끝" outputs/work/run_night.log 2>/dev/null; do sleep 30; done

for cut in "" 2023-06-01 2022-06-01; do
  f="outputs/preds/S2bx2${cut:+_cut${cut:0:4}}.parquet"; [ -f "$f" ] && continue
  log "S2bx2 ${cut:-2024-25}"
  $PY src_tools/train_site_model.py --exp S2bx2 --base E9-00x2 --reset-add --add features_gfs:add_gfs_regions \
    --add features_gfs:add_gfs_layers_regions "${D[@]}" ${cut:+--cut $cut} > "outputs/work/xbatch/S2bx2_${cut:0:4}.log" 2>&1
done
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E11-10x2 E11-12x2 E11-11x2 G5bx2 S2x2 V07x2 $c > /dev/null 2>&1
  $PY src_tools/blend.py --out E11-10bx2 E11-12x2 E11-11x2 G5bx2 S2bx2 V07x2 $c > /dev/null 2>&1
done
$PY src_tools/compare_years.py E10-08x2 E10-10x2 E11-08x2 E11-10x2 E11-10bx2
log "출처별 앙상블 끝"
