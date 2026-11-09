#!/usr/bin/env bash
# 외부 대회 조사(reference/외부_대회_조사.md)에서 나온 후보 — run_gfs_ext.sh 뒤에 실행, 끝난 것은 건너뜀
#   G6  : E10-00x2 + GFS 0.5° 칸 공간 통계(최대·최소·25/75% 분위수·표준편차)   ← HEFTCom2024 4위 GEB
#   S2  : GFS 만 쓰는 모델(지역 예보 A·B 와 그 파생 피처 모두 뺌)                ← HEFTCom2024 1위 SVK(예보 출처별 모델)
#   앙상블: E10-08(3모델) 에 S2 / S2 + V07x2(지역 예보만 쓰는 모델) 를 더한 같은 비중 → 세 해 비교
#   bash analysis_lab/run_more.sh > outputs/work/run_more.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
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

until grep -a -q "GFS 확장 끝" outputs/work/run_gfs_ext.log 2>/dev/null; do sleep 30; done

run G6x2 E10-00x2 --add features_gfs:add_gfs_stats
$PY src_tools/compare_years.py E10-00x2 G6x2 | tail -1

LOCAL=(wa_cloud wa_temp wa_humidity wa_rain wa_wind_speed wa_ground_press wb_cloud wb_temp wb_rel_hum wb_uv_idx
       wb_precip_prob wb_total_liq wb_vis wb_wind_speed wb_dew_point)
D=(); for v in "${LOCAL[@]}"; do D+=(--drop "$v"); done
run S2x2 E9-00x2 --reset-add --add features_gfs:add_gfs_regions "${D[@]}"
$PY src_tools/compare_years.py E10-00x2 S2x2 V07x2 | tail -2

log "앙상블"
for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do
  $PY src_tools/blend.py --out E10-09x2 E10-12x2 E10-11x2 E10-00x2 S2x2 $c > /dev/null 2>&1
  $PY src_tools/blend.py --out E10-10x2 E10-12x2 E10-11x2 E10-00x2 S2x2 V07x2 $c > /dev/null 2>&1
done
$PY src_tools/compare_years.py E10-08x2 E10-09x2 E10-10x2
log "추가 끝"
