#!/usr/bin/env bash
# GEFS(NOAA 앙상블 예보 평균·표준편차) 실험 — 받기 끝 기다림 → 빠진 파일 다시 받기 → 해석 → 세 해 실험 → 앙상블.
# 미리 정한 판정:
#   G7  = G5bT1 + GEFS 1° 지역(172개), G7k = G5bT1 + GEFS 상자 평균(4개) → G5bT1x2 대비 세 해 모두 개선이면 통과.
#         둘 다 통과하면 가장 나쁜 해가 덜 나쁜 것. (채택 전 시드 2·3 재현은 따로)
#   S3  = GEFS 만 쓰는 LightGBM(S2T1 에서 GFS 대신 GEFS) → 앙상블 구성원 후보
#   E12-10  = 제출 후보(E11-10T1 또는 T2)의 G5bT1 을 통과한 G7(k) 로 바꾸고 S3 를 더함(6개 같은 비중)
#   E12-10a = G5bT1 → G7(k) 만,  E12-10b = S3 만 더함
#   → 제출 후보 대비 세 해, 자기 구성으로 E10-08 대비
# 11/10 사용자 결정: G7k 는 건너뜀(CPU 시간 절약, G7 만 판정) → run_gefs2.sh. 실행 중인 G7 학습이 끝나기를 기다렸다가 이어받는다.
# 끝난 것은 건너뜀.   bash analysis_lab/run_gefs2.sh >> outputs/work/run_gefs.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/c/Users/109so/miniconda3/envs/bio_proj/python.exe}
GPY=/c/Users/109so/miniconda3/envs/gfs/python.exe
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%m-%d\ %H:%M)] $*"; }
until grep -aq "1825/1825일" outputs/work/gefs_fetch.log 2>/dev/null; do sleep 60; done
while [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'exp G7x2 ' }).Count")" != "0" ]; do sleep 30; done
log "run_gefs2 시작 (G7k 건너뜀)"
if [ ! -f outputs/work/gefs_kr.parquet ]; then
  log "빠진 파일 다시 받기"
  $PY -u src_tools/gfs_fetch.py --source gefs --workers 16 >> outputs/work/gefs_fetch.log 2>&1
  grep -a -c "실패" outputs/work/gefs_fetch.log
  log "해석"
  $GPY src_tools/gfs_decode.py --raw outputs/work/gefs_raw --vars DSWRFM DSWRFS TCDCM TCDCS --spread DSWRFS TCDCS \
    --prefix gefs --out outputs/work/gefs_kr.parquet > outputs/work/gefs_decode.log 2>&1 || log "  !! 해석 실패"
  tail -3 outputs/work/gefs_decode.log
  cp outputs/work/gefs_raw/manifest.csv outputs/gefs_ens_manifest.csv
fi
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
run G7x2 G5bT1x2 --add features_gfs:add_gefs_regions
run S3x2 S2T1x2 --reset-add --add features_gfs:add_gefs_regions
$PY src_tools/compare_years.py G5bT1x2 G7x2 | tail -1
$PY src_tools/compare_years.py S2T1x2 S3x2 | tail -1
log "GEFS 단일 모델 끝"
# 통과한 G7 쪽 고르기 (미리 정한 규칙: 세 해 모두 개선 중 가장 나쁜 해가 덜 나쁜 것, 없으면 G5bT1 유지)
G=$($PY - <<'EOF'
import sys
sys.path.insert(0, "src_tools")
from compare_years import YEARS, penalty
base = [penalty("G5bT1x2", c)[0] for _, c in YEARS]
best, bw = "G5bT1x2", 1.0
for n in ["G7x2"]:
    if any(penalty(n, c) is None for _, c in YEARS):
        continue
    r = [penalty(n, c)[0] / b for c, b in zip([c for _, c in YEARS], base)]
    if max(r) < 1 and max(r) < bw:
        best, bw = n, max(r)
print(best)
EOF
)
log "G 구성원: $G"
until [ -f outputs/work/submit_T.choice ]; do sleep 60; done
C=$(cat outputs/work/submit_T.choice)
if [ "$C" = T2 ]; then X=(E11-12T2x2 E11-11T2x2); else X=(E11-12x2 E11-11x2); fi
REF="E11-10${C}x2"
blend E12-10x2 "${X[@]}" "$G" S2T1x2 V07T1x2 S3x2
blend E12-10ax2 "${X[@]}" "$G" S2T1x2 V07T1x2
blend E12-10bx2 "${X[@]}" G5bT1x2 S2T1x2 V07T1x2 S3x2
$PY src_tools/compare_years.py "$REF" E12-10x2 E12-10ax2 E12-10bx2 | tail -3
for n in E12-10x2 E12-10ax2 E12-10bx2; do
  [ -f "outputs/compositions/${n}.csv" ] || $PY src_tools/optimize_composition.py --preds "$n" 2>&1 | grep -a -E "홀수|짝수|다른 해|최종|저장"
done
$PY src_tools/compare_own_comp.py E10-08x2 "$REF" E12-10x2 E12-10ax2 E12-10bx2
log "GEFS 앙상블 끝"
