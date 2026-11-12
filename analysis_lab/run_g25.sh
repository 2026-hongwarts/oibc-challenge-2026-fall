#!/usr/bin/env bash
# GFS 0.25° 1시간 일사량(G10, add_gfs25_regions) — 받기 끝 기다림 → 해석 → 세 해 실험 → 앙상블 → (통과 시) 시드 재현 → 제출 파일.
# 미리 정한 판정(세 해 모두 개선, 참조 구성):
#   G5bHx2 = G5bT1x2 + G10 vs G5bT1x2,  S2Hx2 = S2T1x2 + G10 vs S2T1x2
#   앙상블: E17Gx2 = E13-13 의 G5bT1 자리를 G5bH 로 / E17Sx2 = E13-13 + S2H (일곱 같은 비중) vs E13-13x2
#   둘 다 통과하면 가장 나쁜 해가 덜 나쁜 것. 시드 재현(시드 2·3): 바꾼 구성원의 s2 판으로 같은 비교.
#   → 평가 기간 학습 → 빠짐 처리 → 일출·일몰 → 제출 파일(구성 E12-10cx2.csv) + 공식 형식 검사
#   bash analysis_lab/run_g25.sh >> outputs/work/run_g25.log 2>&1   (끝난 것은 건너뜀)
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
blend() { local out=$1; shift; for c in "" "--cut 2023-06-01" "--cut 2022-06-01"; do $PY src_tools/blend.py --out "$out" "$@" $c > /dev/null 2>&1 || log "  !! 섞기 실패 $out $c"; done; }
pass() { $PY src_tools/compare_years.py "$1" "$2" | tail -1 | tee -a /dev/stderr | grep -q "통과"; }
while [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match ('gfs25_'+'fetch') }).Count")" != "0" ]; do sleep 60; done   # 받기 프로세스가 모두 끝날 때까지
N=$(ls outputs/work/gfs25_crop/*.npz | wc -l)
if [ "$N" -lt 1800 ]; then log "받은 날 $N 일 — 너무 적어 멈춤(받기 실패 확인 필요)"; log "G25 끝"; exit 0; fi
log "받기 끝 ($(ls outputs/work/gfs25_crop/*.npz | wc -l)일) → 해석"
$PY src_tools/gfs25_decode.py
cat outputs/work/gfs25_crop/manifest_*.csv | grep -v "^run_date" | sort -u > outputs/gfs25_manifest_rows.csv
B6=(E11-12x2 E11-11x2 G5bT1x2 S3x2 V07T1x2 M2x2)
run S2Hx2 S2T1x2 --add features_gfs:add_gfs25_regions
run G5bHx2 G5bT1x2 --add features_gfs:add_gfs25_regions
$PY src_tools/compare_years.py S2T1x2 S2Hx2 | tail -1
$PY src_tools/compare_years.py G5bT1x2 G5bHx2 | tail -1
blend E17Gx2 E11-12x2 E11-11x2 G5bHx2 S3x2 V07T1x2 M2x2
blend E17Sx2 "${B6[@]}" S2Hx2
RG=$($PY src_tools/compare_years.py E13-13x2 E17Gx2 | tail -1); echo "$RG"
RS=$($PY src_tools/compare_years.py E13-13x2 E17Sx2 | tail -1); echo "$RS"
C=""
echo "$RG" | grep -q "통과" && C=G
echo "$RS" | grep -q "통과" && { [ -z "$C" ] && C=S || C=GS; }
if [ -z "$C" ]; then log "앙상블 불통과 → 끝"; log "G25 끝"; exit 0; fi
if [ "$C" = GS ]; then   # 둘 다 통과: 가장 나쁜 해가 덜 나쁜 것
  C=$($PY - <<'PY'
import sys; sys.path.insert(0, "src_tools")
from compare_years import YEARS, penalty
b = [penalty("E13-13x2", c)[0] for _, c in YEARS]
r = {k: max(penalty(n, c)[0] / x for (_, c), x in zip(YEARS, b)) for k, n in [("G", "E17Gx2"), ("S", "E17Sx2")]}
print(min(r, key=r.get))
PY
)
fi
log "선택 E17${C}x2 → 시드 재현"
if [ "$C" = G ]; then
  run G5bHs2x2 G5bHx2 --seed 2
  blend E17Gs2x2 E11-12x2 E11-11x2 G5bHs2x2 S3x2 V07T1x2 M2x2
  blend E13-13gs2x2 E11-12x2 E11-11x2 G5bT1s2x2 S3x2 V07T1x2 M2x2
  R2=$($PY src_tools/compare_years.py E13-13gs2x2 E17Gs2x2 | tail -1); M=(E11-12x2 E11-11x2 G5bHx2 S3x2 V07T1x2 M2x2)
else
  run S2Hs2x2 S2Hx2 --seed 2
  blend E17Ss2x2 "${B6[@]}" S2Hs2x2
  R2=$($PY src_tools/compare_years.py E13-13x2 E17Ss2x2 | tail -1); M=("${B6[@]}" S2Hx2)
fi
echo "$R2"
if ! echo "$R2" | grep -q "통과"; then log "시드 재현 불통과 → 끝"; log "G25 끝"; exit 0; fi
for m in "${M[@]}"; do
  [ -f "outputs/preds_test/${m}.parquet" ] && continue
  log "$m 평가 기간 학습"; $PY src_tools/train_site_model.py --base "$m" --mode test > "outputs/work/xbatch/${m}_test.log" 2>&1 || log "  !! 실패 $m"
done
$PY src_tools/blend.py --out "E17${C}fx2" "${M[@]}" --mode test 2>&1 | grep -a "빠짐\|저장"
$PY src_tools/twilight_fix.py --preds "E17${C}fx2" --out "E17${C}ftx2" 2>&1 | grep -a "preds_test"
$PY src_tools/make_submission.py "outputs/preds_test/E17${C}ftx2.parquet" --composition outputs/compositions/E12-10cx2.csv --name "E17${C}ftx2_comp" 2>&1 | tail -2
touch "outputs/submissions/E17${C}ftx2_comp/.done"
$PY outputs/work/validate_submission.py --composition "outputs/submissions/E17${C}ftx2_comp/composition.csv" --forecast "outputs/submissions/E17${C}ftx2_comp/forecast.csv" --data 01_participant_dataset | head -1
log "G25 끝"
