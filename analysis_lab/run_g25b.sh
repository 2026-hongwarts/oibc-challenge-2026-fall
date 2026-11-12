#!/usr/bin/env bash
# 시간별 GFS 일사량(G10)을 나머지 GFS 사용 구성원(XGB 둘, MLP)에도 — GPU (11/12 밤 사용자 허락).
# 미리 정한 판정(세 해 모두 개선, 참조 구성, 같은 장치끼리):
#   단독(참고): E11-11Hgx2 vs E11-11gx2, E11-12Hgx2 vs E11-12gx2, M2Hx2 vs M2x2
#   앙상블: E18x2 = [E11-12Hg, E11-11Hg, G5bH, S3, V07T1, M2H]  vs  E17Ggx2 = [E11-12g, E11-11g, G5bH, S3, V07T1, M2]
#   시드 재현: 바꾼 셋의 시드 2·3 판 vs 기준의 시드 2·3 판(E11-12gs2, E11-11gs2, M2s2)
#   → 평가 기간 학습(XGB 는 GPU) → 빠짐 처리 → 일출·일몰 → 제출 파일 E18ftx2_comp(구성 E12-10cx2.csv) + 공식 형식 검사
#   bash analysis_lab/run_g25b.sh >> outputs/work/run_g25b.log 2>&1   (끝난 것은 건너뜀)
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
G10=(--add features_gfs:add_gfs25_regions)
run E11-11Hgx2 E11-11gx2 "${G10[@]}"
run E11-12Hgx2 E11-12gx2 "${G10[@]}"
run M2Hx2 M2x2 "${G10[@]}"
$PY src_tools/compare_years.py E11-11gx2 E11-11Hgx2 | tail -1
$PY src_tools/compare_years.py E11-12gx2 E11-12Hgx2 | tail -1
$PY src_tools/compare_years.py M2x2 M2Hx2 | tail -1
blend E17Ggx2 E11-12gx2 E11-11gx2 G5bHx2 S3x2 V07T1x2 M2x2
blend E18x2 E11-12Hgx2 E11-11Hgx2 G5bHx2 S3x2 V07T1x2 M2Hx2
R=$($PY src_tools/compare_years.py E17Ggx2 E18x2 | tail -1); echo "$R"
if ! echo "$R" | grep -q "통과"; then log "앙상블 불통과 → 끝"; log "G25b 끝"; exit 0; fi
log "시드 재현"
run E11-11gs2x2 E11-11gx2 --seed 2
run E11-12gs2x2 E11-12gx2 --seed 2
run E11-11Hgs2x2 E11-11Hgx2 --seed 2
run E11-12Hgs2x2 E11-12Hgx2 --seed 2
run M2Hs2x2 M2Hx2 --seed 2
blend E17Ggs2x2 E11-12gs2x2 E11-11gs2x2 G5bHx2 S3x2 V07T1x2 M2s2x2
blend E18s2x2 E11-12Hgs2x2 E11-11Hgs2x2 G5bHx2 S3x2 V07T1x2 M2Hs2x2
R2=$($PY src_tools/compare_years.py E17Ggs2x2 E18s2x2 | tail -1); echo "$R2"
if ! echo "$R2" | grep -q "통과"; then log "시드 재현 불통과 → 끝"; log "G25b 끝"; exit 0; fi
M=(E11-12Hgx2 E11-11Hgx2 G5bHx2 S3x2 V07T1x2 M2Hx2)
for m in "${M[@]}"; do
  [ -f "outputs/preds_test/${m}.parquet" ] && continue
  log "$m 평가 기간 학습"; $PY src_tools/train_site_model.py --base "$m" --mode test > "outputs/work/xbatch/${m}_test.log" 2>&1 || log "  !! 실패 $m"
done
$PY src_tools/blend.py --out E18fx2 "${M[@]}" --mode test 2>&1 | grep -a "빠짐\|저장"
$PY src_tools/twilight_fix.py --preds E18fx2 --out E18ftx2 2>&1 | grep -a "preds_test"
$PY src_tools/make_submission.py outputs/preds_test/E18ftx2.parquet --composition outputs/compositions/E12-10cx2.csv --name E18ftx2_comp 2>&1 | tail -2
touch outputs/submissions/E18ftx2_comp/.done
$PY outputs/work/validate_submission.py --composition outputs/submissions/E18ftx2_comp/composition.csv --forecast outputs/submissions/E18ftx2_comp/forecast.csv --data 01_participant_dataset | head -1
log "G25b 끝"
