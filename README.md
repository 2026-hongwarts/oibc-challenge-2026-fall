# OIBC 2026 태양광 집합자원 — 팀 저장소

2026 POSTECH OIBC Big Data Challenge. 태양광 발전소 253곳을 두 묶음(G1, G2)으로 나누고, 묶음별 시간 발전량을 하루 전에 예측한다. 정산금 S에서 수익은 모든 팀이 같으므로 승부는 예측 오차 벌금이다. 마감 2026-11-13.

- 역할 분담·일정: https://claude.ai/artifact/VqqzYusqpUCfzMm7EiFk4i
- 계획서: [docs/eda_preprocessing_plan.md](docs/eda_preprocessing_plan.md)
- EDA: [notebooks/eda_v2.ipynb](notebooks/eda_v2.ipynb) — 데이터 품질과 점수 구조를 분석해 전처리 규칙과 팀의 우선순위를 정한 근거

## 폴더

```
├─ src/                   코드. 팀원이 만드는 피처 파일(features_*.py)도 여기
│  ├─ oibc_common.py      공통 함수: 점수식, 데이터 읽기, 태양 기하, 정지일 탐지, 구성 탐색
│  ├─ build_base.py       공통 테이블 만들기
│  ├─ train_site_model.py 발전소 단위 LightGBM 학습·예측 (옵션 없이 돌리면 baseline 재현)
│  └─ evaluate.py         가상 평가연도(2024-06 ~ 2025-05) 채점기
├─ notebooks/             분석 노트북
│  └─ eda_v2.ipynb        EDA. 데이터 품질과 점수 구조를 분석해 전처리 규칙과 팀의 우선순위를 정한 근거
├─ docs/                  계획서, 대회 설명
├─ outputs/               실행하면 생기는 파일
│  ├─ work/               공통 테이블                      (git 제외)
│  ├─ preds/              가상 평가연도 예측. *.json 과 champion.txt 만 git 에 올림
│  ├─ preds_test/         실제 평가 기간 예측               (git 제외)
│  ├─ submissions/        제출 파일                         (git 제외)
│  └─ eda/                노트북 산출물                     (git 제외)
├─ 01_participant_dataset/  대회 데이터 — 각자 받아서 둔다  (git 제외)
└─ archive/               옛 버전                           (git 제외)
```

## EDA에서 정한 것

탐색적 분석으로 데이터를 정제할 규칙을 정하고, 점수가 어디서 갈리는지와 구성 최적화의 기대 효과를 수치로 확인했다. 이를 바탕으로 역할 분담과 실험 순서를 정했다.

| 확인한 것 | 그래서 정한 것 |
|---|---|
| 수익은 모든 팀이 같고, 벌금의 82.5%가 11~15시에서 나옴 | 한낮 예측 정확도에 집중 |
| 구성으로 줄일 수 있는 벌금은 약 1%, 지역별로 나누면 +12% | 구성은 지역 분할만 피하고 마지막에 다듬기 |
| 고장 난 날 729건, 편입 후 결측은 거의 없음, 평가 기간 예보 264행 빠짐 | 정지일은 학습에서 제외, 보간은 안 함, 빠진 예보는 다른 라운드로 채움 |
| 운량·습도·운고(`wb_ceiling`)가 핵심 변수, 두 라운드 평균이 더 나음 | 피처 우선순위 |
| 가상 평가연도 채점 방식의 원형을 만듦 | `src/evaluate.py`로 정식화 |

노트북 §9의 빠른 모델(벌금 223,789,723원)은 채점 방식을 시험한 원형이다. 팀의 공식 기준 모델은 `src/train_site_model.py`로 만든 E4-00x2(223,140,191원)다.

## 준비

명령은 모두 저장소 맨 위 폴더에서 실행한다.

1. Python 3.10 이상, `pip install pandas pyarrow lightgbm` (청천 일사 작업은 `pvlib`도)
2. 대회 데이터를 `01_participant_dataset/`에 둔다.
3. 공통 테이블 만들기 (약 2분, 메모리 약 6GB)
   ```
   python src/build_base.py
   ```
4. 기준 모델 예측을 각자 PC에서 한 번 만든다. PC나 라이브러리 버전에 따라 숫자가 조금 다를 수 있으니, 비교는 같은 PC에서 만든 결과끼리 한다.
   ```
   python src/train_site_model.py --exp E4-00x2 --seeds 2
   ```

## 실험

```
python src/train_site_model.py --exp E3-01 --drop-dead                # 탐색용 (시드 1개)
python src/train_site_model.py --exp E3-01x2 --drop-dead --seeds 2    # 채택 판단용 (시드 2개 평균)
python src/train_site_model.py --exp E2-02 --add features_weather:add_weather_features
python src/evaluate.py outputs/preds/E3-01x2.parquet                  # 채점만 다시
```

- 결과는 `outputs/preds/실험번호.parquet`에 저장되고, `outputs/preds/champion.txt`가 가리키는 기준 모델과 바로 비교된다.
- 옵션: `--drop-dead` 정지일 제외 · `--rounds both|10|17|avg` · `--years N` 학습 기간 · `--weight cap|cap_k` · `--objective l1` · `--params '{"num_leaves": 63}'` · `--mode test` 실제 평가 기간 예측
- 피처 함수는 `src/features_*.py`에 만든다. 규칙은 [src/train_site_model.py](src/train_site_model.py) 맨 위 설명에 있다. 발전량으로 만드는 값은 `df[df.is_fit]`에서만 계산한다.

## 규칙

- 실험 번호는 `E역할-순번`(예: `E2-01`), 시드 2개 결과는 끝에 `x2`. 한 실험에 바꾸는 것은 하나.
- 시드만 바꿔도 벌금이 약 0.5% 움직인다. 채택은 `--seeds 2` 결과가 기준 모델보다 벌금 0.5% 이상 낮고 11~15시도 낮을 때.
- 금지: 실측 기상(`observed_*`)을 입력으로 쓰기 · 2024-06 이후 발전량으로 만든 값을 학습에 쓰기 · 가상 평가연도로 조기 종료 · 좌표로 실제 위치 찾기(실격) · 대회 데이터 업로드
