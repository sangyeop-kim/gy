# SMT2020 Dataset 2: LV/HM 반도체 팹 시뮬레이션 데이터

이 저장소는 SMT2020(Semiconductor Manufacturing Testbed 2020)의 `dataset 2` 데이터를 CSV로 변환해 정리한 것이다. 원본 엑셀 파일은 `SMT_2020 - Final/General Data/dataset 2/SMT_2020_Model_Data_-_LVHM.xlsx`이며, 변환된 CSV 파일은 `dataset/` 폴더에 있다.

SMT2020은 현대 반도체 웨이퍼 팹의 복잡도를 연구용 이산사건 시뮬레이션(discrete-event simulation) 모델로 재현하기 위해 제안된 공개 테스트베드이다. 본 데이터는 그중 저용량/고혼합(Low-Volume/High-Mix, LV/HM) 팹을 나타내며, 10개 제품이 서로 다른 긴 공정 라우트를 따라 동일한 수준으로 투입되는 make-to-order 성격의 모델이다. lot별 납기, hot lot 우선순위, 설비 고장, 예방보전, 배치 공정, cascading, setup, sampling, rework, critical queue time(CQT) 같은 반도체 팹 운영 요소가 포함되어 있다.

참고 문헌:

> Denny Kopp, Michael Hassoun, Adar Kalir, and Lars Monch, "SMT2020 - A Semiconductor Manufacturing Testbed," IEEE Transactions on Semiconductor Manufacturing, Vol. 33, No. 4, 2020, pp. 522-531. DOI: 10.1109/TSM.2020.3001933

## 데이터 규모

- CSV 파일 수: 17개
- 제품 수: 10개 (`Product_1`부터 `Product_10`)
- 라우트 수: 10개 (`Route_Product_1`부터 `Route_Product_10`)
- tool group 수: 106개
- 총 tool 수: 1,313개
- lot release 기본 시작일: `2018-01-01`
- 상세 release horizon: `2018-01-01 00:00:00`부터 `2025-12-31 20:30:20`까지
- lot당 wafer 수: 25장
- 기본 우선순위: regular lot 10, hot lot 20, 일부 super hot lot 30

## 파일 구성

| 파일 | 행 수 | 설명 |
|---|---:|---|
| `Toolgroups.csv` | 106 | 공정 영역별 tool group, tool 수, loading/unloading time, dispatching rule |
| `PM.csv` | 292 | tool group별 예방보전(PM, scheduled downtime) 이벤트 |
| `Breakdown.csv` | 11 | 공정 영역별 비계획 다운타임/고장(UDT) 이벤트 |
| `Setups.csv` | 13 | setup group별 sequence-dependent setup 시간 |
| `Transport.csv` | 1 | fab 내부 tool group 간 이송 시간 분포 |
| `Lotrelease.csv` | 21 | 제품별 반복 release 패턴과 기준 due date |
| `Lotrelease - variable due dates.csv` | 167,129 | 개별 lot 단위 release 시각과 variable due date |
| `Route_Product_1.csv` | 521 | Product 1의 공정 라우트 |
| `Route_Product_2.csv` | 529 | Product 2의 공정 라우트 |
| `Route_Product_3.csv` | 583 | Product 3의 공정 라우트 |
| `Route_Product_4.csv` | 343 | Product 4의 공정 라우트 |
| `Route_Product_5.csv` | 242 | Product 5의 공정 라우트 |
| `Route_Product_6.csv` | 293 | Product 6의 공정 라우트 |
| `Route_Product_7.csv` | 353 | Product 7의 공정 라우트 |
| `Route_Product_8.csv` | 375 | Product 8의 공정 라우트 |
| `Route_Product_9.csv` | 384 | Product 9의 공정 라우트 |
| `Route_Product_10.csv` | 390 | Product 10의 공정 라우트 |

## 공정 영역과 설비 구성

`Toolgroups.csv`는 팹의 capacity structure를 정의한다. 각 행은 하나의 tool group이며, 동일한 성격의 tool 여러 대를 묶은 work center로 볼 수 있다.

| Area | Tool group 수 | Tool 수 |
|---|---:|---:|
| `Def_Met` | 7 | 15 |
| `Delay_32` | 1 | 400 |
| `Dielectric` | 10 | 54 |
| `Diffusion` | 10 | 73 |
| `Dry_Etch` | 21 | 312 |
| `Implant` | 9 | 36 |
| `Litho` | 11 | 170 |
| `Litho_Met` | 4 | 45 |
| `Planar` | 6 | 26 |
| `TF` | 11 | 79 |
| `TF_Met` | 2 | 4 |
| `Wet_Etch` | 14 | 99 |

주요 컬럼은 다음과 같다.

- `AREA`: diffusion, lithography, dry etch 등 공정 영역
- `TOOLGROUP`: 라우트 step에서 참조하는 설비 그룹 이름
- `NUMBER OF TOOLS`: 해당 tool group의 병렬 설비 수
- `CASCADINGTOOL`, `BACTHINGTOOL`: cascading 또는 batch processing 가능 여부
- `BATCHCRITERION`, `BATCHING UNIT`: batch 형성 기준
- `LOADINGTIME`, `UNLOADINGTIME`: lot 또는 batch의 loading/unloading 시간
- `DISPATCHING`, `Ranking 1-3`: dispatching 우선순위 규칙

대부분 tool group은 `SuperHotLotFIRST_and_Reservation` dispatching을 사용하며, 일부 setup 관련 tool group은 `Setupavoidancerule`을 사용한다.

## 제품 라우트 데이터

`Route_Product_*.csv` 파일은 각 제품이 거치는 공정 step sequence를 정의한다. 행 하나가 하나의 공정 step이다. Product 3이 583 step으로 가장 길고 Product 5가 242 step으로 가장 짧다.

| 제품 | Step 수 | 사용 area 수 | 사용 tool group 수 | Lot step | Wafer step | Batch step |
|---|---:|---:|---:|---:|---:|---:|
| Product 1 | 521 | 12 | 104 | 273 | 233 | 15 |
| Product 2 | 529 | 12 | 101 | 277 | 233 | 19 |
| Product 3 | 583 | 12 | 105 | 310 | 256 | 17 |
| Product 4 | 343 | 12 | 82 | 180 | 152 | 11 |
| Product 5 | 242 | 12 | 70 | 120 | 113 | 9 |
| Product 6 | 293 | 12 | 80 | 148 | 133 | 12 |
| Product 7 | 353 | 12 | 92 | 184 | 157 | 12 |
| Product 8 | 375 | 12 | 92 | 200 | 163 | 12 |
| Product 9 | 384 | 12 | 94 | 208 | 163 | 13 |
| Product 10 | 390 | 12 | 94 | 204 | 171 | 15 |

라우트 파일의 주요 컬럼은 다음과 같다.

- `ROUTE`, `STEP`, `STEP DESCRIPTION`: 라우트 이름, step 번호, 공정 설명
- `AREA`, `TOOLGROUP`: 해당 step을 처리하는 공정 영역과 설비 그룹
- `PROCESSING UNIT`: 처리 단위. `Wafer`, `Lot`, `Batch` 중 하나
- `PROCESSINGTIME DISTRIBUTION`, `MEAN`, `OFFSET`, `PT UNITS`: 처리 시간 분포와 파라미터
- `CASCADING INTERVAL`: cascading tool에서 후속 wafer/lot의 간격
- `BATCH MINIMUM`, `BATCH MAXIMUM`: batch 공정의 최소/최대 batch 크기
- `SETUP`, `WHEN`, `SETUP DISTRIBUTION`, `SETUP TIME`: setup 조건과 시간
- `STEP FOR LTL DEDICATION`: lithography lot-to-lens dedication 제약
- `REWORK PROBABILITY in %`, `STEP FOR REWORK`: rework 발생 확률과 되돌아갈 step
- `PROCESSING PROBABILITY in % (Sampling)`: metrology sampling rate
- `STEP FOR CRITICAL QUEUE TIME`, `CQT`, `CQTUNITS`: CQT 제약 구간과 제한 시간

전체 라우트 기준으로 setup이 정의된 step은 401개, lot-to-lens dedication step은 73개, rework probability가 있는 step은 52개, sampling probability가 있는 step은 955개, CQT 관련 step은 264개이다. Batch step은 135개, cascading interval이 있는 step은 1,668개이다.

## Lot release와 수요/납기

`Lotrelease.csv`는 제품별 반복 투입 패턴을 정의한다. Dataset 2는 10개 제품을 모두 사용하며, 각 제품에 대해 정규 lot과 hot lot이 정의되어 있다. 논문 설명 기준으로는 10,000 WSPW 작업점을 목표로 하며, 제품당 1,000 WSPW 수준의 저용량/고혼합 투입을 가정한다. Hot lot은 전체 lot의 약 2.5%로 설정된다.

`Lotrelease - variable due dates.csv`는 개별 lot 단위의 release와 due date를 제공한다. 총 167,129개 lot record가 있으며, regular lot 162,800개, hot lot 4,180개, super hot lot 149개가 포함되어 있다. 이 파일은 납기 기반 dispatching, critical ratio(CR) rule, on-time delivery 분석에 바로 사용할 수 있는 형태이다.

주요 컬럼은 다음과 같다.

- `PRODUCT NAME`: 제품 이름
- `ROUTE NAME`: 적용 라우트
- `LOT NAME/TYPE`: lot 이름 또는 lot 유형
- `PRIORITY`: dispatching 우선순위
- `SUPERHOTLOT`: super hot lot 여부
- `WAFERS PER LOT`: lot당 wafer 수
- `START DATE`: lot release 시각
- `DUE DATE`: 납기
- `Release Scenario`: release 또는 due date 생성 시나리오

## 고장, 예방보전, setup, 이송

`PM.csv`는 scheduled downtime을 나타낸다. 292개 PM 이벤트가 있으며, 213개는 counter-based PM, 79개는 calendar time-based PM이다. 각 PM은 tool group 단위로 정의되고, `MTBeforePM`, `TTR DISTRIBUTION`, `MEAN`, `OFFSET`, `FIRST ONE AT DISTRIBUTION` 등을 통해 PM 주기와 지속시간을 표현한다.

`Breakdown.csv`는 unscheduled downtime을 나타낸다. 11개 공정 영역에 대해 고장 이벤트가 정의되어 있으며, time before failure(TTF)와 time to repair(TTR)는 주로 exponential distribution으로 표현된다. 논문에서는 MTBF를 기본적으로 1주일로 두고, 공정 영역별 target availability와 SDT/UDT 비율에 맞추어 MTTR을 조정한다고 설명한다.

`Setups.csv`는 sequence-dependent setup 시간을 정의한다. 현재 데이터에는 13개 setup 전환 규칙이 있으며 setup time은 7분부터 80분까지 존재한다.

`Transport.csv`는 fab 내부에서 한 설비 공정이 끝난 lot이 다음 설비 공정으로 이동하는 이송 시간을 정의한다. 현재 데이터는 `Fab`에서 `Fab`으로 이동하는 단일 규칙이므로 tool group별 위치 matrix가 아니라 모든 연속 공정 사이에 적용되는 공통 설비 간 이동 지연으로 해석한다. Transport time은 uniform distribution으로 평균 7.5분, offset 2.5분이며, 즉 5-10분 범위의 이송 지연이다.

## 시뮬레이션 모델 관점에서의 해석

이 데이터셋은 특정 상용 시뮬레이터에만 묶이지 않는 generic format에 가깝다. 시뮬레이션 엔진을 직접 구현하거나 AnyLogic, SimPy, AutoSched AP 같은 이산사건 시뮬레이션 환경으로 옮길 때 다음 객체로 매핑할 수 있다.

- Product: `Product_1`부터 `Product_10`
- Route: `Route_Product_*.csv`의 step sequence
- Lot: `Lotrelease*.csv`의 개별 release entity
- Tool group: `Toolgroups.csv`의 work center
- Tool capacity: `NUMBER OF TOOLS`, loading/unloading time, batch/cascade 설정
- Process time: 라우트 step의 processing distribution, mean, offset
- Downtime: `PM.csv`와 `Breakdown.csv`
- Dispatching priority: lot priority, hot lot 여부, tool group ranking rule
- Operational constraints: setup, lot-to-lens dedication, sampling, rework, CQT

Dataset 2는 LV/HM foundry-like fab을 대상으로 하므로 단순 FCFS보다 due date와 remaining processing time을 고려하는 critical ratio(CR) 기반 dispatching 분석에 적합하다. 특히 제품이 10개이고 라우트 길이가 242-583 step으로 다양하기 때문에, 제품 mix 변화, 납기 준수율, 병목 tool group, WIP 흐름, hot lot 영향, PM/고장에 따른 capacity 변동성 등을 연구하기 좋은 구조이다.

## 변환 이력

원본 엑셀 파일의 17개 sheet를 각각 동일한 sheet 이름의 CSV 파일로 변환했다. CSV는 Excel 호환성을 고려해 `utf-8-sig` 인코딩으로 저장했다.

## 시뮬레이터 구조

기본 시뮬레이터 코드는 `src/simulator/` 패키지에 있다. 데이터셋에서 고정되어야 하는 부분과 실험마다 바뀔 수 있는 설정을 분리했다.

| 경로 | 역할 |
|---|---|
| `configs/default_simulation.json` | 기본 실험 설정과 하이퍼파라미터 |
| `configs/example_material_overrides.json` | 설비/PM/고장/setup/이송 조건 override 예시 |
| `src/simulator/config.py` | 설정 dataclass와 JSON 로더 |
| `src/simulator/domain.py` | Route, RouteStep, ToolGroup, Lot 등 도메인 객체 |
| `src/simulator/data_loader.py` | CSV를 읽어 불변 `FabModel` 구성 |
| `src/simulator/engine.py` | 이산사건 시뮬레이션 엔진 |
| `src/simulator/utils.py` | dispatching rule, random stream, metrics 등 보조 기능 |
| `src/simulator/cli.py` | 커맨드라인 실행 진입점 |
| `main.py` | CLI wrapper |

`FabModel` 안의 product별 route는 CSV에서 로드되는 고정 입력이다. 제품별로 어떤 step을 어떤 순서로 방문하는지는 실험 중 임의로 바꾸지 않는 전제이다. 반면 tool 수, PM, breakdown, setup, transport 같은 시뮬레이션 재료는 config에서 파일을 교체하거나 일부 값을 override할 수 있다.

Config에서 교체 가능한 입력 파일은 다음과 같다.

- `route_file_glob`: 제품 라우트 파일 glob. 보통 고정한다.
- `toolgroups_file`: tool group/capacity 데이터
- `pm_file`: 예방보전 데이터
- `breakdown_file`: 설비 고장/비계획 다운타임 데이터
- `setups_file`: sequence-dependent setup 데이터
- `transport_file`: 이송 시간 데이터
- `release_file`: lot release/due date 데이터

Config에서 일부 값만 바꾸는 override도 가능하다.

```json
{
  "toolgroup_overrides": {
    "LithoTrack_FE_95": {
      "number_of_tools": 18
    }
  },
  "transport_override": {
    "mean": 8.0,
    "offset": 3.0
  },
  "pm_overrides": {
    "LithoTrack_FE_95_WK": {
      "repair_mean": 6.0,
      "repair_offset": 1.0
    }
  },
  "breakdown_overrides": {
    "BREAK_Litho": {
      "mean_time_to_repair": 240.0
    }
  },
  "setup_overrides": {
    "DE_BE_13_1->DE_BE_13_2": {
      "setup_time": 10.0
    }
  }
}
```

override key는 로드된 객체의 ID를 사용한다. Tool group은 `TOOLGROUP`, PM은 `PM EVENT NAME`, breakdown은 `DOWN EVENT NAME`, setup은 기본적으로 `CURRENT SETUP->NEW SETUP` 형식의 key를 사용한다. 존재하지 않는 key나 dataclass에 없는 필드를 override하려고 하면 실행 전에 에러가 나도록 했다.

현재 엔진은 release, queueing, dispatching, transport time, stochastic process time, metrology sampling skip, simple rework뿐 아니라 PM, breakdown, batch processing, cascading interval, sequence-dependent setup, CQT violation 추적까지 처리한다. 기본 config는 이 기능들을 켠 상태이며, 필요하면 실험 목적에 맞게 각 flag를 끌 수 있다.

Tool group의 물리적 운영 속성은 `Toolgroups.csv`에서 읽어 반영한다. Cascading은 해당 tool group의 `CASCADINGTOOL`이 켜져 있고 route step에 `CASCADING INTERVAL`이 있을 때만 적용하며, batching은 `BACTHINGTOOL`, `BATCHCRITERION`, `BATCHING UNIT`과 route step의 `BATCH MINIMUM`/`BATCH MAXIMUM`을 함께 사용한다. 현재 데이터의 batch criterion인 `Same Product and Same Step`은 같은 제품/같은 step lot만 묶고, batching unit이 `Wafer`이면 wafer 수 기준으로 batch 크기를 계산한다.

직접 구현 엔진이므로 상용 SMT2020 모델의 세부 tool-level rule을 완전히 동일하게 복제한 것은 아니다. PM과 breakdown은 tool group capacity를 시간에 따라 줄였다가 복구하는 방식으로 반영하고, setup은 route step의 `SETUP` 상태 전환과 `Setups.csv`의 sequence-dependent setup time을 사용한다. 원본 `DISPATCHING` 문자열과 `Ranking 1-3`은 보존하지만, 정책 비교 실험에서는 config의 dispatching rule 또는 tool group override rule을 사용한다.

## 실행 방법

기본 설정으로 실행:

```bash
python main.py
```

작은 smoke test:

```bash
python main.py --max-lots 5 --output-dir outputs/smoke_5
```

재료 override 예시 실행:

```bash
python main.py --config configs/example_material_overrides.json
```

주요 옵션:

- `--config`: 사용할 JSON config 경로
- `--dataset-dir`: CSV 데이터 폴더
- `--output-dir`: 결과 저장 폴더
- `--max-lots`: release할 최대 lot 수
- `--until-minutes`: 지정 시점까지만 시뮬레이션
- `--seed`: random seed
- `--write-event-log`: event-level log 저장

실행 결과는 기본적으로 `summary.json`과 `lots.csv`로 저장된다. `summary.json`에는 완료 lot 수, 평균 cycle time, 평균 tardiness, on-time ratio가 들어가며, `lots.csv`에는 lot별 release/due/completion/cycle time이 저장된다.
