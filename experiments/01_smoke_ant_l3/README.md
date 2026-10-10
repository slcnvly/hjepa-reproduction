# 01. 스모크 테스트: Visual AntMaze `ant_l3`, 과제 1개 (2026-10-10)

**결과: 성공.** 환경 구축부터 평가까지 끝까지 돌았고, 과제 1개는 T4에서 **177.9초** 걸렸다.

목표는 재현 성공 여부 판정이 아니다. 파이프라인이 도는지 확인하고, 본 실험 규모를 정하기 위해 과제 1개당 시간을 재는 것이다.

## 바로 가기

| 무엇 | 링크 |
|---|---|
| Kaggle 커널 | https://www.kaggle.com/code/seungwonryoo/hjepa-smoke-ant-l3 |
| 요약 리포트 (단계별 시간, 과제 시간) | [results/smoke_report.json](results/smoke_report.json) |
| 과제 0의 결과 json | [results/tasks_000-001.json](results/tasks_000-001.json) |
| 평가 로그 | [results/logs/eval.log](results/logs/eval.log) |
| 실제 적용된 평가 설정 | [results/eval_config.yaml](results/eval_config.yaml) |
| 3.11 환경 패키지 목록 | [results/pip_freeze_venv311.txt](results/pip_freeze_venv311.txt) |
| 커널 전체 로그 | [results/kaggle_kernel_full.log](results/kaggle_kernel_full.log) |
| 커널 스크립트 | [run.py](run.py) |
| 사용한 저장소 커밋 | https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5 |

## 실행한 것

- Kaggle 커널 `seungwonryoo/hjepa-smoke-ant-l3` v1, 비공개, T4 2장, 전체 실행 297.5초.
- 저장소는 커밋 `97054ec` 그대로 썼다. 작성한 것은 환경 구축·실행 스크립트 [run.py](run.py)뿐이다.
- 평가 명령:

```bash
python -u eval.py --config-name ant_l3 seed=42 output.dir=smoke +eval.chunk_size=1 \
  policy=/kaggle/working/hjepa/ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt
```

- `chunks/tasks_000-001.json`이 생기는 즉시 프로세스를 끊었다 (50개를 다 돌리지 않음).

## 환경 구축

| 단계 | 내용 | 시간 |
|---|---|---|
| 가상환경 | uv로 Python 3.11.13, `/kaggle/temp/venv311` | 1.9초 |
| 설치 | `uv pip install -e '.[train,env]'`, PyTorch cu128 인덱스 | 60.3초 |
| 평가 과제 다운로드 | `eval_trajs/*`, sha256 검증 통과 | 12.6초 |
| 체크포인트 다운로드 | `ant/ant_hjepa_l3/seed42/*` (140MB) | 2.5초 |

환경변수: `MUJOCO_GL=egl`, `MPLBACKEND=Agg`, `HJEPA_HOME=/kaggle/working/hjepa`. 시스템 파이썬은 건드리지 않았다.

검증 결과 ([results/logs/verify_basic.log](results/logs/verify_basic.log) 등에서 읽은 값):

| 항목 | 값 |
|---|---|
| Python | 3.11.13 |
| torch | 2.7.1+cu128 |
| `torch.cuda.is_available()` | True |
| `torch.cuda.device_count()` | 2 (Tesla T4, Tesla T4) |
| import | `stable_worldmodel`, `eval`, `droid_eval`, `planning_eval` 성공 |
| EGL 렌더링 | 정상 (mujoco 3.14.0). osmesa 재시도 불필요 |

`h_jepa`는 설치 패키지가 아니라 스크립트 폴더다. 저장소 루트에서 namespace 패키지로만 import된다.

## 과제 1개 소요 시간

| 측정 | 값 |
|---|---|
| 평가 구간 (`evaluation_time`) | **177.9초** |
| 프로세스 시작 → json 생성 (모델 로드 포함) | 190.0초 |
| 플래너 호출 | 36회 |
| solve 기록 | 95건 |
| solve 시간 합계: level 1 / 2 / 3 | 51.8초 / 33.4초 / 56.9초 |
| 이 과제의 성공 여부 | 실패 (`episode_successes: [false]`) |

- 과제 1개의 성공·실패는 의미를 둘 수 없다.
- 평가 루프는 성공해도 조기 종료하지 않고 350스텝(`eval_budget`)을 다 돈다. 그래서 과제당 시간은 거의 일정할 것으로 보인다.
- 단순 곱으로 50과제는 약 2.5시간이다 (설정 1개, 시드 1개 기준). 이 값은 실측이 아니라 추정이다.

## 처음 지시와 다르게 한 점

- **체크포인트 범위**: `ant/*`는 7개 모델 × 3시드로 2.6GB다. 필요한 `ant/ant_hjepa_l3/seed42/*`만 받았다. 폴더명 `ant`는 맞았다.
- **uv 옵션**: `--index-strategy unsafe-best-match`를 추가했다. pip처럼 PyTorch 인덱스와 PyPI를 함께 보게 하기 위해서다.
- **체크포인트 삭제**: 커널 출력을 작게 유지하려고 종료 시 `/kaggle/working/hjepa`를 지웠다. 결과 json, 설정, 로그는 남겼다.

## 코드 조사 (읽기만 함, 실행·수정 없음)

### 플래닝 롤아웃의 관측 프레임을 저장하는 옵션이 있는가: 없다

Hydra 키, 플래그, 함수 어느 것도 없다.

- 프레임은 `AddPixelsWrapper`(`stable_worldmodel/wrapper.py`, `_get_pixels`)가 매 스텝 렌더링해 `info['pixels']`에 넣는다.
  정책 입력으로만 쓰이고 버려진다.
- 저장하려면 `World.evaluate_from_dataset`(`stable_worldmodel/world.py`)의 스텝 루프
  `for i in range(eval_budget):` 안, `self.step()` 직후에 `self.infos['pixels']`를 모아 쓰는 코드를 넣어야 한다.
  목표 이미지는 같은 위치의 `self.infos['goal']`이다.
- `stable_worldmodel/utils.py`의 `record_video_from_dataset`은 저장된 HDF5 데이터셋을 재생하는 용도라 플래닝 롤아웃에는 쓸 수 없다.

### 과제 수를 50개보다 적게 제한하는 오버라이드가 있는가: 있다, `eval.num_eval=N`

- 로드한 50개 중 앞의 N개만 평가한다 (`h_jepa/planning_eval.py`, `_run_chunked_eval`). N개가 끝나면 `metrics.yaml`까지 정상 기록된다.
- `eval.chunk_size`는 기본값이 이미 1이다. `+eval.chunk_size=1`은 없어도 동작이 같다.
- 청크마다 플래너 시드를 `seed + 시작 인덱스`로 다시 잡는다. 과제를 나눠 돌려도 과제별 플래너 시드는 같다.

### 그 밖에 확인한 것

- `eval.py`가 맨 위에서 `os.environ['MUJOCO_GL'] = 'egl'`로 덮어쓴다. 환경변수로 osmesa를 줘도 무시된다.
- 결과는 체크포인트 옆 `<output.dir>/chunks/tasks_XXX-YYY.json`에 과제(청크)별로 쌓이고, 다시 실행하면 이미 있는 청크는 건너뛴다.
- 평가만 할 때는 학습 데이터셋(HDF5)이 필요 없다. 정규화 값은 체크포인트 옆 `normalizer.pt`에서 읽는다.
