# H-JEPA 재현 기록

[H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning](https://arxiv.org/abs/2610.06805)
(코드: [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA))를 Kaggle에서 재현하는 과정의 실험 기록이다.
실험 하나가 끝날 때마다 `experiments/` 아래에 폴더를 하나 추가하고, 아래 표에 한 줄을 더한다.

## 실험 목록

| # | 날짜 | 실험 | 결과 한 줄 | 정리 |
|---|---|---|---|---|
| 01 | 2026-10-10 | 스모크 테스트: Visual AntMaze `ant_l3`, 과제 1개 | 파이프라인 정상 동작. 과제 1개 = 177.9초 (T4) | [정리](experiments/01_smoke_ant_l3/README.md) |

## 지금까지 알게 된 것

- 환경 구축부터 평가까지 Kaggle T4에서 저장소 코드 수정 없이 돈다. 구축은 약 2분이다.
- `ant_l3` 평가는 과제 1개에 약 178초가 걸린다. 평가 루프가 성공 여부와 무관하게 350스텝을 다 돌기 때문에
  과제당 시간은 거의 일정할 것으로 보인다. 단순 곱으로 50과제는 약 2.5시간이다 (추정, 실측 아님).
- 과제 수는 `eval.num_eval=N`으로 줄일 수 있다.
- 플래닝 롤아웃의 관측 프레임을 저장하는 옵션은 저장소에 없다. 저장하려면 코드 수정이 필요하다.

## 규칙

- 저장소(kevinghst/H-JEPA) 코드는 수정하지 않는다. 수정이 필요하면 먼저 결정한다.
- 플래너 설정(샘플 수, 반복 횟수 등)은 설정 파일 기본값을 쓴다.
- 보고하는 숫자는 모두 실제 로그와 결과 파일에서 읽은 값이다. 추정은 추정이라고 적는다.
