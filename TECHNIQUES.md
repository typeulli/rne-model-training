# 기술 목록 — 어떤 모델이 무엇을 쓰는가

`models/` 아래 등록된 21개 모델(레거시 `*_old` 3종 제외)이 각각 어떤 기술을 쓰는지
한 표에 모았다. 표의 기술 이름은 모두 아래 설명 절로 연결된다.

모델들은 서로의 **대조군**으로 설계되어 있다. 두 모델이 한 칸만 다르면 그 칸이
숫자 차이의 원인이라는 뜻이고, 그것이 이 표를 읽는 방법이다. 각 모델의 정체성은
README.md와 각 패키지의 `__init__.py` docstring에 있으며, 여기서는 *기술* 단위로
쪼개서 정리한다.

---

## 모델별 기술 표

| 모델 | 아키텍처 | 물리 prior | 목적함수 | 코퍼스·샘플링 | 학습·최적화 |
|---|---|---|---|---|---|
| `mlp` | [밀집 스택](#t-dense), [활성화 선택](#t-activation) | — | [dT 정규화 MSE](#t-scaled-mse) | [power hold-out](#t-holdout-power), [초기 시점 제외](#t-exclude) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `gmlp` | [밀집 스택](#t-dense), [활성화 선택](#t-activation) | [가우시안 게이트](#t-gauss-gate), [게이트 바닥값 p](#t-gate-offset), [단위 피크 정규화](#t-unit-peak) | [dT 정규화 MSE](#t-scaled-mse) | [power hold-out](#t-holdout-power), [초기 시점 제외](#t-exclude), [물리 상수 확보](#t-calibrate) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `pimlp` | [밀집 스택](#t-dense), [활성화 선택](#t-activation) | — | [PINN 잔차](#t-pinn), [잔차 무차원화](#t-residual-scaling), [collocation 재샘플링](#t-resampled-collocation) | [power hold-out](#t-holdout-power), [초기 시점 제외](#t-exclude), [물리 상수 확보](#t-calibrate) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `cmlp` | [밀집 스택](#t-dense), [활성화 선택](#t-activation) | — | [dT 정규화 MSE](#t-scaled-mse) | [P 제거](#t-p-blind), [초기 시점 제외](#t-exclude), [창 양도](#t-cede) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs), [웜스타트](#t-warm-start) |
| `cgmlp` | [밀집 스택](#t-dense), [활성화 선택](#t-activation) | [가우시안 게이트](#t-gauss-gate), [게이트 바닥값 p](#t-gate-offset), [단위 피크 정규화](#t-unit-peak) | [dT 정규화 MSE](#t-scaled-mse) | [P 제거](#t-p-blind), [초기 시점 제외](#t-exclude), [창 양도](#t-cede), [물리 상수 확보](#t-calibrate) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs), [웜스타트](#t-warm-start) |
| `cjmlp` | [밀집 스택](#t-dense), [활성화 선택](#t-activation) | [Rosenthal 게이트](#t-rosenthal), [영상법](#t-image-source), [소프트 코어](#t-soft-core), [게이트 바닥값 p](#t-gate-offset), [단위 피크 정규화](#t-unit-peak) | [dT 정규화 MSE](#t-scaled-mse) | [P 제거](#t-p-blind), [초기 시점 제외](#t-exclude), [창 양도](#t-cede), [물리 상수 확보](#t-calibrate) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs), [웜스타트](#t-warm-start) |
| `cpimlp` | [밀집 스택](#t-dense), [활성화 선택](#t-activation) | — | [PINN 잔차](#t-pinn), [잔차 무차원화](#t-residual-scaling), [collocation 재샘플링](#t-resampled-collocation), [물리항 power 고정](#t-physics-power) | [P 제거](#t-p-blind), [초기 시점 제외](#t-exclude), [물리 상수 확보](#t-calibrate) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs), [웜스타트](#t-warm-start) |
| `cpkmlp` | [밀집 스택](#t-dense), [활성화 선택](#t-activation) | [빔 앵커](#t-beam-anchor) | [dT 정규화 MSE](#t-scaled-mse) | [패치 코퍼스](#t-patch-corpus), [패치 페이스트](#t-paste), [P 제거](#t-p-blind), [초기 시점 제외](#t-exclude), [물리 상수 확보](#t-calibrate) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs), [웜스타트](#t-warm-start) |
| `fmlp` | [밀집 스택](#t-dense), [스펙트럼 출력](#t-spectral), [활성화 선택](#t-activation) | — | [Parseval 계수 MSE](#t-parseval) | [xy 전용 변환](#t-rfft-xy), [에너지 모드 절단](#t-energy-box), [위상 역회전](#t-derotate), [x 램프 분리](#t-detrend), [power hold-out](#t-holdout-power), [초기 시점 제외](#t-exclude), [물리 상수 확보](#t-calibrate) | [Adam+코사인](#t-adam-cosine) |
| `don` | [DeepONet](#t-deeponet), [툴패스 branch](#t-toolpath-branch), [활성화 선택](#t-activation) | — | [dT 정규화 MSE](#t-scaled-mse) | [power hold-out](#t-holdout-power), [조건별 균등 배칭](#t-equal-condition-batch), [인덱스 디코딩](#t-index-decoded) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `gdon` | [DeepONet](#t-deeponet), [툴패스 branch](#t-toolpath-branch), [활성화 선택](#t-activation) | [가우시안 게이트](#t-gauss-gate), [게이트 바닥값 p](#t-gate-offset), [단위 피크 정규화](#t-unit-peak), [물리 상수 확보](#t-calibrate) | [dT 정규화 MSE](#t-scaled-mse) | [power hold-out](#t-holdout-power), [조건별 균등 배칭](#t-equal-condition-batch), [인덱스 디코딩](#t-index-decoded) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `jdon` | [DeepONet](#t-deeponet), [툴패스 branch](#t-toolpath-branch), [활성화 선택](#t-activation) | [Rosenthal 게이트](#t-rosenthal), [영상법](#t-image-source), [소프트 코어](#t-soft-core), [빔 진행 프레임](#t-beam-frame), [게이트 바닥값 p](#t-gate-offset), [단위 피크 정규화](#t-unit-peak), [물리 상수 확보](#t-calibrate) | [dT 정규화 MSE](#t-scaled-mse) | [power hold-out](#t-holdout-power), [조건별 균등 배칭](#t-equal-condition-batch), [인덱스 디코딩](#t-index-decoded) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `pkdon` | [DeepONet](#t-deeponet), [툴패스 branch](#t-toolpath-branch), [활성화 선택](#t-activation) | [빔 진행 프레임](#t-beam-frame), [빔 앵커](#t-beam-anchor) | [dT 정규화 MSE](#t-scaled-mse) | [패치 코퍼스](#t-patch-corpus), [패치 페이스트](#t-paste), [power hold-out](#t-holdout-power), [조건별 균등 배칭](#t-equal-condition-batch) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `fdon` | [DeepONet](#t-deeponet), [툴패스 branch](#t-toolpath-branch), [시퀀스 출력](#t-sequence), [활성화 선택](#t-activation) | — | [마스크 손실](#t-masked-loss), [dT 정규화 MSE](#t-scaled-mse) | [power hold-out](#t-holdout-power), [점 단위 분할](#t-point-split), [창 양도](#t-cede), [조건별 균등 배칭](#t-equal-condition-batch), [인덱스 디코딩](#t-index-decoded) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `gfdon` | [DeepONet](#t-deeponet), [툴패스 branch](#t-toolpath-branch), [시퀀스 출력](#t-sequence), [활성화 선택](#t-activation) | [가우시안 게이트](#t-gauss-gate), [게이트 바닥값 p](#t-gate-offset), [단위 피크 정규화](#t-unit-peak), [물리 상수 확보](#t-calibrate) | [마스크 손실](#t-masked-loss), [dT 정규화 MSE](#t-scaled-mse) | [power hold-out](#t-holdout-power), [점 단위 분할](#t-point-split), [창 양도](#t-cede), [조건별 균등 배칭](#t-equal-condition-batch), [인덱스 디코딩](#t-index-decoded) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `jfdon` | [DeepONet](#t-deeponet), [툴패스 branch](#t-toolpath-branch), [시퀀스 출력](#t-sequence), [활성화 선택](#t-activation) | [Rosenthal 게이트](#t-rosenthal), [영상법](#t-image-source), [소프트 코어](#t-soft-core), [빔 진행 프레임](#t-beam-frame), [게이트 바닥값 p](#t-gate-offset), [단위 피크 정규화](#t-unit-peak), [물리 상수 확보](#t-calibrate) | [마스크 손실](#t-masked-loss), [dT 정규화 MSE](#t-scaled-mse) | [power hold-out](#t-holdout-power), [점 단위 분할](#t-point-split), [창 양도](#t-cede), [조건별 균등 배칭](#t-equal-condition-batch), [인덱스 디코딩](#t-index-decoded) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `pkfdon` | [DeepONet](#t-deeponet), [툴패스 branch](#t-toolpath-branch), [시퀀스 출력](#t-sequence), [활성화 선택](#t-activation) | [빔 진행 프레임](#t-beam-frame), [빔 앵커](#t-beam-anchor) | [마스크 손실](#t-masked-loss), [dT 정규화 MSE](#t-scaled-mse) | [패치 코퍼스](#t-patch-corpus), [격자 보간](#t-lattice-interp), [패치 페이스트](#t-paste), [power hold-out](#t-holdout-power), [점 단위 분할](#t-point-split), [조건별 균등 배칭](#t-equal-condition-batch) | [Adam+코사인](#t-adam-cosine), [L-BFGS](#t-lbfgs) |
| `rfdon` | `fdon`과 동일 | — | `fdon`과 동일 | `fdon` + [소등 프레임 고정](#t-retouched) | `fdon`과 동일 |
| `rgfdon` | `gfdon`과 동일 | `gfdon`과 동일 | `gfdon`과 동일 | `gfdon` + [소등 프레임 고정](#t-retouched) | `gfdon`과 동일 |
| `rjfdon` | `jfdon`과 동일 | `jfdon`과 동일 | `jfdon`과 동일 | `jfdon` + [소등 프레임 고정](#t-retouched) | `jfdon`과 동일 |
| `rpkfdon` | `pkfdon`과 동일 | `pkfdon`과 동일 | `pkfdon`과 동일 | `pkfdon` + [소등 프레임 고정](#t-retouched) | `pkfdon`과 동일 |
| **전 모델 공통** | [정규화 버퍼](#t-normalisation) | — | — | — | [그래디언트 클리핑](#t-grad-clip), [최고 체크포인트 선택](#t-best-checkpoint), [공통 추론 계약](#t-agent-contract) |

`r*` 4종은 대응하는 원본 모델(`fdon`/`gfdon`/`jfdon`/`pkfdon`)과 **[소등 프레임
고정](#t-retouched) 한 가지만** 다르다. 그것이 이 계열이 존재하는 이유이므로 표에서도
차이만 적었다.

---

## 모델별 Input / Output

I/O는 **두 층**으로 봐야 한다. 아래 두 표가 그 두 층이다.

- **네트워크 층** — `nn.Module.forward`가 실제로 받는 것. 모델마다 다르다.
- **agent 층** — [공통 추론 계약](#t-agent-contract). 21개 모델이 **전부 동일**하고,
  downstream 스크립트(`visualize.py`, `scanline.py`, `benchmark.py`)가 아는 유일한 면이다.

모든 값은 SI(미터, 초, Kelvin, 와트)다. `.npy` 파일은 밀리미터로 저장되어 있고 적재
시점에 변환된다.

### 1. 네트워크 층 (`forward`)

| 모델 | 입력 | 출력 |
|---|---|---|
| `mlp` | `[B, 5]` — `(P, t, z, y, x)` | `[B, 1]` K |
| `gmlp` | `[B, 5]` — `(P, t, z, y, x)` | `[B, 1]` K |
| `pimlp` | `laser_power [B, 1]` + `coords [B, 4]` — `(x, y, z, t)` | `[B, 1]` K, 그리고 `derivatives()` → `FieldDerivatives` |
| `cmlp` | `[B, 4]` — `(t, z, y, x)` | `[B, 1]` K |
| `cgmlp` | `[B, 4]` — `(t, z, y, x)` | `[B, 1]` K |
| `cjmlp` | `[B, 4]` — `(t, z, y, x)` | `[B, 1]` K |
| `cpimlp` | `coords [B, 4]` — `(x, y, z, t)` | `[B, 1]` K, 그리고 `derivatives()` → `FieldDerivatives` |
| `cpkmlp` | `[B, 4]` — `(t, z, dy, dx)`, 빔 기준 오프셋 | `[B, 1]` K |
| `fmlp` | `[B, 2]` — `(P, t)` | `forward`: `[B, n_out]` 정규화 계수 · `field()`: `[B, nz, ny, nx]` K |
| `don` `gdon` `jdon` | `branch [B, 193]` + `coords [B, 4]` `(x, y, z, t)` + `beam [B, 5]` | `[B, 1]` K |
| `pkdon` | `branch [B, 193]` + `coords [B, 4]` `(along, across, z, t)` + `beam=None` | `[B, 1]` K |
| `fdon` `gfdon` `jfdon` | `branch [B, 193]` + `xyz [B, 3]` `(x, y, z)` + `beam [Nt, 5]` 또는 `[C, Nt, 5]` | `[B, Nt]` K |
| `pkfdon` | `branch [B, 193]` + `xyz [B, 3]` `(along, across, z)` + `beam=None` | `[B, Nt]` K |
| `rfdon` `rgfdon` `rjfdon` | `fdon` 계열과 동일 | `fdon` 계열과 동일 |
| `rpkfdon` | `pkfdon`과 동일 | `pkfdon`과 동일 |

각 항목의 뜻:

- **컬럼 순서가 계열마다 다르다.** 밀집 스택은 `(P, t, z, y, x)` — 데이터셋이 그 순서로
  넘긴다. PINN 모델만 `(x, y, z, t)`를 **별도 텐서**로 받는데, 잔차가 그 네 좌표에 대해
  쓰여 있고 autograd가 정확히 그 네 열짜리 텐서를 미분해야 하기 때문이다. `pimlp`가
  power를 따로 떼는 것도 같은 이유(autograd에서 떼어놓기)이고, `cpimlp`는 뗄 power가
  없어서 `coords` 하나만 받는다. 내부에서는 둘 다 `(t, z, y, x)`로 재조립해 스택에 넣는다.
- **`branch [B, 193]`** = `(x_l, y_l, lit)` × 64 sensor + `P` 한 칸.
  [툴패스 branch](#t-toolpath-branch) 참고. `--sensors`를 바꾸면 `3m + 1`로 따라간다.
- **`beam [B, 5]`** = `(x_l, y_l, lit, v_x, v_y)` — 그 행의 시각에 그 행의 경로 위에서의
  빔 상태. 게이트가 이것으로 자기 위치와 진행 방향을 잡는다. 게이트 없는 모델
  (`don`, `fdon`, `pkdon`, `pkfdon`)은 받고 쓰지 않는다. [시퀀스 모델](#t-sequence)에서는
  점이 아니라 **런**의 속성이라 `[Nt, 5]` 표 하나로 오고, 배치가 조건별 블록으로 쌓여
  있으면 `[C, Nt, 5]`가 된다.
- **`FieldDerivatives`** = `T, T_x, T_y, T_z, T_t, T_xx, T_yy, T_zz` 각각 `[B, 1]`.
  2차 항은 `second_order=False`로 부르면 `None`이다(1차만 필요한 Neumann 경계항용).
- **`Nt = 27`** — 툴패스 코퍼스의 공통 시각 격자(0.4 s 간격, 0 ~ 10.4 s). 짧은 런은 그
  접두사이고 나머지는 [마스크](#t-masked-loss)가 지운다.
- **`n_out`** (`fmlp`) = `2 * (2*kx + 1) * (ky + 1) * nz` 실수. 복소 계수의 실부·허부를
  펼친 것이다. `--energy-target 0.9999` 기본에서 `kx = 19, ky = 5, nz = 25` → **11 700**.
  [`--detrend`](#t-detrend)를 켜면 `ny * nz = 1 025`개가 더 붙는다.
- **`(dy, dx)`** (`cpkmlp`) — 빔에서 잰 오프셋. 스택 자체는 `cmlp`의 것이라 컬럼이 무엇을
  *뜻하는지* 들여다보지 않는다. 바뀌는 것은 코퍼스와 그에 따르는 정규화뿐이다.

### 2. agent 층 (공통 계약, 21개 모델 전부 동일)

| 메서드 | 입력 | 출력 |
|---|---|---|
| `predict_at` | `[B, 5]` — `(x, y, z, t, P)` | `[B, 1]` K |
| `predict_of` | `[B, 2]` — `(t, P)` | `[B, 1, D, H, W]` K |

`(D, H, W)`는 `Conv3d` 관례로 `D = z, H = y, W = x`다. 직선 패스 코퍼스는
`(25, 41, 161)`, 툴패스 코퍼스는 `(25, 81, 81)`.

네트워크 층과 agent 층 사이에서 각 agent가 하는 변환:

| 모델 | agent가 추가로 하는 일 |
|---|---|
| `mlp` `gmlp` | 열 순서 바꾸기 `(x,y,z,t,P) -> (P,t,z,y,x)` |
| `pimlp` `cpimlp` | `P` 열을 떼어내고 `(x,y,z,t)`를 따로 넘기기 |
| `cmlp` `cgmlp` `cjmlp` | `P` 열 **버리기** ([P 제거](#t-p-blind)) |
| `cpkmlp` | `P` 버리고, `(x, y)`를 빔 기준 오프셋으로 |
| `fmlp` | `(t, P)`로 부피를 만든 뒤 격자에서 보간 — **역방향**(`predict_of`가 원시 연산) |
| `don` `gdon` `jdon` | 고정된 툴패스로 `branch`를 만들고, 질의 시각의 `beam` 상태를 계산 |
| `pkdon` | 위 + 평판 좌표를 [빔 진행 프레임](#t-beam-frame)으로 회전 |
| `fdon` `gfdon` `jfdon` | 히스토리 `[B, 27]`을 받아 질의 `t`에서 **선형 보간** |
| `pkfdon` `rpkfdon` | 위 + 빔 프레임 회전 |
| `r*` 4종 | 위와 같되 `beam` 상태가 [소등 구간에 고정](#t-retouched)된 것 |
| 페이스트 agent | 행마다 베이스/패치 중 하나를 골라 **한 번만** 평가 ([패치 페이스트](#t-paste)) |

계약이 `P`를 못 쓰는 모델에게도 `P`를 넘기는 것, 툴패스 모델의 "여섯 번째 입력"인 경로가
시그니처가 아니라 **agent 생성 시점**에 고정되는 것은 [공통 추론 계약](#t-agent-contract)
절에 이유를 적었다.

### 3. 학습 시 한 배치의 구성

`forward`의 입력과 다르다 — 라벨과 마스크가 함께 오고, 코퍼스 종류마다 모양이 다르다.

| 코퍼스 | 배치 | 해당 모델 |
|---|---|---|
| `SimulationDataset` + 모델별 `*Dataset.batch(n)` | `(inputs [B, 4 or 5], T [B, 1])` | 밀집 스택 계열 |
| PINN 6배치 | 라벨 데이터 + 내부 collocation + 아랫면/윗면/측면 + `t = 0`, 각각 `PointSet(laser_power, coords, temperature?, normal?, q_laser?)` | `pimlp`, `cpimlp` |
| `FourierCorpus` | `((P, t) [B, 2], 계수 [B, n_out])` — 전체가 **56쌍**(7 power × 8 스냅샷) | `fmlp` |
| `PathCorpus` | `(branch, coords [B,4], beam [B,5], T [B,1])` | `don` `gdon` `jdon` |
| `SequenceCorpus` | `(branch, xyz [B,3], beam [C,Nt,5], target [B,Nt], mask [B,Nt])` | `fdon` `gfdon` `jfdon` + `r*` 3종 |
| `PatchCorpus` | `(branch, coords [B,4] 빔 프레임, T [B,1])` | `pkdon` |
| `SequencePatchCorpus` | `(branch, coords [B,3] 빔 프레임, None, target [B,Nt], mask [B,Nt])` | `pkfdon`, `rpkfdon` |

배치 크기는 [조건별로 균등하게](#t-equal-condition-batch) 쪼개진다 — `--batch-data`
(기본 4096)를 조건 수의 배수로 반올림한다. 툴패스 코퍼스의 조건 하나는 81×81×25 =
**164 025** 점이고, [패치 격자](#t-lattice-interp)는 25 × 21 × 21 = **11 025** 오프셋이다.

---

# 기술 설명

## 아키텍처

<a id="t-dense"></a>
### 밀집 스택 (Dense MLP stack)

좌표 다섯 개(또는 네 개)를 한꺼번에 받아 은닉층을 통과시켜 온도 하나를 내는 가장
단순한 구조. `(P, t, z, y, x) -> T`가 기본형이고, [P를 뺀 모델](#t-p-blind)은
`(t, z, y, x) -> T`다.

이 구조가 존재하는 이유는 성능이 아니라 **바닥**이다. 구조화된 모델이 무엇을 얻었다고
말하려면 아무것도 넣지 않은 것이 먼저 있어야 한다. `mlp`가 그 바닥이고,
`gmlp`/`pimlp`는 각각 게이트 하나·물리 하나만 더한 것이다.

`cpkmlp`는 이 저장소에서 유일하게 다른 모델의 클래스를 **복사하지 않고 import**한다
(`models/cmlp/model.py`의 `ControlMLP`). 나머지는 한 모델의 실험이 다른 모델을 몰래
움직이는 일을 막으려고 일부러 복사해 두었지만, `cpkmlp`는 "`cmlp`를 용융풀로 제한한
것"이 정의 자체라서 두 스택이 갈라지면 비교가 무의미해진다.

> 구현: `models/mlp/model.py`, `models/cmlp/model.py`, `models/cpkmlp/model.py`

<a id="t-deeponet"></a>
### DeepONet (branch / trunk 내적)

함수를 함수로 보내는 연산자를 배우는 구조. **branch**가 입력 함수를, **trunk**가
질의 좌표를 각각 잠재 벡터로 보내고, 둘의 내적이 답이 된다.

```
T_hat = T_amb + dT * ( <branch(u), trunk(x, y, z, t)> * G + c )
```

좌표를 다섯 번째 축으로 취급하는 대신 공정 조건을 *함수*로 분리하는 것이 요점이다.
`c`는 게이트 **바깥**에 더해지는 편향이라 게이트가 죽은 곳에서도 살아남는다.

기본 크기는 은닉 128×4, 잠재 차원 128이다.

> 구현: `models/_pathdon.py` — `PathDeepONet`

<a id="t-toolpath-branch"></a>
### 툴패스 branch (toolpath-conditioned branch)

`mlp` 계열의 branch가 본 것은 레이저 **출력값 하나**뿐이다. 직선 1패스가 각 모델의
`laser.py`에 하드코딩되어 있어서, 스캔 경로가 바뀌면 전부 틀린다.

이 계열은 **경로 자체**를 branch에 넣는다. 궤적 `u(tau) = (x_l, y_l, lit)`를
코퍼스 전체가 공유하는 고정 시각(기본 64개 sensor)에서 읽어 `3 x 64 + 1 = 193`차원
벡터로 만들고, 마지막 한 칸에 `P`를 붙인다. 학습된 연산자는

```
G: (toolpath, P)  ->  (T: (x, y, z, t) -> K)
```

이므로 한 번 학습한 네트워크가 serpentine에도 spiral에도 답한다.

sensor 시각은 각 경로의 길이로 정규화하지 **않고** 절대 시간을 쓴다. 네 패턴이
9.20 s ~ 10.78 s로 서로 다른데, 정규화하면 그 차이가 지워지기 때문이다. 자기 경로가
끝난 뒤는 마지막 노드에 `lit = 0`으로 고정되므로 "이 스캔은 저것보다 1초 먼저 끝났다"를
branch가 읽어낼 수 있다.

> 구현: `models/_pathdon.py` — `branch_vector`, `beam_state`

<a id="t-sequence"></a>
### 시퀀스 출력 (`[B, Nt]` 히스토리)

trunk에서 `t`를 빼고 출력에 붙인다. trunk는 `(x, y, z)`만 받고, 답은 그 점의 **온도
히스토리 전체** `[B, Nt]`다. branch가 이미 경로를 시계열로 읽고 있으므로, 출력의
시계를 입력의 시계와 맞추는 셈이다.

```
T_hat(x, y, z)_j = T_amb + dT * ( <b(u), tau_j(x, y, z)> * G_j + c )
```

얻는 것: 점 하나당 forward가 `Nt`번이 아니라 1번이고, 히스토리가 하나의 잠재에서
나오므로 스스로 모순될 수 없다.

잃는 것: 답이 코퍼스 자신의 격자 위에만 존재한다. `t`에 대해 더 이상 연속이 아니며,
저장 시각 사이는 [agent가 선형 보간](#t-agent-contract)한다. 솔버의 0.4 s 간격 동안
빔은 4 mm 움직이므로 그 사이 값은 진짜로 추측이다.

> 구현: `models/_pathdon.py` — `SequenceDeepONet`, `SequenceAgent`

<a id="t-spectral"></a>
### 스펙트럼 출력 (Fourier 계수 회귀)

`fmlp` 전용. 다른 모든 모델은 점별(pointwise)이라 부피 하나에 `25 x 41 x 161 =
165 025`번의 forward가 든다. 이 모델은

```
(P, t) -> 저장된 모든 Fourier 계수 -> 역변환 한 번 -> 부피 전체
```

숫자 둘을 넣어 부피 하나를 받는다. `predict_of`가 **원시 연산**이고 `predict_at`이
파생 연산인, 저장소에서 유일하게 뒤집힌 모델이다.

네트워크 본체는 은닉 3층뿐이다. 공간 의존성은 이미 기저가 들고 있으므로 남은 것은
"계수가 `P`와 시계에 따라 어떻게 움직이는가"뿐이고, `R^2`에서 오는 사상은 깊이가
아니라 **잘 조건화된 타깃**을 필요로 한다. 출력층이 전체 가중치의 99%를 넘지만,
타깃은 두 파라미터로 매끄럽게 변하는 56개 스냅샷이라 실제로는 최대 56차원 부분공간에
갇혀 있다. 병목은 용량이 아니었다.

> 구현: `models/fmlp/model.py` — `FourierMLP`

<a id="t-activation"></a>
### 활성화 선택 (SiLU / Tanh / ReLU)

세 갈래가 있고, 각각 이유가 다르다.

- **SiLU** — 밀집 스택 전부(`mlp`, `gmlp`, `pimlp`, `c*` 계열). 매끄러워서 PDE 잔차가
  요구하는 2차 미분이 존재한다. `ReLU` 같은 구간 선형 활성화를 쓰면 Laplacian이
  항등적으로 0이 되어버린다. PINN이 아닌 모델도 같은 것을 쓰는데, 그래야
  `mlp`↔`pimlp` 비교에서 활성화가 변수로 끼어들지 않는다.
- **Tanh** — 툴패스 DeepONet 12종 전부. 같은 2차 미분 이유이고, 연산자망에서
  관례적으로 쓰이는 선택이다.
- **ReLU** — `fmlp`만. 여기서는 아무것도 미분되지 않으므로 매끄러움을 요구할 근거가
  없다.

`c*` 계열은 `--activation {silu,tanh,relu,sigmoid}`로 바꿀 수 있게 되어 있다.

> 구현: 각 모델 `model.py`의 `activation` 인자, `models/*/train.py`의 `ACTIVATIONS`

<a id="t-normalisation"></a>
### 정규화 버퍼 (입출력 스케일링)

정규화 상수를 **버퍼로 체크포인트에 실어** 학습 때와 추론 때가 같은 스케일임을
구조적으로 보장한다. 호출자는 항상 SI 단위를 넘기고, 정규화는 `forward` 안에서 일어난다.
게이트는 물리 좌표에서 계산되어야 하므로 이 순서가 필수다.

- 좌표: 도메인 중심·반폭으로 `[-1, 1]`
- branch: 빔 위치는 평판 크기로, `lit`는 이미 0/1, `P`는 코퍼스 최대 출력으로 —
  코퍼스 통계를 따로 돌 필요 없이 전부 O(1)
- 온도: `T = T_amb + dT * (출력)`, `dT`는 코퍼스의 최대 상승폭

[패치 모델](#t-patch-corpus)에는 함정이 하나 있다. trunk의 도메인은 빔 주변 몇 mm인데
branch가 읽는 빔 위치는 여전히 **평판** 좌표다. 20 mm 좌표를 2.5 mm 반폭으로 나누면
입력이 8 정도가 되어 정규화의 의미가 사라지므로, branch 쪽은 trunk가 무엇을 재고 있든
평판 크기로 스케일한다.

> 구현: `models/_pathdon.py` — `run()`의 `branch_mean`/`branch_scale`, 각 모델의
> `register_buffer`

---

## 물리 prior — 게이트

게이트는 **예측이 아니라 곱해지는 인자**다. 네트워크는 `T / G`를 만들어내야 하므로,
좋은 `G`가 갖춰야 할 성질은 점별 정확도가 아니라 *매끄럽고 대충 맞는 것*이다. 이
구분이 아래 [소프트 코어](#t-soft-core) 절의 반직관적인 결과를 만든다.

<a id="t-gauss-gate"></a>
### 가우시안 게이트 (`G = g + p`)

움직이는 빔 위에 올라탄 단위 피크 가우시안을 네트워크 출력에 곱한다.

```
T_hat = T_amb + dT * ( net(...) * G + b )
G     = g(x, y, z, t) + p
```

`g`는 빔 중심에서 1이고 멀어지면 0으로 죽는다. 직선 패스 모델(`gmlp`, `cgmlp`)에서는
빔 중심이 `x_l(t) = x_0 + v t`로 하드코딩되어 있고, 툴패스 모델(`gdon`, `gfdon`,
`rgfdon`)에서는 그 행 자신의 경로에서 온 `beam_state`가 위치를 준다.

`lit`이 곱해져 있어서 **소등 구간에는 게이트가 통째로 0**이 된다. 가우시안은 *열원*의
그림이고, 열원이 없는 곳에는 그릴 것이 없기 때문이다. 그때 남은 열은 게이트가 설명하지
못하는 자취이고, 그것을 덮는 것이 [바닥값 `p`](#t-gate-offset)의 역할이다.

[시퀀스 모델](#t-sequence)에서는 게이트도 히스토리여야 한다 — 저장 시각마다 빔이
움직였으므로 `sequence()`가 `[B, Nt]`를 만든다.

`--gaussian-exponent-scale`로 폭을 좁히거나(>1) 넓힐(<1) 수 있고, 중심 진폭은 `exp(0) = 1`
이라 바뀌지 않는다.

> 구현: `models/_pathdon.py` — `GaussianGate`; `models/gmlp/laser.py`,
> `models/cgmlp/laser.py`

<a id="t-rosenthal"></a>
### Rosenthal 이동 열원 게이트 (`G = j + p`)

가우시안은 레이저가 **넣는 것**의 모양이지 금속이 그것으로 **하는 일**의 모양이 아니다.
대칭이고, 자취가 없고, 평판에 바닥이 있다는 것도 빔이 움직인다는 것도 모른다. 이
게이트는 두 번째 것으로 바꾼다.

빔과 함께 움직이는 좌표계 `xi = x - x_l(t)`에서, 속도 `v`로 움직이는 점 열원이 남기는
정상 해가 Rosenthal 해다.

```
T - T_inf = Q / (2 pi k) * exp[-v (r + xi) / (2 alpha)] / r
```

지수항이 이것을 *움직이는* 열원으로 만든다. 빔 앞쪽은 `r ~ +xi`라 `alpha/v` 안에서
장이 죽고, 뒤쪽은 `r ~ -xi`라 `r + xi -> 0`이 되어 지수가 더 이상 감쇠하지 않고 `1/r`
꼬리만 남는다. 그 **비대칭**이 가우시안과의 전부이며, 모든 그림에서 모델들이 고전하는
후행 열자취가 바로 그것이다.

툴패스 모델(`jdon`, `jfdon`, `rjfdon`)에서는 `xi`를 `x`축이 아니라 **빔 자신의 속도
벡터** `(v_x, v_y)/|v|`에 투영한다([빔 진행 프레임](#t-beam-frame)). 그래서 경로가
코너를 돌면 자취도 함께 돈다. `+x`를 가정하는 게이트라면 네 패턴 중 셋에서 자취를
엉뚱한 곳에 놓게 된다.

이 해가 가정하지만 이 문제가 정확히 만족하지 않는 것: 윗면의 대류·복사 없음(단열로
취급), 무한히 넓은 평판(측면 없음), 빔 프레임에서의 정상 상태(시동 과도 없음). 그럼에도
*게이트*로서는 무방하다 — 예측이 아니라 네트워크가 곱하는 prior이기 때문이다.

175 W 단독·64×3 기준으로 게이트 없음 1.689 K, 가우시안 1.532 K, Rosenthal **1.204 K**.

> 구현: `models/cjmlp/laser.py` — `moving_source`; `models/_pathdon.py` —
> `MovingSourceGate`, `_rosenthal_series`

<a id="t-image-source"></a>
### 영상법 (method of images)

Rosenthal 해는 무한 매질의 점 열원 해다. 실제 평판은 두께 `d`에 윗면 단열(빔이 지나는
면), 아랫면 `T_inf` 고정이다. 두 조건 모두 **영상 열원**으로 정확히 만족시킨다:
`z = 2nd`에 부호 `(-1)^n`을 갖는 복제 열원을 놓는다.

- 윗면 양쪽의 쌍은 부호가 같아 `dT/dz`가 그 면에서 0이 된다.
- 아랫면 양쪽의 쌍은 부호가 반대라 `T`가 그 면에서 0이 된다.

`n = 0` 좌우로 3항씩 유지한다. 영상들은 `2d = 12 mm` 간격인데 장은 `alpha/v = 0.24 mm`
스케일로 죽으므로 개별 항은 무시할 만큼 작다 — 그러나 아랫면에서는 진짜 열원도 똑같이
작으므로, 거기서의 상쇄는 둘 다 아무리 작아도 **정확**하다. 그것이 항을 남기는 이유다.

> 구현: `models/cjmlp/laser.py` — `_series`; `models/_pathdon.py` — `_rosenthal_series`

<a id="t-soft-core"></a>
### 소프트 코어 (`sigma` 정칙화)

진짜 점 열원은 `r = 0`에서 발산한다. 게이트는 어디서나 유한해야 하므로 반경을
`sqrt(xi^2 + y^2 + z^2 + sigma^2)`로 부드럽게 만든다. 원점에서 유한하고, 멀리서는 점
열원으로 환원된다.

`sigma`를 고르는 방법이 이 저장소에서 가장 반직관적인 결과다. **게이트가 장처럼 보이게**
최소제곱으로 맞추면 0.45 mm가 나오고, 이는 확산 길이 `2*alpha/v = 0.479 mm`와 7% 안에서
일치한다. 그 게이트는 모든 척도에서 더 나은 *그림*이다 — 자취를 따라가고, 깊이 방향
감쇠가 가우시안보다 다섯 배 정확하고, 최대점이 솔버와 0.04 mm 차이로 맞는다. 그런데
그걸로 학습하면 셋 중 가장 나쁘다.

```
sigma [mm]            0.45    0.90    1.70
shape RMSE, line     0.092   0.181   0.397     <- 낮을수록 좋은 그림
val RMSE [K]         2.149   1.799   1.204     <- 낮을수록 좋은 모델
val max  [K]         192.1    77.7    29.5
```

순위가 뒤집히는 이유는 게이트가 예측이 아니라 인자이기 때문이다. 0.45 mm에서 게이트는
0.25 mm 셀 하나를 지나며 1.00에서 0.55로 떨어지고, 빔 1 mm 앞에서 이미 0.005다 — 장이
아직 피크의 1/3인 지점에서. 네트워크가 떠안는 몫 `T / G`는 출발했던 장보다 **조건이 더
나쁘고**, 가파른 두 함수의 곱은 용융풀을 220 K 넘겨버린다. 그래서 `sigma`는 빔 반경
`BEAM_RADIUS = 1.6971 mm`로 둔다.

> 구현: `models/cjmlp/laser.py` — `SOFT_CORE`와 모듈 docstring

<a id="t-gate-offset"></a>
### 게이트 바닥값 `p` (learnable gate offset)

게이트를 구속복으로 만들지 않는 장치. `G = g` 하나만 쓰면 게이트가 죽은 모든 곳에서
예측이 `T_amb + dT*c`에 못 박히는데, 이 경로들에서는 그게 대부분의 시간 대부분의
평판이고, 빔이 이미 지나간 자취 전체가 거기 포함된다. 확산된 장과 후행 자취가 맞추기
어려운 정도가 아니라 아예 **표현 불가능**해진다.

`p`는 학습되는 스칼라 하나다. `p = 0`이면 순수 게이트, `p`가 커질수록 게이트 없는
모델에 접근한다. 그래서 학습된 `p`는 "이 fit이 prior에서 얼마나 물러서야 했는가"를
직접 읽어주는 값이 된다. 기본 초기값은 0.5(`--gate-offset`).

레거시 `gdon_old`에서의 증거: 맨 `g` 게이트만 쓴 런은 27.298 K였고, 빔 피크는 잘
맞췄지만 가우시안이 죽은 곳은 전부 ambient로 예측해 평판 대부분에서 −348 K의 오차를
냈다. `p`를 더하자 2.950 K가 되었다.

`p`는 공간·시간에 대해 상수이므로 게이트를 들어올리기만 하고 자기 미분을 만들지 않는다.
PINN과 결합해도 PDE 잔차를 건드리지 않는다.

> 구현: `models/_pathdon.py` — `PathDeepONet.gate_offset`

<a id="t-unit-peak"></a>
### 단위 피크 정규화

게이트를 최대값 1로 정규화한다. Rosenthal 급수의 최대는 닫힌 형태가 없고(최대점이 빔
**뒤쪽**에 있으며 모든 상수에 동시에 의존한다) CPU에서 float64로 중심선을 한 번 훑어
찾는다. 이 코퍼스의 모든 점등 구간은 같은 속도로 움직이므로 한 번의 스윕이 런 전체에
쓰이고, 소등 구간은 `lit = 0`이라 애초에 여기 도달하지 않는다.

정규화의 의미는 두 가지다. 진폭(`Q`, `k`)은 네트워크와 `dT` 스케일이 이미 들고 있으므로
게이트에서 살아남는 것은 **모양**뿐이고, `G = j + p`와 `G = g + p`가 같은 스케일 위에
놓여야 두 모델이 모양만으로 갈린다.

가우시안의 피크는 정의상 빔 **위**에 있고 Rosenthal의 피크는 조금 뒤에 있는데, 그
지연은 넣은 것이 아니라 지수항에서 나온다.

> 구현: `models/cjmlp/laser.py` — `_peak`; `models/_pathdon.py` — `MovingSourceGate._peak`

<a id="t-beam-frame"></a>
### 빔 진행 프레임 (along / across)

평판 좌표를 빔의 진행 방향 성분 `along`(뒤쪽이 음수, 자취가 있는 쪽)과 면내 수직 성분
`across`로 바꾼다.

두 군데에 쓰인다.

1. **게이트 안에서** — `jdon`/`jfdon`/`rjfdon`의 Rosenthal `xi`를 속도 벡터에 투영한다.
   경로가 코너를 돌면 자취도 돈다.
2. **코퍼스 좌표로서** — [패치 모델](#t-patch-corpus)의 trunk가 아예 이 프레임에서
   질문받는다. 준정상 용융풀은 이 좌표에서 경로가 무엇을 하든 **정지해 있고**, 그래서
   네트워크가 배울 수 있는 함수가 된다.

`cpkmlp`는 회전하지 않는데, 그것이 맞다 — 직선 패스라 `dx`가 곧 진행축이다. 툴패스
계열은 빔이 코너를 돌기 때문에 회전 없는 창은 스냅샷마다 다른 각도로 자취를 담게 되고,
네트워크가 시계로부터 진행 방향을 역추론해야 한다. 프레임은 바로 그것을 없애려고 있다.

> 구현: `models/_pathdon.py` — `beam_frame`, `PatchAgent.frame`

<a id="t-beam-anchor"></a>
### 빔 앵커 (peak가 아니라 beam)

패치 창을 어디에 놓을지의 문제. 당연한 답은 "가장 뜨거운 노드"이고 실제로 그것이 처음
시도였는데, 격자와 만나면 무너진다.

솔버의 최대점은 빔을 0.24 mm 뒤따르고, 그래서 두 노드의 **중점에서 0.003 mm 지난**
자리에 떨어진다. 그 두 노드의 온도 차는 1353 K 상승 중 0.46 K다. 게다가 빔은 저장
스텝마다 정확히 16노드씩 전진하므로 이 아슬아슬한 동률이 평균화되지 않고 **매 순간
똑같이** 반복된다. `models/`의 모든 모델이 연속 최대점을 솔버의 0.06 mm 안(노드의 1/5)에
놓는데도 전부 그 중점의 반대편에 떨어지고, 결국 argmax는 **항상** 한 칸 뒤 노드를 고른다.
패치는 0.25 mm 어긋나 붙고, 빔 반경에 걸쳐 1350 K 떨어지는 사면에서 그 대가는 수백
Kelvin — 고치려던 fit 오차보다 훨씬 크다.

argmax는 두 노드의 *차이*를 읽는데, 여기서 그 차이는 좋은 fit이 어느 한쪽에 내는 오차보다
30배 작다.

빔은 그런 문제가 없다. 장에서 추정되는 값이 아니라 보정된 숫자 몇 개와 시계이므로
(`x_l(t) = x_0 + v t`, 툴패스 계열은 `Toolpath.position`), 패치를 *학습한* 프레임과
*붙이는* 프레임이 노드 오차 이내가 아니라 **정확히 같은 프레임**이다.

> 구현: `models/cpkmlp/dataset.py` docstring, `models/cpkmlp/laser.py` — `BeamPath`

---

## 목적함수

<a id="t-scaled-mse"></a>
### dT 정규화 MSE

```
mean( ((T_hat - T) / dT)^2 )
```

특성 온도 상승 `dT`(코퍼스 최대 온도 − ambient)로 나눠 손실을 무차원으로 만든다.
모든 데이터 구동 모델이 같은 스케일을 쓰므로 한 모델의 손실값을 다른 모델의 것과
견줄 수 있다.

> 구현: `models/_pathdon.py` — `ScaledMSELoss`, 각 모델 `loss.py`

<a id="t-masked-loss"></a>
### 마스크 손실 (ragged clock / 평판 밖)

[시퀀스 모델](#t-sequence) 전용. 출력은 `Nt = 27`로 고정인데 런마다 스냅샷 수가 다르다
— `spiral`은 24개(9.20 s), `raster`는 27개(10.78 s). 짧은 런은 공통 격자의 **접두사**이고,
자기 답이 없는 시각은 마스크가 손실에서 지운다.

0으로 패딩하고 그대로 채점하면 "`raster`는 10.4 s에 차갑고 `spiral`은 거기서 298 K"라고
가르치게 되는데, 둘 다 사실이 아니다.

마스크는 두 가지를 더 담는다.

- [창 양도](#t-cede)로 비운 영역 — 검증 채점도 같은 마스크로 하므로, 학습하지 않은
  영역의 정확도로 체크포인트를 고르는 일이 없다.
- [격자 보간](#t-lattice-interp) 격자점이 평판 밖으로 나간 경우 — 창의 모서리는
  `radius * sqrt(2) = 3.54 mm`까지 닿는데 스캔은 가장자리에서 3 mm만 안쪽이다.

> 구현: `models/_pathdon.py` — `MaskedScaledMSELoss`, `SequenceCorpus._mask`

<a id="t-pinn"></a>
### PINN 잔차 (열방정식 + 경계 + 초기 조건)

라벨과 **함께** 물리를 맞춘다.

```
L = w_D·L_data + w_PDE·L_cond + w_BC·(L_bottom + L_top + L_surr) + w_IC·L_init
```

- `L_cond` — `rho*c_p*dT/dt - k*(T_xx + T_yy + T_zz)`, 내부 collocation 점에서
- `L_bottom` — 아랫면 `T = T_amb` (Dirichlet)
- `L_top` — `k*dT/dn = q_laser - h(T - T_amb) - sigma*eps*(T^4 - T_amb^4)`.
  `q_laser`는 윗면의 가우시안 이동 열원 `2*A*P/(pi*r_b^2) * exp(...)`
- `L_surr` — 측면, 열원 없이 대류·복사만
- `L_init` — 도메인의 가장 이른 시각. 타깃은 `T_amb`가 아니라 **솔버가 거기서 가졌던
  장**이다. [`--exclude`](#t-exclude)가 시작 시각을 이미 가열된 장으로 옮기면 ambient는
  데이터 항이 같은 점에서 부정하는 조건이 되어버린다.

잔차는 네트워크의 모양을 신경 쓰지 않으므로 `pimlp`와 (레거시) `pidon_old`가 **정확히
같은** 목적함수를 쓴다. 그래서 `mlp`↔`pimlp`는 물리를, `pimlp`↔`pidon_old`는
아키텍처를 각각 분리해낸다.

기본 가중치는 `w_D = 1`, 나머지 `1e-4`.

> 구현: `models/pimlp/loss.py`, `models/cpimlp/loss.py` — `PINNLoss`

<a id="t-residual-scaling"></a>
### 잔차 무차원화

이것이 없으면 손실 항들을 더할 수 없다. Dirichlet 잔차는 K, PDE 잔차는 W/m³, Neumann
잔차는 W/m² 단위라, 세 경계 항을 한 가중치로 묶으면 flux 항이 수십 자릿수 차이로
지배한다.

각 잔차를 자기 특성값으로 나눠 전부 O(1)로 만든다.

- 온도: `dT`
- PDE: `rho * c_p * dT / t_scale`
- flux: 피크 레이저 강도

그러고 나서야 `LossWeights`의 숫자가 진짜 상대적 중요도를 뜻한다.

> 구현: `models/pimlp/loss.py` — `ResidualScales.characteristic`

<a id="t-resampled-collocation"></a>
### collocation 재샘플링 (epoch가 아니라 iteration)

PINN 런은 스텝마다 배치를 **여섯 개** 만든다: 라벨 데이터, 내부 collocation, 아랫면/윗면/
측면, 그리고 `t = 0`. 라벨 없는 다섯은 매 반복 새로 뽑는다. 데이터를 한 바퀴 돈다는
개념이 없어서 이 모델들이 `--epochs`가 아니라 `--iterations`로 세는 이유다.

이 때문에 L-BFGS와는 궁합이 나쁘다. L-BFGS는 연속한 그래디언트에서 곡률을 추정하는데
그것이 의미를 가지려면 연속한 스텝이 **같은 함수**에서 나와야 한다. 그래서 L-BFGS를
고르면 PINN 모델은 자기 점 집합들을 얼려버린다.

> 구현: `models/pimlp/train.py`, `models/cpimlp/train.py`

<a id="t-physics-power"></a>
### 물리항 power 고정 (`--physics-power`)

`cpimlp`만. [P를 입력에서 뺐지만](#t-p-blind) 물리에서까지 뺄 수는 없다 — 윗면 경계
조건에 레이저가 들어 있고 그 flux는 출력에 비례하며, 열원 없는 열방정식은 이 문제가
아니다. 그래서 power는 네트워크를 떠나되 물리에는 남고, "그럼 물리는 어느 power에 대한
것인가"가 새 질문이 된다.

배치마다 샘플링하지 **않고** 단일 값(기본: 코퍼스 평균 175 W)에 고정한다. 그래야 잔차가
하나의 명확하고 자기 일관된 문제를 서술한다. 매 스텝 다른 열 flux를 같은 점에서
요구하되 그 이유를 네트워크는 관측할 수 없는 상황을 피하는 것이다. 값은 체크포인트에
버퍼로 저장되어, 그 체크포인트가 어떤 물리에 맞춰졌는지가 가중치와 함께 이동한다.

7-power 코퍼스에서는 데이터 항(전 power 평균)과 물리 항(175 W)이 찢어질 것처럼 보이지만
실제로는 거의 그렇지 않다. `T`가 `P`에 대해 거의 선형이라 power 평균 장과 175 W 장은
RMSE 0.1 K 차이다. 175 W 단독 코퍼스에서는 긴장이 **정확히 0**이다.

> 구현: `models/cpimlp/train.py`

<a id="t-parseval"></a>
### Parseval 계수 MSE

`fmlp`의 손실은 격자가 아니라 스펙트럼 공간에서 계산된다. 기본값 `--norm global`에서
이것은 타협이 아니라 **같은 양**이다. 변환이 정규직교(`norm="ortho"`)이므로 Parseval
정리에 의해

```
|T_hat - T|^2 를 kappa에 대해 합한 것  =  |T_hat - T|^2 를 격자에 대해 합한 것
```

(버려진 모드는 제외 — 어떤 손실을 써도 되찾을 수 없다.) 모든 계수에 **하나의 스케일**을
쓰는 것이 이 항등식을 보존하고, 그래서 이 손실을 줄이는 것이 Kelvin 단위 L2 오차를
줄이는 것과 같다.

`--norm per-coef`는 이 항등식을 일부러 깬다. 계수마다 자기 분산으로 표준화하면 에너지의
백만분의 일을 나르는 모드가 DC 항과 같은 무게를 갖는다. *스펙트럼*이 관심 대상이면 그게
맞는 목적함수이고 *장*이 관심 대상이면 틀린 목적함수라, 기본값이 아니라 플래그다.

> 구현: `models/fmlp/loss.py` — `CoefficientMSELoss`

---

## 코퍼스 · 샘플링

<a id="t-p-blind"></a>
### P 제거 (P-blind control)

`c` 접두 모델 전부. 레이저 출력을 **네트워크 입력에서만** 빼고 나머지는 전부 그대로
둔다 — 같은 스택, 같은 활성화, 같은 목적함수. 그래서 짝지어진 두 모델의 차이는 `P`
하나다.

7-power 코퍼스에서는 비용이 크다. 같은 `(x, y, z, t)`가 power마다 다른 온도를 나르는데
`P`를 못 보는 네트워크는 어느 쪽을 묻는지 구분할 수 없으므로, 할 수 있는 최선이
power 평균 장이다. 그 평균은 넘을 수 없는 바닥이다.

```
RMSE 바닥 = 12.821 K   # 전체 코퍼스에서 어떤 P-blind 모델도 넘지 못하는 값
```

현재 `checkpoints/{cmlp,cgmlp,cpimlp}/`의 체크포인트는 그 실험이 **아니다**. 175 W
단독으로 학습·검증했으므로 분할 안에서 `P`가 상수이고 애초에 잃을 정보가 없다. 이
셋이 지금 답하는 질문은 "P-blindness가 바닥 대비 얼마나 비싼가"가 아니라 "변하지 않는
입력 하나를 빼는 것이 조금이라도 비용인가"이다.

계약 상으로는 `predict_at`이 여전히 `P` 열을 받는다(받고 버린다). 그래서 downstream
스크립트에 그대로 꽂히고, 100 W와 250 W를 같은 점에서 물으면 비트 단위로 동일한 답이
나온다 — 구조적으로.

> 구현: `models/cmlp/`, `models/cgmlp/`, `models/cjmlp/`, `models/cpimlp/`,
> `models/cpkmlp/`

<a id="t-holdout-power"></a>
### power hold-out

일반화를 **`P`를 가로질러** 재기 위한 분할. 이미 본 power의 다른 점들에 대한 일반화가
아니다.

- 직선 패스 계열: 100…250 W를 25 W 간격으로 7개 학습, 160 W는 `data/valid`에 통째로
  격리
- 툴패스 계열: `--holdout`(기본 175 W)에 해당하는 조건들을 별도 집합으로 두고 **한 번도
  샘플링하지 않으며** 채점만 한다

체크포인트 선택은 항상 in-distribution 검증 분할로 한다. hold-out은 보고되는 숫자
자체이므로, 그것으로 체크포인트를 고르면 N번 뽑기의 최선을 한 번의 결과인 것처럼
보고하는 셈이 된다.

> 구현: `models/_pathdon.py` — `PathCorpus.__init__`, `run()`의 `best.update`

<a id="t-exclude"></a>
### 초기 시점 제외 (`--exclude N`)

학습 전에 앞쪽 `N`개 스냅샷을 버린다. `--exclude 1`은 `t = 0`(평판이 아직 균일하게
ambient인 시점)을, `--exclude 2`는 `t = 0`과 `0.4 s`를 지운다.

대가는 명확하다. `ex1` 런에게 `t = 0`을 묻는 것은 학습 도메인 밖의 외삽이고, 그래서
이 런들의 전체 RMSE는 `t >= 0.8` RMSE보다 한 자릿수 나쁘다. `RESULTS.md`가 `t >= 0.8`
열을 비교 기준으로 삼는 이유다.

> 구현: 각 모델 `train.py`의 `--exclude`

<a id="t-patch-corpus"></a>
### 패치 코퍼스 (용융풀 창)

평판 전체를 맞추는 모델은 용융풀을 특정한 방식으로 틀린다. 풀은 40 mm 블록(또는 20 mm
정사각) 중 몇 mm라서 평균제곱오차에 거의 기여하지 않고, fit은 봉우리를 뭉갠다.

이 모델들은 **그 근방만** 받는다. 각 스냅샷에서 빔으로부터 `radius`(기본 2.5 mm) 안의
노드만 남기고 — `z`는 전 깊이, 풀이 표면에서 아래로 뻗고 빔에는 중심을 잡을 깊이가
없으므로 — 위치를 [빔 프레임](#t-beam-frame) `(along, across)`로 다시 쓴다.

창이 *그 프레임에서* 정사각이라는 점이 요점이다. 패치가 **학습되는** 영역과 **붙는**
영역이 빔이 어느 방향을 향하든 같은 정사각이 된다.

행을 미리 물질화한다([인덱스 디코딩](#t-index-decoded)과 반대). 2.5 mm에서 7 M 행이지
115 M 행이 아니고, 어느 노드가 창에 들어오는지가 스냅샷마다 바뀌므로 이용할 인덱스
산술이 없다.

소등 스냅샷도 **버리지 않는다**. 그때 빔 자리에 풀은 없지만(`raster`는 런의 1/5),
페이스트는 자기가 덮는 모든 순간에 답이 필요하고, 정직한 답이 "식어가는 평판"인
경우를 맞추기를 거부하면 패치가 그 구간으로 외삽하게 된다.

패치는 그 자체로 대리모델이 아니다. 풀의 모양을 알고 평판의 나머지는 모른다.

> 구현: `models/_pathdon.py` — `PatchCorpus`, `SequencePatchCorpus`;
> `models/cpkmlp/dataset.py`

<a id="t-lattice-interp"></a>
### 격자 보간 (빔 프레임 lattice)

[패치 코퍼스](#t-patch-corpus)는 스냅샷마다 창에 들어오는 *노드*를 남기므로 오프셋
집합이 순간마다 조금씩 다르다. 행별로 채점할 때는 문제없지만 [시퀀스](#t-sequence)는
안 된다 — "빔 뒤 1 mm의 온도가 스캔 동안 어떻게 변하는가"를 물으려면 매 시각 **같은**
오프셋이어야 하는데, 그런 노드는 없다.

그래서 격자를 빔 프레임에서 정의하고(격자 간격의 `(along, across)`, `|.| <= radius`,
`z`는 평판 자신의 깊이) 솔버 장을 그 위로 **보간**한다. 이 저장소에서 타깃이 원시 솔버
노드가 아닌 유일한 곳이라, 대가를 정확히 적어둘 값어치가 있다: 0.25 mm 격자에서 이중선형
보간은 최악 11 K, RMS 0.2 K다(장을 0.5 mm로 성기게 만든 뒤 건너뛴 노드를 복원해 측정).
용융풀 안에서는 5 K에 가깝다. 오차가 수백 Kelvin인 패치에 대해 그것은 잡음이고,
**학습 타깃에만** 닿는다 — 페이스트 결과는 여전히 원시 노드로 채점된다.

보간은 `(x, y)`에 대해서만 이중선형이고 `z`는 정확하다. 격자가 평판 자신의 깊이를
쓰므로 그 축에는 보간할 것이 없다.

> 구현: `models/_pathdon.py` — `SequencePatchCorpus._sample`

<a id="t-cede"></a>
### 창 양도 (`--cede-from` / `--cede-radius`)

패치가 덮어쓸 창을 베이스 모델의 코퍼스에서 **미리 빼는** 기법. 그러지 않으면 베이스는
`PeakCorrectedAgent`가 곧바로 버릴 답에 용량을 쓴다. 툴패스 코퍼스에서 그 창은 샘플의
6.3%이면서 베이스 제곱오차의 **54%**를 나른다. 양도한 결과 페이스트 성능이
7.49 K → 6.39 K로 개선됐다.

대가는 이음매다. 베이스가 창 가장자리에서 연속이어야 할 장을 더 이상 보지 못하므로,
창 바로 바깥은 내삽이 아니라 외삽이 된다. 그 거래가 남는 장사인지는 가정하지 않고
측정한다 — 플래그가 있는 이유가 그것이다.

두 계열이 서로 다른 방식으로 지정한다.

- **`--cede-from <체크포인트>`** (`cmlp`/`cgmlp`/`cjmlp`) — 반경이 아니라 **패치
  체크포인트**를 받는다. 반경으로 지정하는 것은 틀리고, 측정 가능하게 틀리다: 대칭
  2.5 mm 정사각을 양도하면 80 000행이 빠지는데 페이스트는 68 500행만 덮어, **어느 모델도
  배우지 않은 11 500행**이 남는다. 이유는 둘 다 플래그에서는 보이지 않는다 — 페이스트
  경계는 패치가 실제로 남긴 노드 오프셋의 min/max(`[-2.368, +2.382] mm`, 빔이 노드
  사이에 있어서)이고, 거기에는 시간 범위도 붙어 있어서 `--exclude 1`로 학습한 패치는
  `t = 0.4 s`부터 시작한다.
- **`--cede-radius <미터>`** (`fdon`/`gfdon`/`jfdon` + `r*` 3종) — 여기서는 반경이 맞다.
  `pkfdon`이 데이터 유래 경계가 아니라 자기 반경으로 붙이고 저장된 모든 시각을 덮으므로
  둘이 정확히 일치한다.

양도한 창은 베이스 체크포인트에 기록된다. 그것이 없으면 페이스트 시점에 양도된 베이스와
평판 전체 베이스를 구분할 방법이 없고, 크기가 어긋나면 어느 모델도 배우지 않은 고리가
남거나 배운 땅을 덮어쓰게 되는데 — 둘 다 출력에서는 그냥 "더 나쁜 모델"로만 보인다.

> 구현: `models/_ceded.py`, `models/_pathdon.py` — `SequenceCorpus._mask`

<a id="t-paste"></a>
### 패치 페이스트 (PeakCorrectedAgent)

네트워크가 아니라 **두 개의 agent로 만들어진 agent**. 어디서나 베이스로 답하고, 빔
주변 창 안에서는 패치로 답한다. 같은 [추론 계약](#t-agent-contract)을 만족하므로 감싼
것의 drop-in 대체가 된다.

- 붙는 온도는 패치 자신의 Kelvin 값이고 **블렌딩하지 않는다**. 움직이는 것은 *창*이지
  값이 아니므로, 가장자리의 이음매는 진짜 불연속이고 그것이 뜻하는 바 그대로 읽힌다 —
  그 점에서 두 모델의 불일치.
- 각 행은 **정확히 하나**의 모델이 답하고 한 번만 평가된다. 예전에는 베이스를 전부 돌린
  뒤 창 안을 덮어썼는데, 베이스 작업의 6.3%를 버리는 데다 두 모델이 거기서 발언권을
  나눠 가진 것처럼 읽혔다. 그렇지 않다.
- 페어링을 신뢰하지 않고 **검사한다**: 같은 툴패스인가, [양도한 창](#t-cede)과 패치가
  덮는 창이 같은가, [소등 프레임 고정](#t-retouched) 여부가 양쪽에서 일치하는가.

툴패스 계열에서는 한 클래스(`GenericPeakCorrectedAgent`)가 `pkdon`↔`don`과
`pkfdon`↔`fdon` 양쪽을 처리한다. 패치에게 "어느 행이 네 소관인가"(`frame`)와 "그것들에
대해 뭐라고 말하는가"(`predict_at`) 두 가지만 물어보고 내부에 손대지 않기 때문이다.

> 구현: `agent.py` — `PeakCorrectedAgent`; `models/_pathdon.py` —
> `GenericPeakCorrectedAgent`

<a id="t-retouched"></a>
### 소등 프레임 고정 (retouched beam state)

`r*` 4종을 원본과 구분하는 **유일한** 차이.

소등 구간 동안 헤드는 30 mm/s로 다음 트랙으로 이동하며 아무것도 증착하지 않는다. 기본
빔 상태는 그 헤드를 따라가는데, 그러면 두 가지가 동시에 틀어진다.

- 헤드를 따라가는 창이 용융풀이 없는 곳에 있게 되고, 정작 서술해야 할 풀은 레이저가
  마지막으로 켜져 있던 자리에서 식고 있다.
- 게이트는 `lit`을 곱하므로 0이 되어, 그 풀에 대해 모델이 아무 모양도 갖지 못한다.
  `raster`는 런의 22%, `nested_l`은 16%가 이 상태이고 현재 전부 사각지대다.

`retouched_beam_state`는 위치·진행 방향·속도를 마지막 점등 값에 고정하고 `lit`을 1로
유지한다. 프레임이 **헤드 추적을 멈추고 풀 추적을 시작**하는 것이고, 풀이야말로
downstream의 모든 모델이 실제로 서술하려는 대상이다.

첫 점등 이전과 스캔 종료 이후는 고정하지 않는다. 전자는 아직 풀이 없고, 후자는 런이
끝났기 때문이다.

> 구현: `models/_pathdon.py` — `retouched_beam_state`, `beam_of`

<a id="t-equal-condition-batch"></a>
### 조건별 균등 배칭

전체 점에 대해 균일하게 뽑지 않고, `(toolpath, power)` 조건마다 **같은 개수**를 뽑는다.

런들의 길이가 최대 12% 차이 나므로, 그것을 반영한 배치는 `raster`를 `spiral`보다
무겁게 가중하게 된다 — 그 런의 점프가 오래 걸린다는 이유로. 네 패턴과 여섯 power가
동등하게 기여해야 branch가 분리하도록 요구받는 것을 실제로 분리한다.

배치 크기는 `--batch-data`를 조건 수의 배수로 반올림해서 정한다.

> 구현: `models/_pathdon.py` — `PathCorpus.batch`

<a id="t-index-decoded"></a>
### 인덱스 디코딩 코퍼스 (좌표 비물질화)

런은 균일 격자이므로 `(x, y, z, t)`는 평탄 인덱스의 산술 함수다. 저장하면 정보 없이
온도의 여섯 배 메모리를 쓴다 — 115 M 점이면 `torch.arange`가 정확히 재현하는 좌표
3.7 GB.

그래서 `T`만 GPU에 올리고, 배치는 자기가 뽑은 인덱스에서 좌표를 **디코딩**한다. 파일이
`t` 바깥, 그다음 `x, y, z` 순서로 쓰이므로 인덱스는
`((it * nx + ix) * ny + iy) * nz + iz`로 나뉜다.

train/validation 분할도 조건별로 한 번만 하고 인덱스 배열로 들고 있어서, 배치가 거부
루프가 아니라 gather 한 번이다. 조건 최대가 4.4 M 점이라 int32면 충분하고, 인덱스
비용이 절반이 된다.

`t`는 코퍼스 공통 스텝이 아니라 **각 조건 자신의 스냅샷 시각**에서 가져온다. 네 패턴의
길이가 달라 저장 스냅샷 수도 다르므로, 공유 스텝은 모든 런이 우연히 같은 `--snap_dt`를
쓰는 동안만 맞다.

> 구현: `models/_pathdon.py` — `PathCorpus._decode`, `SequenceCorpus._xyz`

<a id="t-point-split"></a>
### 점 단위 분할

[시퀀스 모델](#t-sequence)의 train/validation 분할은 시공간 샘플이 아니라 **점** 단위다.
한 점과 그 히스토리 전체가 한쪽으로 함께 간다.

히스토리 내부를 쪼개면 어떤 노드의 `t = 4.0 s`가 학습에, `t = 4.4 s`가 검증에 들어가는데,
그것은 무엇도 hold-out한 것이 아니다.

> 구현: `models/_pathdon.py` — `SequenceCorpus.__init__`

<a id="t-calibrate"></a>
### 물리 상수 확보 (보정 vs 직독)

게이트와 PINN 잔차에는 `.npy` 파일이 들고 있지 않은 재료·레이저 상수가 필요하다. 두
경로가 있다.

**직선 패스 계열 — `python calibrate.py`로 데이터에서 복원.** 흡수율 0.4593, 빔 반경
1.6971 mm, 빔 시작 `x` 4.8683 mm, `y` 4.9929 mm, 스캔 속도 10.0 mm/s, 확산계수
`alpha = 2.3935e-6 m²/s`(내부 Laplacian에 대해 상관 0.979). 윗면 에너지 수지가 피크
flux의 0.86%까지 닫힌다. `k = alpha * rho * c_p`로만 결정되므로 `rho`와 `c_p`는 가정값이고,
대류계수 `h`와 방사율 `eps`는 **식별 불가능**이라(측면 법선미분이 0.5 mm 내보내기 격자의
잡음 바닥에 있다) 입력으로 남는다 — 윗면 수지의 약 2.5%를 차지한다.

이 상수 블록은 각 모델의 `laser.py`/`loss.py`에 **복사**되어 있다. 한 모델의 실험이
다른 모델을 몰래 움직이는 것을 막기 위해서다. 데이터 디렉터리 내용이 바뀌면
`calibrate.py`를 다시 돌려 블록을 붙여 넣는다.

**툴패스 계열 — `config.json` 직독.** 이 런들은 설정 파일을 함께 내보내므로 복원할 것이
없다. 빔 반경, 확산계수, 평판 두께는 솔버가 실제로 받은 값이다. 다시 맞추는 것은 오차만
더할 뿐이다.

> 구현: `calibrate.py`, `models/cjmlp/laser.py`, `models/pimlp/loss.py`,
> `models/_pathdon.py` — `PathCorpus._physics`

---

## 스펙트럼 전용 (`fmlp`)

<a id="t-rfft-xy"></a>
### xy 전용 변환 (`z`는 격자 유지)

`x`와 `y`만 변환하고 `z`는 손대지 않는다. `z`는 아래쪽 Dirichlet 기판(솔버가 정확히
298 K로 유지)에서 레이저가 달구는 윗면(250 W에서 ambient + 1917 K)까지 간다. DFT는 그
두 면을 서로 감아 붙이므로, 존재하지 않는 1917 K 절벽을 보게 되고 계수는 `1/m`으로
감쇠하며 절단하면 링잉한다.

`z`를 격자에 남기면 저장 숫자가 `nz`배 늘지만, **정확한 아랫면**을 공짜로 되돌려받는다 —
이 계수들로부터의 복원이 `z = 0`을 0.00 K로 재현한다. 구조적으로.

장이 실수이므로 `C(-kx, -ky) = conj(C(kx, ky))`이고 `y`는 음이 아닌 절반만 저장한다
(`rfftn`이 버리고 `field()`가 되돌린다).

변환되는 것은 `T`가 아니라 `dT = T - T_amb`다. 298 K 받침대 위에 앉은 장을 변환하면
DC 계수 전체를 그 사실을 말하는 데 쓰게 된다.

> 구현: `models/fmlp/dataset.py`

<a id="t-energy-box"></a>
### 에너지 모드 절단

몇 개의 모드를 저장할지를 손이 아니라 **에너지**로 정한다. `energy_box`는
`--energy-target` 비율을 **모든 power 각각에 대해** 유지하는 가장 싼 `|mx| <= kx,
|my| <= ky`를 찾는다. 합계가 아니라 각각인 이유는 에너지가 `P^2`으로 스케일해서, 합에
맞춘 예산은 100 W를 통째로 버려도 통과하기 때문이다.

```
target      kx    ky    출력 수      바닥 RMSE    피크
0.9999      19     5     11 700      0.53 K     -0.68%
0.99999     38     6     26 950      0.20 K     -0.04%
```

(일곱 power 중 최악인 250 W 기준.) **바닥**은 남긴 모드가 *정확히* 복원하는 값이다.
거기에 맞춘 어떤 네트워크도 그보다 잘할 수 없으므로, 학습 스크립트가 모델 자신의 오차와
나란히 출력한다. 둘 사이의 간극만이 네트워크가 책임질 몫이다.

> 구현: `models/fmlp/dataset.py` — `energy_box`

<a id="t-detrend"></a>
### x 램프 분리 (`--detrend`)

`kx`가 19에서 38로 뛰는 것은 용융풀이 날카로워져서가 아니라 **x-wrap 능선** 때문이다.
DFT가 평판의 먼 끝과 가까운 끝 사이에서 보는 15.7 K 계단 — 평판이 갖고 있지 않은
주기성이 만든 인공물이다.

변환 전에 그 불일치 전체를 나르는 선형 램프를 빼고, 램프는 `ny * nz`개 숫자로 따로
저장한다. 꼬리가 무너진다(0.99999에서 `kx = 38 -> 24`). 비용만의 문제가 아니다: 그
능선은 레이저를 따라 움직이지 않고 **도메인에 고정**되어 있는데, 그것이 바로
[위상 역회전](#t-derotate)이 다룰 수 없는 성분이다.

> 구현: `models/fmlp/dataset.py` — `detrend_x`; `models/fmlp/model.py` — `retrend_x`

<a id="t-derotate"></a>
### 위상 역회전 (de-rotation)

타깃을 "여덟 개의 숫자 목록"이 아니라 **시간의 함수**로 만드는 기법.

속도 `v`로 움직이는 열원은 장을 대략 `f(x - v t)`로 만들고, 그 변환은
`g(kx) * exp(-2i pi kx v t)`다 — 계수마다 복소평면에서 회전하고, 높은 모드일수록 빨리
돈다. 모드 `mx = 19`는 2.8 s 런 동안 13바퀴 도는데 런은 0.4 s마다 8번만 샘플링되므로
`|mx| = 5`까지만 분해된다. 그 위는 전부 aliasing이다.

`spin`은 그 위상이고, **해석적으로 알려져 있다**. 데이터셋에서 나눠 빼고 복원에서 곱해
넣는다 — 정확하고 가역인 곱셈이라 아무것도 잃지 않는다.

스냅샷이 그것을 aliasing한다는 사실은 반론이 되지 않고, 이것이 핵심이다. 위상은 정확한
시각에서 **평가**되는 것이지 시각으로부터 **추론**되는 것이 아니다. aliasing이 막는 것은
추론이다.

건너뛰었을 때의 대가는 헤드라인 지표에 거의 가려진다. `--no-derotate`로 학습해도 hold-out
power에서 0.421 K(역회전 0.353 K)라 사소한 ablation처럼 보이는데, 검증이 *power*를
hold-out하고 회전은 power에 의존하지 않기 때문이다. 보여준 적 없는 **시각**을 물으면
둘은 완전히 갈라진다: 스냅샷 사이에서 raw 모델의 피크는 1230 K에서 770 K로 무너졌다가
왕복하고, 역회전 모델은 1230 K를 평평하게 유지한다 — 준정상 용융풀이 실제로 하는 일이
그것이다. raw 모델은 aliasing된 여덟 샘플을 외웠고, 역회전 모델은 `t`의 함수를 배웠다.
대리모델인 것은 후자뿐이다.

> 구현: `models/fmlp/dataset.py` — `spin`; `models/fmlp/model.py` — `_lab_frame`

---

## 학습 · 최적화

<a id="t-adam-cosine"></a>
### Adam + 코사인 어닐링

기본 옵티마이저. 학습률은 전체 일정에 걸쳐 `CosineAnnealingLR`로 감쇠한다.

툴패스 DeepONet들은 기본 학습률이 다르다. `utils.DEFAULT_LR`은 Adam에서 1e-3이고 다른
모델들은 4층인데 이들은 6층이라, 1e-3에서는 셋 다 발산한다 — 그것도 얌전하지 않게:
`don` 런의 8000번째 반복이 **17000 K 검증 오차**로 갔다. 그래디언트가 이미 norm 1.0으로
클리핑된 상태에서. 15000 반복에 걸쳐 재보니 3e-4는 튐 없이 단조 감소한다. 그래서
이 계열의 기본값은 3e-4이고, 명시적 `--lr`은 여전히 이긴다.

> 구현: `models/_pathdon.py` — `DEFAULT_LR`, 각 `train.py`의 scheduler

<a id="t-lbfgs"></a>
### L-BFGS 고정 배치

`--optimizer lbfgs`. L-BFGS는 연속한 그래디언트에서 곡률을 추정하는데, 그것이 의미를
가지려면 연속한 스텝이 **같은 함수**에서 나와야 한다. 그래서 선택되는 순간 배치는 한 번만
뽑히고 절대 재샘플되지 않는다.

무엇이 그 고정된 목적함수인지를 세 플래그가 정한다.

- `--lbfgs-batch` (기본 65 536) — 학습 시작에 한 번 뽑는, L-BFGS가 맞출 **그 한** 샘플
- `--lbfgs-full` (현재 `mlp`만) — 고정된 목적함수가 샘플이 아니라 **전체 train 분할**
  (7 power에 걸쳐 8 317 260행)이다. `--lbfgs-batch` 크기 청크로 흘려보내며 그래디언트를
  누적하므로 결과가 진짜 full-batch 스텝과 수학적으로 동일하다. 근사가 아니다. 스텝당
  훨씬 느리다.
- `--freeze-batch` — Adam에서도 배치를 한 번만 뽑는다. 배치와 옵티마이저를 독립적으로
  바꿔볼 수 있어야 어떤 효과가 어느 쪽 것인지 구분할 수 있다.

[PINN 모델](#t-resampled-collocation)은 `--lbfgs-batch` 대신 자기 점 집합들
(`--batch-data`/`-physics`/`-boundary`)을 얼린다.

관측된 천장: Adam 체크포인트에서 웜스타트한 `cmlp`/`cgmlp`는 첫 epoch 이후 정체한다.
line search가 그 고정 샘플에서 개선할 것을 못 찾으므로 남은 19 epoch가 전부 no-op다.
`cpimlp`는 Adam 체크포인트가 애초에 수렴에서 멀어서 같은 웜스타트가 10.734 K → 4.800 K의
실질 개선을 낸다.

> 구현: `utils.py` — `add_optimizer_args`, `build_optimizer`; `models/mlp/train.py`

<a id="t-warm-start"></a>
### 웜스타트 (`--init-from`)

학습 전에 다른 체크포인트의 가중치에서 시작한다(아키텍처가 일치해야 한다). Adam으로
학습한 모델을 무작위 초기화가 아니라 그 지점에서 L-BFGS로 이어가려고 쓴다.

> 구현: `models/{cmlp,cgmlp,cjmlp,cpimlp,cpkmlp}/train.py`

<a id="t-grad-clip"></a>
### 그래디언트 클리핑 + 비유한 스텝 건너뛰기

모든 학습 루프가 그래디언트를 norm 1.0으로 클리핑한다. 추가로, 비유한 그래디언트나
손실이 나온 스텝은 **밟지 않고 건너뛴다** — 비유한 값으로 밟은 스텝은 Adam의 모멘트를
오염시키고, 그 뒤로는 아무리 작은 스텝도 전부 틀리기 때문이다.

이 검사가 [위에서 말한 발산](#t-adam-cosine)을 잡지는 **못한다**는 점을 코드가 명시해
두었다. 그 실패에서는 모든 숫자가 유한했고 클리핑도 이미 걸려 있었다 — 스텝 크기가 곡률에
비해 컸을 뿐이다. 그건 학습률로 고치고, 이 검사는 다른 실패를 위한 것이다.

> 구현: 각 `train.py`의 `clip_grad_norm_`, `models/_pathdon.py` — `run()`

<a id="t-best-checkpoint"></a>
### 최고 체크포인트 선택

검증 RMSE 기준 **단 하나**의 최고 체크포인트를 `checkpoints/<model>/<run-name>.pt`에
남긴다. 선택은 항상 in-distribution 검증 분할로 하고 [hold-out](#t-holdout-power)으로는
절대 하지 않는다.

체크포인트에는 가중치뿐 아니라 그것을 재현하는 데 필요한 전부가 들어간다: `architecture`
(생성자 키워드 그대로), `bounds`, `sensors` 격자, `field_shape`, `val_rmse`,
`holdout_rmse`, 그리고 해당되면 `radius` / `times` / `retouched` / `cede_radius`.
sensor 격자와 복원 shape가 체크포인트에 실려 있어서, branch가 호출자가 추측한 시각이
아니라 **학습에 쓰인 시각**에서 읽힌다.

> 구현: `utils.py` — `BestCheckpoint`; `models/_pathdon.py` — `run()`

<a id="t-agent-contract"></a>
### 공통 추론 계약 (`predict_at` / `predict_of`)

모든 모델이 `agent.py`의 `build_agent(checkpoint)`를 노출하고, 안에 무엇이 들었든 같은
두 질문에 답한다.

| 메서드 | 입력 | 출력 |
|---|---|---|
| `predict_at` | `[B, 5]` of `(x, y, z, t, P)` | `[B, 1]` K |
| `predict_of` | `[B, 2]` of `(t, P)` | `[B, 1, D, H, W]` K — 부피 전체 |

`predict_of`는 베이스 클래스에서 `predict_at`으로부터 파생되므로(`fmlp`는 반대) 새 모델이
부피 뷰를 공짜로 얻고 둘이 어긋날 수 없다. `(D, H, W)`는 `Conv3d` 관례를 따른다:
`D = z`, `H = y`, `W = x`. 공간·시간 경계는 데이터셋이 아니라 **체크포인트**에서 오므로
저장된 모델을 그리는 데 13 GB 코퍼스를 건드리지 않는다.

계약은 [P를 못 쓰는 모델](#t-p-blind)에게도 `P`를 넘긴다. 그들은 열을 받고 버리므로 모든
스크립트에 그대로 꽂힌다.

툴패스 계열에는 여섯 번째 것 — **어느 경로인가** — 이 필요한데 그 시그니처에는 자리가
없다. 그래서 agent를 만들 때 고정한다. agent는 "어떤 스캔 패턴에서의 모델"이고, 두 패턴을
비교하려면 하나의 체크포인트에서 두 개의 agent를 만든다. 이것은 임시방편이 아니다 —
branch는 장 전체에 대해 상수라서 경로·power 하나당 한 번만 평가되고, 여기서 고정하는
덕분에 `predict_of`가 부피를 단 한 번의 pass로 복원한다.

`toolpath`를 넘기지 않으면 첫 번째 런으로 기본값이 잡히되 **경고한다**. 기본값은 진짜
함정이다: `visualize.py`는 경로를 넘길 자리 없이 `build_agent`를 부르고 비교 대상 장은
`--data-dir`로 고른다. 둘이 일치한다는 보장이 어디에도 없고, `nested_l`의 빔 아래
`spiral`의 온도를 그린 그림은 맞아 보이는 틀린 그림이다.

[시퀀스 모델](#t-sequence)의 agent는 히스토리를 저장 시각 사이에서 선형 보간해 이 계약에
맞춘다. 그 보간은 모델의 것이 아니라 **agent의 것**이다. `visualize_don.py`처럼 저장된
시각을 물으면 아무 비용도 들지 않는다.

> 구현: `agent.py` — `BaseAgent`; `models/_pathdon.py` — `PathAgent`, `SequenceAgent`,
> `SequencePatchAgent`, `PatchAgent`, `build_agent`
