# 에어백 최적 제어 RL 프로젝트 기술 설계 보고서

> 작성 기준: `env/airbag.py`, `rl/reward.py`, `env/airbag_env.py`, `env/human.py`, `env/scenario.py`, `rl/ppo.py`, `assets/create_car_usd.py`, `config/config.yaml`

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [에셋 구성 및 파일 스펙](#2-에셋-구성-및-파일-스펙)
3. [에어백 시스템 설계](#3-에어백-시스템-설계)
4. [압력-감쇠율 모델 (Reverse U-Curve)](#4-압력-감쇠율-모델-reverse-u-curve)
5. [에어백 종류별 감쇠율 스케일링 (k 상수)](#5-에어백-종류별-감쇠율-스케일링-k-상수)
6. [충격량 측정 설계](#6-충격량-측정-설계)
7. [상해 지표 및 기준선](#7-상해-지표-및-기준선)
8. [보상 함수 설계](#8-보상-함수-설계)
9. [RL 알고리즘 설계](#9-rl-알고리즘-설계)
10. [시나리오 샘플링 및 State 벡터](#10-시나리오-샘플링-및-state-벡터)
11. [출처 및 근거](#11-출처-및-근거)

---

## 1. 프로젝트 개요

본 프로젝트는 **NVIDIA Isaac Sim 6.0** 물리 시뮬레이터 위에서 차량 충돌 시 에어백의 **전개 여부·타이밍·압력**을 실시간으로 최적화하는 강화학습 에이전트를 설계한다.

| 항목 | 상세 |
|---|---|
| 시뮬레이터 | Isaac Sim 6.0 (PhysX 5, USD) |
| 인체 모델 | Newton humanoid.usda (관절 17개 이상) |
| 에어백 수 | 5개 (운전석 전면·동승석 전면·운전석 측면·동승석 측면·커튼) |
| RL 알고리즘 | PPO (Proximal Policy Optimization) |
| State 차원 | 12 |
| Action 차원 | 15 (에어백 5개 × [전개여부, 타이밍, 압력]) |
| 물리 스텝 | 1 ms (1,000 Hz) |
| 제어 주기 | ~16.7 ms (60 Hz) |
| 에피소드 길이 | 60 스텝 (≈ 1초 충돌 구간) |

---

## 2. 에셋 구성 및 파일 스펙

시뮬레이션에 등장하는 4가지 오브젝트(차량, 인체, 에어백, 안전벨트)는 각각 다른 방식으로 구성된다.

### 2.1 차량 (Vehicle)

| 항목 | 내용 |
|---|---|
| 파일 | `assets/vehicle.usd` |
| 생성 방법 | `assets/create_car_usd.py` 로 **프로그래밍 생성** (외부 3D 모델 없음) |
| 표현 방식 | USD `UsdGeom.Cube` / `UsdGeom.Cylinder` 프리미티브 조합 — **별도 메시 없음** |
| 물리 방식 | `UsdPhysics.RigidBodyAPI` (SingleRigidPrim) |
| 질량 | 1,500 kg (SUV 기준) |
| 무게중심 | 바닥 기준 +0.6 m (z축) |

**구성 파트 및 치수**:

| 파트 | 형태 | 치수 |
|---|---|---|
| body (차체) | Box 콜라이더 | 4.2 m(L) × 1.9 m(W) × 1.2 m(H), z=0.7 m |
| roof (루프) | Box 콜라이더 | 2.0 m(L) × 1.6 m(W) × 0.7 m(H), z=1.65 m |
| wheel_FL/FR/RL/RR (바퀴 4개) | Cylinder 콜라이더 | 반지름 0.35 m, 두께 0.22 m |

- USD 계층: `/World/vehicle` → 자식으로 `/World/vehicle/body`, `/World/vehicle/roof`, `/World/vehicle/wheel_*`
- 에어백 sphere prim도 `/World/vehicle/airbag_<i>` 로 **vehicle 계층 아래 배치** → 차량 추종 자동화

### 2.2 인체 (Human)

| 항목 | 내용 |
|---|---|
| 파일 | Newton 패키지 내 `examples/assets/humanoid.usda` |
| 출처 | **Isaac Sim 번들 Newton humanoid** (NVIDIA 공식 제공) |
| 표현 방식 | **USDA 참조(Reference)** — `add_reference_to_stage()` 로 Stage에 로드 |
| 물리 방식 | `UsdPhysics.ArticulationRootAPI` (관절 아티큘레이션) |
| 파이썬 인터페이스 | `isaacsim.core.prims.SingleArticulation` |
| 관절 수 | Newton humanoid 기준 17개 이상 |
| 체형 파라미터 | 신장 1.55~1.90 m, 체중 50~100 kg (에피소드마다 샘플링) |

**관절 매핑 (DOF 이름 힌트 기반 자동 탐지)**:

| 부위 | 힌트 키워드 | 적용 |
|---|---|---|
| 고관절 굴곡 | `hip_y`, `hip_pitch`, `hip_flex` | 착좌 시 −90° 인가 |
| 무릎 굴곡 | `knee` | 착좌 시 +90° 인가 |
| 척추 기울기 | `abdomen_y`, `lower_waist`, `spine`, `lumbar`, `torso_pitch` | −10°~+20° 랜덤 |

**링크 인덱스 자동 탐지**:
- `torso/chest/pelvis` 키워드 → 흉부 링크 인덱스
- `head` 키워드 → 두부 링크 인덱스
- `right_thigh` / `thigh+right` → 우측 대퇴부 링크 인덱스

### 2.3 에어백 (Airbag)

| 항목 | 내용 |
|---|---|
| 파일 | 없음 — **런타임에 프로그래밍 생성** |
| 표현 방식 | `isaacsim.core.api.objects.VisualSphere` (USD Sphere Prim) — **메시 없음, 구체 프리미티브** |
| 물리 방식 | `UsdPhysics.CollisionAPI` + `RigidBodyAPI(kinematic=True)` + PhysxMaterialAPI |
| 재질 파일 | 없음 — 런타임에 `/World/airbag_soft_material` prim 생성 |
| USD 경로 | `/World/vehicle/airbag_0` ~ `/World/vehicle/airbag_4` |
| 팽창 표현 | `UsdGeom.Sphere.GetRadiusAttr().Set(r)` 로 반지름을 매 스텝 업데이트 |

**PhysX 재질 설정**:

| 파라미터 | 값 |
|---|---|
| Restitution (반발계수) | 0.0 |
| Static / Dynamic Friction | 0.05 |
| CompliantContactStiffness | 150,000 N/m |
| CompliantContactDamping | 15,000 Ns/m |

### 2.4 안전벨트 (Seatbelt)

| 항목 | 내용 |
|---|---|
| 파일 | 없음 — **런타임에 프로그래밍 생성** |
| 시각적 표현 | `isaacsim.core.api.objects.VisualCapsule` 2개 (어깨벨트, 무릎벨트) — **메시 없음** |
| 물리 표현 | 별도 충돌체 없음 — **수식 기반 힘 인가 방식** |
| 동작 방식 | 흉부 링크 속도(relative vel)에 비례한 제동력을 `physics_view.apply_forces_and_torques_at_position()` 으로 직접 인가 |

**안전벨트 물리 파라미터** (FMVSS 209 / ECE R16 기준):

| 파라미터 | 값 | 의미 |
|---|---|---|
| k_belt | 8,000 N/(m/s) | 벨트 강성·감쇠 계수 |
| F_cap | 15,000 N | 로드 리미터 최대 하중 (프리텐셔너 포함) |

```
F_belt = clip(-rel_vel × k_belt, -F_cap, +F_cap)
rel_vel = v_torso - v_vehicle
```

### 2.5 충돌 벽 (Collision Wall)

| 항목 | 내용 |
|---|---|
| 파일 | 없음 — 런타임 생성 |
| 표현 방식 | `isaacsim.core.api.objects.FixedCuboid` |
| 크기 | 0.5 m(D) × 5.0 m(W) × 3.0 m(H) |
| 배치 | 차량에서 3.5 m 거리, 충돌 각도에 따라 회전 배치 |
| 강성 옵션 | concrete(기본)/vehicle/wood — 시나리오에 따라 다름 |

### 2.6 에셋 구성 요약

| 오브젝트 | USD 파일 | 표현 방식 | 물리 방식 |
|---|---|---|---|
| 차량 | `assets/vehicle.usd` (스크립트 생성) | Box + Cylinder 프리미티브 | RigidBody |
| 인체 | Newton `humanoid.usda` (Isaac Sim 번들) | USDA Reference | Articulation (관절체) |
| 에어백 | 없음 (런타임 생성) | VisualSphere 프리미티브 | Kinematic + Compliant Contact |
| 안전벨트 | 없음 (런타임 생성) | VisualCapsule (시각용만) | 수식 힘 인가 |
| 충돌 벽 | 없음 (런타임 생성) | FixedCuboid | Static Collider |

---

## 3. 에어백 시스템 설계

### 3.1 에어백 5종 스펙

| ID | 이름 | 체적(L) | 최대 반지름(m) | 보호 부위 | 유효 충돌 각도 |
|---|---|---|---|---|---|
| 0 | front_driver (운전석 전면) | 60 | 0.30 | 머리, 흉부 | 315°~45° (전면) |
| 1 | front_passenger (동승석 전면) | 120 | 0.35 | 머리, 흉부 | 315°~45° (전면) |
| 2 | side_driver (운전석 측면) | 15 | 0.22 | 흉부 | 225°~315° (좌측) |
| 3 | side_passenger (동승석 측면) | 15 | 0.22 | 흉부 | 45°~135° (우측) |
| 4 | curtain (커튼) | 40 | 0.28 | 머리 | 45°~315° (측·후면) |

- **유효 각도 로직**: 충돌 각도가 각 에어백의 `angle_range` 밖이면 전개하더라도 감쇠력을 인가하지 않음
  - front_driver/passenger는 `lo > hi`(315 → 45) 형태의 wrap-around 범위를 처리

### 3.2 물리 모델: Kinematic Soft-Bag

에어백은 **USD PhysX** 기반 물리 재질을 직접 부여한 구체(Sphere)로 구현한다.

```
CollisionAPI  →  인체와 물리 접촉 활성화
RigidBodyAPI  →  kinematic=True  (에어백 자체는 외력에 날아가지 않음)
PhysxMaterialAPI compliant contact  →  낮은 강성 + 높은 감쇠 (말랑한 쿠션)
```

**Compliant Contact 파라미터 (PhysX soft spring-damper)**:

```
F_contact = K × penetration + D × penetration_rate

K (Contact Stiffness) = 150,000 N/m
D (Contact Damping)   =  15,000 Ns/m
```

- `K = 150,000 N/m` → rigid wall 대비 1/100 이하 → 접촉 시 튕기지 않고 에너지 흡수
- `D = 15,000 Ns/m` → 과감쇠(overdamped) → 반동 없이 속도 감쇠
- 마찰계수: 정지·동적 모두 0.05 (미끄러짐 허용, 회전 상해 방지)
- 반발계수(Restitution): 0.0 (반발력 없음)

### 3.3 팽창 모델

**팽창 기준**: FMVSS 208 실험 기준값 50 ms 이내 팽창 완료

$$\text{radius}(t) = r_{\max} \times \min\!\left(\frac{t - t_{\text{start}}}{50\,\text{ms}},\; 1\right)$$

- `t_start`: RL 에이전트가 출력한 `timing_ms`에 도달한 시점
- 팽창 완료(ratio=1.0) 이후에는 반지름을 고정하고 물리 감쇠만 적용
- 전개 조건: `deploy > 0.5` AND `current_ms ≥ timing_ms` AND 충돌 각도 유효

---

## 4. 압력-감쇠율 모델 (Reverse U-Curve)

### 4.1 설계 근거

실제 에어백은 **압력이 너무 낮으면** 쿠션이 없어 충격 흡수 불가, **압력이 너무 높으면** 에어백이 딱딱해져 오히려 상해를 유발한다. 이를 모델링하기 위해 최적 압력에서 정점을 갖는 역U자 곡선(Reverse U-Curve)을 적용한다.

**출처**: NHTSA 에어백 압력-상해 관계 연구 (Brinkley/Eiband 곡선 형태 참조), FMVSS 208 에어백 성능 기준

### 4.2 수식

$$\delta(P) = D_{\text{base}} \cdot \frac{2x}{1 + x^2}, \quad x = \frac{P}{P_{\text{opt}}}$$

| 파라미터 | 값 | 의미 |
|---|---|---|
| $P_{\text{opt}}$ | 300 kPa | 감쇠 효과 최대화 최적 압력 |
| $D_{\text{base}}$ | 0.75 | 감쇠율 최대값 (상한) |

- $x = 1$ (P = 300 kPa)일 때: $\delta = 0.75$ (최대)
- $x \to 0$ (P → 0): $\delta \to 0$ (쿠션 없음)
- $x \to \infty$ (P → ∞): $\delta \to 0$ (너무 딱딱해짐)
- 출력은 `[0, 0.75]` 범위로 클리핑

```python
# env/airbag.py: _reverse_u_curve()
def _reverse_u_curve(pressure_kpa: float) -> float:
    x = pressure_kpa / PRESSURE_OPT          # PRESSURE_OPT = 300.0
    return DAMPING_BASE * (2 * x) / (1 + x ** 2)   # DAMPING_BASE = 0.75
```

---

## 5. 에어백 종류별 감쇠율 스케일링 (k 상수)

### 5.1 설계 기준: 체적(Volume)과 분사율(Inflation Rate)

에어백 종류마다 **체적**과 **분사율**이 다르기 때문에, 동일한 압력 커브를 적용해도 실제 쿠션 특성이 달라진다.

- **체적이 클수록** (동승석 120L): 동일 압력에서 팽창에 더 많은 가스가 필요 → 분사율 대비 압력 상승이 느림 → 감쇠 스케일을 낮추는 방향으로 `k`를 설계
- **체적이 작을수록** (측면 15L): 소량의 가스로 고압 달성 가능 → 빠른 팽창 + 높은 감쇠 → `k`를 크게 설계

### 5.2 k 상수 표

| 에어백 | 체적(L) | k 상수 | 설계 근거 |
|---|---|---|---|
| front_driver | 60 L | **1.0** | 기준값 (표준 운전석 전면 에어백) |
| front_passenger | 120 L | **0.5** | 체적 2배 → 분사율 대비 압력 효율 절반, 감쇠 스케일 절반 |
| side_driver | 15 L | **4.0** | 소용량 + 고압 분사 → 단시간 고감쇠 필요, 스케일 4배 |
| side_passenger | 15 L | **4.0** | 위와 동일 |
| curtain | 40 L | **1.5** | 중간 체적, 측면 충격 보호 특화 → 1.5배 스케일 |

### 5.3 최종 감쇠력 계산

$$F_{\text{damp},i} = k_i \cdot \delta(P_i) \cdot F_{\text{base}} \cdot (-v_{\text{part}})$$

| 파라미터 | 값 | 의미 |
|---|---|---|
| $k_i$ | 에어백별 (위 표) | 체적·분사율 기반 스케일 상수 |
| $\delta(P_i)$ | Reverse U-Curve 출력 | 압력별 감쇠율 |
| $F_{\text{base}}$ | 500 N | 기준 감쇠력 |
| $v_{\text{part}}$ | 해당 신체 부위 속도 벡터 | 속도 반대 방향으로 힘 인가 |

**예시**: 운전석 전면 에어백(k=1.0), 압력 300 kPa(δ=0.75) 시  
→ $F = 1.0 \times 0.75 \times 500 = 375\,\text{N}$ 반력 인가

```python
# env/airbag.py: _apply_damping()
scale = damping * BASE_FORCE           # damping = k × δ(P), BASE_FORCE = 500.0
force = -vel * scale                   # 속도 역방향 감쇠력
human._apply_link_force(link_idx, force)
```

---

## 6. 충격량 측정 설계

### 6.1 측정 부위 4곳

충격량은 **1 ms 물리 콜백** (`world.add_physics_callback`) 으로 매 스텝 측정한다.  
Isaac Sim의 `ArticulationView.get_link_velocities()` 로 각 링크의 속도를 읽고, **수치미분(유한차분)**으로 가속도를 계산한다.

$$a(t) = \frac{v(t) - v(t - \Delta t)}{\Delta t}, \quad \Delta t = 1\,\text{ms}$$

| 측정 부위 | 링크 | 수집 데이터 | 산출 지표 |
|---|---|---|---|
| **두부 (Head)** | `head` 링크 | 속도 → 3D 가속도 벡터, 합성 가속도(g) | HIC15, Nij |
| **흉부 가속도 (Torso)** | `torso`/`chest` 링크 | 속도 → 합성 가속도(g) | chest_g, chest_3ms |
| **흉부 변위 (Torso)** | `torso`/`chest` 링크 | 위치 이력 | chest_compression |
| **대퇴부 (Thigh)** | `right_thigh` 링크 | 속도 → 3D 가속도 벡터 | femur_force |

```python
# rl/reward.py: InjuryDataCollector.record()
head_acc  = (head_vel  - prev_head_vel)  / dt      # 수치미분
torso_acc = (torso_vel - prev_torso_vel) / dt
thigh_acc = (thigh_vel - prev_thigh_vel) / dt
```

### 6.2 HIC15 계산 (두부 상해 지수)

HIC(Head Injury Criterion)는 15 ms 이내 구간에서의 평균 가속도의 2.5승으로 정의된다.

$$\text{HIC}_{15} = \max_{(t_2 - t_1) \leq 15\,\text{ms}} \left[ (t_2 - t_1) \cdot \left(\frac{1}{t_2 - t_1}\int_{t_1}^{t_2} a(t)\,dt\right)^{2.5} \right]$$

구현상 누적합으로 O(N) 계산:

```python
# rl/reward.py: compute_hic15()
cum = np.concatenate([[0.0], np.cumsum(arr) * dt])
mean_acc = (cum[j] - cum[i]) / dt_window
hic = dt_window * (mean_acc ** 2.5)
```

### 6.3 Nij 계산 (목 상해 지수)

NHTSA FMVSS 208 표준 Hybrid III 기준:

$$N_{ij} = \left|\frac{F_z}{F_{zc}}\right| + \left|\frac{M_y}{M_{yc}}\right|$$

- $F_z = m_{\text{head}} \times a_z$ (두부 질량 × 축방향 가속도, 인장/압축력)
- $M_y = m_{\text{head}} \times a_x \times l_{\text{lever}}$ (시상면 굽힘 모멘트)

| 파라미터 | 값 | 출처 |
|---|---|---|
| $m_{\text{head}}$ | 4.54 kg | Hybrid III 표준 두부 질량 |
| $l_{\text{lever}}$ | 0.105 m | 두부 무게중심 → 후두과 거리 |
| $F_{zc}$ (인장) | 6,806 N | FMVSS 208 Table |
| $F_{zc}$ (압축) | 6,160 N | FMVSS 208 Table |
| $M_{yc}$ (신전) | 310 N·m | FMVSS 208 Table |
| $M_{yc}$ (굴곡) | 135 N·m | FMVSS 208 Table |

### 6.4 흉부 3ms 클립 가속도

3 ms 이상 연속으로 지속되는 최고 가속도를 의미. `scipy.ndimage.minimum_filter1d`로 슬라이딩 최솟값의 최댓값을 계산:

```python
windowed_min = minimum_filter1d(arr, size=3_steps, mode='nearest')
chest_3ms = windowed_min.max()
```

### 6.5 흉부 압축량

전후 방향(x축) 최대 변위를 압축량으로 근사:

$$d_{\text{compression}} = \max_t |x_{\text{torso}}(t) - x_{\text{torso}}(0)| \times 1000\,[\text{mm}]$$

### 6.6 대퇴부 압축력

$$F_{\text{femur}} = m_{\text{thigh}} \times \max_t \|a_{\text{thigh}}(t)\|$$

- $m_{\text{thigh}} = 8.55\,\text{kg}$ (Hybrid III 우측 대퇴부 질량)

---

## 7. 상해 지표 및 기준선

모든 기준값은 **NHTSA FMVSS 208** 및 **Hybrid III 더미 표준**을 적용한다.

| 지표 | 설명 | 기준선 | 출처 |
|---|---|---|---|
| HIC15 | 두부 상해 지수 (15 ms 구간) | ≤ **700** | NHTSA FMVSS 208 |
| chest_g | 흉부 최대 합성 가속도 | ≤ **60 g** | FMVSS 208 |
| chest_3ms | 흉부 3 ms 클립 가속도 | ≤ **60 g** | FMVSS 208 |
| chest_compression | 흉부 전후 압축량 | ≤ **50 mm** | FMVSS 208 |
| femur_force | 대퇴부 최대 압축력 | ≤ **10,000 N** | FMVSS 208 |
| Nij | 목 상해 지수 | ≤ **1.0** | FMVSS 208 |

---

## 8. 보상 함수 설계

### 8.1 설계 철학

**Dense Reward** 구조를 채택하여 에이전트가 매 에피소드마다 연속적인 gradient를 받을 수 있도록 했다.  
에피소드 종료 시(충돌 구간 1초 완료) 한 번 계산된다.

### 8.2 보상 수식

$$r = r_{\text{base}} + r_{\text{violation}} + r_{\text{bonus}} + r_{\text{no\_deploy}}$$

**① Base (연속 페널티)**:

$$r_{\text{base}} = -\sum_{i} \frac{v_i}{s_i}$$

- $v_i$: 각 지표 측정값, $s_i$: 기준선
- 항상 음수 → 지표를 낮출수록 보상 증가

**② Violation (기준 초과 페널티)**:

$$r_{\text{violation}} = -5.0 \times \sum_{i:\, v_i > s_i} \left(\frac{v_i}{s_i} - 1\right)$$

- 기준 초과 항목마다 초과율에 비례해 추가 감점

**③ Safety Bonus**:

$$r_{\text{bonus}} = +2.0 \quad \text{if } \forall i:\, v_i \leq s_i$$

- 전 항목이 기준 이하일 때만 지급

**④ No-Deploy Penalty**:

$$r_{\text{no\_deploy}} = -2.0 \quad \text{if } \sum_i \text{deploy}_i = 0$$

- 에어백을 하나도 전개하지 않으면 감점

```python
# rl/reward.py: compute_reward()
r = sum(-(val / safe) for val, safe in metrics)            # base
for val, safe in metrics:
    if val > safe:
        r -= 5.0 * (val / safe - 1.0)                     # violation
if all(val <= safe for val, safe in metrics):
    r += 2.0                                               # bonus
if sum(deploy_flags) == 0:
    r -= 2.0                                               # no-deploy
```

**예시**: HIC=420 (기준 700의 60%), chest_g=36 g, 나머지 0, 전 항목 안전, 에어백 전개 시  
→ `r ≈ -(420/700 + 36/60 + 0 + 0 + 0 + 0) + 2.0 = -(0.6 + 0.6) + 2.0 = +0.8`

---

## 9. RL 알고리즘 설계

### 9.1 PPO (Proximal Policy Optimization)

에어백 제어의 특성상 **전개 여부(이산)** 와 **타이밍·압력(연속)** 이 혼재하므로, 분리된 분포 헤드를 갖는 **MultiHead Actor-Critic** 구조를 설계했다.

### 9.2 MultiHeadActor 구조

```
입력 (12차원 state)
    ↓
Shared MLP [12 → 128 → 128, Tanh]
    ├── deploy_head  → Bernoulli(logit)  × 5   (전개여부, 이산)
    ├── timing_head  → Normal(μ, σ)     × 5   (타이밍,  연속)
    └── pressure_head→ Normal(μ, σ)     × 5   (압력,    연속)
```

- **deploy=0이면 timing·pressure의 gradient를 마스킹**: 에어백 미전개 시 타이밍·압력이 불필요하므로 `timing = timing × deploy`로 처리
- **log_prob**: 세 분포의 로그 확률 합산, deploy=0이면 timing·pressure 항 기여 0

### 9.3 Action 공간 (15차원)

| 인덱스 | 의미 | 범위 |
|---|---|---|
| [0:5] | 에어백 0~4 전개 여부 | {0, 1} (Bernoulli) |
| [5:10] | 에어백 0~4 타이밍 | 0~30 ms |
| [10:15] | 에어백 0~4 압력 | 0~600 kPa |

### 9.4 PPO 갱신

- **Return**: 에피소드 단위 Monte Carlo (단일 보상)
- **Advantage**: $A = G - V(s)$ ($V$: Critic, detach)
- **Clip ratio**: ε = 0.2
- **Update epochs**: 10회 반복
- **Loss**: $L = L_{\text{actor}} + 0.5 \times L_{\text{critic}}$

$$L_{\text{actor}} = -\mathbb{E}\left[\min\!\left(\rho A,\; \text{clip}(\rho, 1-\varepsilon, 1+\varepsilon)\cdot A\right)\right], \quad \rho = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$$

---

## 10. 시나리오 샘플링 및 State 벡터

### 10.1 시나리오 파라미터 (에피소드마다 무작위 샘플링)

| 파라미터 | 범위 | 의미 |
|---|---|---|
| angle | 0°~360° (uniform) | 충돌 방향 |
| speed | 20~120 km/h (uniform) | 충돌 속도 |
| stiffness | concrete/vehicle/wood | 충돌 대상 강성 |
| seatbelt | 착용/미착용 (50%) | 안전벨트 여부 |
| height | 1.55~1.90 m (uniform) | 탑승자 신장 |
| weight | 50~100 kg (uniform) | 탑승자 체중 |

### 10.2 State 벡터 12차원 (정규화 [0, 1])

| 인덱스 | 의미 | 정규화 식 |
|---|---|---|
| [0] | 충돌 방향 | angle / 360 |
| [1] | 충돌 속도 | speed / 120 |
| [2] | 충돌 강성 | concrete=1.0, vehicle=0.7, wood=0.4 |
| [3] | 안전벨트 | 0 or 1 |
| [4] | 신장 | height / 2.0 |
| [5] | 앉은키 (실측) | sitting_height / 1.2 |
| [6] | 머리 X (전후) | head_pos[0] / 1.5 |
| [7] | 머리 Y (좌우) | (head_pos[1] + 1.0) / 2.0 |
| [8] | 머리 Z (높이) | head_pos[2] / 2.0 |
| [9] | 척추 기울기 | (spine_tilt_deg + 15) / 35 |
| [10] | 머리→스티어링 거리 | head_to_steering / 1.0 |
| [11] | 무릎→대시보드 거리 | knee_to_dashboard / 0.5 |

- **[5]~[11]**: 에피소드 리셋 시 `measure_snapshot()`으로 실측  
  → 신장/자세/착좌 위치에 따른 개인화된 에어백 제어 가능

---

## 11. 출처 및 근거

| 설계 요소 | 출처 |
|---|---|
| 팽창 50 ms 기준 | **FMVSS 208** (Federal Motor Vehicle Safety Standard 208) — 에어백 팽창 시간 기준 |
| HIC15 ≤ 700 | **NHTSA FMVSS 208** — 두부 상해 기준 |
| chest_g ≤ 60 g | **FMVSS 208** — 흉부 최대 가속도 기준 |
| chest_3ms ≤ 60 g | **FMVSS 208** — 흉부 3 ms 클립 기준 |
| chest_compression ≤ 50 mm | **FMVSS 208** — 흉부 압축 변위 기준 |
| femur ≤ 10,000 N | **FMVSS 208** — 대퇴부 압축력 기준 |
| Nij ≤ 1.0 | **NHTSA FMVSS 208** — 목 상해 지수 기준 |
| Hybrid III 두부 4.54 kg | **Hybrid III 표준 더미** (NHTSA 공인 충돌 실험용 인체 모형) |
| 후두과 거리 0.105 m | **Hybrid III 표준 더미** 해부학적 치수 |
| Nij Fzc/Myc 수치 | **FMVSS 208 Table** (인장 6806 N, 압축 6160 N, 신전 310 N·m, 굴곡 135 N·m) |
| 대퇴부 질량 8.55 kg | **Hybrid III 표준 더미** 우측 대퇴부 질량 |
| 안전벨트 k=8000 N/(m/s), F_cap=15000 N | **FMVSS 209 / ECE R16** — 안전벨트 로드 리미터·프리텐셔너 스펙 |
| Compliant Contact K/D | **PhysX 5 문서** — compliant contact stiffness/damping (soft-body 근사) |
| 압력-감쇠 Reverse U-Curve | Brinkley/Eiband 허용 가속도 포락선 형태 참조, NHTSA 에어백 압력-상해 연구 |
| k 상수 (체적·분사율 기반) | ISO 12097 (에어백 모듈 시험 표준) 및 OEM 공개 에어백 사양 기반 추정 |
