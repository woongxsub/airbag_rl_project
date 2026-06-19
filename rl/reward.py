"""
충격량 측정 및 RL 보상 연산 모듈.

물리 데이터 흐름:
  physics_callback (1ms) → InjuryDataCollector.record()
      head_vel  → 수치미분 → head_acc_3d  → HIC15, Nij
      torso_vel → 수치미분 → torso_acc_g  → chest_g, chest_3ms_clip
      torso_pos → 위치이력              → chest_compression
      thigh_vel → 수치미분 → thigh_acc_3d → femur_force
"""

import numpy as np
from scipy.ndimage import minimum_filter1d

# ── NHTSA / FMVSS 안전 기준선 ────────────────────────────────────────────
HIC_SAFE               = 700.0
CHEST_G_SAFE           = 60.0    # 흉부 최대 합성가속도 (g)
CHEST_3MS_SAFE         = 60.0    # 흉부 3ms 클립 (g)            — 현재 보상함수 미사용 (chest_g 중복), 로깅용 유지
CHEST_COMPRESSION_SAFE = 50.0    # 흉부 압축량 (mm)             — 현재 보상함수 미사용 (측정 한계), 로깅용 유지
FEMUR_SAFE             = 10_000.0 # 대퇴부 압축력 (N)            — 현재 무릎 에어백 미구현으로 보상함수에서 제외, 로깅용 유지
NIJ_SAFE               = 1.0     # 목 상해 지수 Nij              — 현재 보상함수 미사용 (HIC15 중복), 로깅용 유지

# ── Hybrid III 참조값 (NHTSA 표준) ──────────────────────────────────────
HEAD_MASS_KG      = 4.54     # 두부 질량
NECK_LEVER_ARM_M  = 0.105    # 두부 무게중심 → 후두과 거리
NIJ_FZC_TENSION   = 6806.0  # N  (목 인장)
NIJ_FZC_COMPRESS  = 6160.0  # N  (목 압축)
NIJ_MYC_EXTENSION = 310.0   # N·m (신전 모멘트)
NIJ_MYC_FLEXION   = 135.0   # N·m (굴곡 모멘트)

THIGH_MASS_KG = 8.55  # Hybrid III 우측 대퇴부 질량 — 현재 무릎 에어백 미구현으로 보상함수에서 제외함

# ── 보상 스케일 ──────────────────────────────────────────────────────────
_VIOLATION_COEFF = 5.0
_SAFETY_BONUS    = 2.0
_NO_DEPLOY_PEN   = 2.0


# ══════════════════════════════════════════════════════════════════════════
# 1. 데이터 수집기
# ══════════════════════════════════════════════════════════════════════════

class InjuryDataCollector:
    """
    physics_callback (1ms)으로 호출됨.
    속도 → 수치미분 → 가속도 누적.

    등록:
        collector = InjuryDataCollector(human)
        world.add_physics_callback("collect_injury", collector.physics_callback)
    """

    def __init__(self, human=None, vehicle_body=None, physics_dt: float = 0.001):
        self.human        = human
        self.vehicle_body = vehicle_body  # 차량 기준 상대 torso 위치 계산용
        self.physics_dt   = physics_dt
        self.reset()

    def physics_callback(self, step_size: float):
        if self.human is None or self.human.articulation is None:
            return

        head_vel  = self.human.get_head_velocity()
        torso_vel = self.human.get_torso_velocity()
        torso_pos = self.human.get_torso_position()
        thigh_vel = self.human.get_thigh_velocity_3d()

        # NaN/Inf 가드: 초기화 직후 또는 물리 폭발 시 skip
        if not (np.all(np.isfinite(head_vel)) and np.all(np.isfinite(torso_vel))):
            return
        # 속도 크기 가드: 물리 솔버 폭발(대형 유한 수) 방지
        _MAX_VEL = 200.0  # 720 km/h 이상이면 물리 발산 상태
        if np.linalg.norm(head_vel) > _MAX_VEL or np.linalg.norm(torso_vel) > _MAX_VEL:
            return

        # chest_compression: 차량 기준 상대 torso 위치 저장
        # get_world_pose()는 physics callback에서 실패할 수 있으므로
        # get_linear_velocity()로 차량 위치를 적분 (apply_seatbelt에서도 동일하게 성공)
        if self.vehicle_body is not None:
            try:
                veh_vel = np.asarray(self.vehicle_body.get_linear_velocity(), dtype=np.float32)
                if np.all(np.isfinite(veh_vel)):
                    self._vehicle_int_pos += veh_vel * step_size
            except Exception:
                pass
        torso_pos = torso_pos - self._vehicle_int_pos

        self.record(
            head_vel  = head_vel,
            torso_vel = torso_vel,
            torso_pos = torso_pos,
            dt        = step_size,
            thigh_vel = thigh_vel,
        )

    def record(
        self,
        head_vel:  np.ndarray,
        torso_vel: np.ndarray,
        torso_pos: np.ndarray,
        dt:        float,
        thigh_vel: np.ndarray = None,
    ):
        if self._prev_head_vel is not None and dt > 0:
            head_acc  = (head_vel  - self._prev_head_vel)  / dt
            torso_acc = (torso_vel - self._prev_torso_vel) / dt

            self.head_acc_3d.append(head_acc.copy())
            self.head_acc_g.append(float(np.linalg.norm(head_acc)  / 9.81))
            self.torso_acc_g.append(float(np.linalg.norm(torso_acc) / 9.81))
            self.torso_pos_history.append(torso_pos.copy())

            if thigh_vel is not None and self._prev_thigh_vel is not None:
                thigh_acc = (thigh_vel - self._prev_thigh_vel) / dt
                self.thigh_acc_3d.append(thigh_acc.copy())

        self._prev_head_vel  = head_vel.copy()
        self._prev_torso_vel = torso_vel.copy()
        self._prev_thigh_vel = thigh_vel.copy() if thigh_vel is not None else None

    def reset(self):
        self.head_acc_3d:       list = []
        self.head_acc_g:        list = []
        self.torso_acc_g:       list = []
        self.torso_pos_history: list = []
        self.thigh_acc_3d:      list = []
        self._prev_head_vel  = None
        self._prev_torso_vel = None
        self._prev_thigh_vel = None
        self._vehicle_int_pos = np.zeros(3, dtype=np.float32)  # 차량 속도 적분 위치


# ══════════════════════════════════════════════════════════════════════════
# 2. HIC15 슬라이딩 윈도우 (dt=1ms → window_steps=15)
# ══════════════════════════════════════════════════════════════════════════

def compute_hic15(acc_g: list, dt: float) -> float:
    """
    HIC15 = max_{t2-t1≤15ms} [ (t2-t1) × (mean_a[t1:t2])^2.5 ]
    누적합으로 O(N) 계산. dt=0.001s → window_steps=15.
    """
    arr = np.asarray(acc_g, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0
    window_steps = max(int(round(0.015 / dt)), 1)
    cum = np.concatenate([[0.0], np.cumsum(arr) * dt])
    hic_max = 0.0
    for i in range(n):
        j = min(i + window_steps, n)
        dt_window = (j - i) * dt
        if dt_window <= 0:
            continue
        mean_acc = (cum[j] - cum[i]) / dt_window
        hic = dt_window * (max(mean_acc, 0.0) ** 2.5)
        if hic > hic_max:
            hic_max = hic
    return float(hic_max)


# ══════════════════════════════════════════════════════════════════════════
# 3. 신체 부위별 지표
# ══════════════════════════════════════════════════════════════════════════

def compute_chest_g(torso_acc_g: list) -> float:
    """흉부 최대 합성 가속도 (g). 기준 60g."""
    return float(max(torso_acc_g)) if torso_acc_g else 0.0


def compute_chest_3ms_clip(torso_acc_g: list, dt: float) -> float:
    """
    흉부 3ms 클립 가속도: 최소 3ms 연속 지속되는 최고 가속도 (g).
    슬라이딩 윈도우 최솟값의 최댓값으로 구현 (scipy.ndimage).

    현재 보상함수에서는 미사용 (chest_g와 동일 원천 데이터 파생, 중복성 높음).
    로깅/분석용으로만 유지.
    """
    arr = np.asarray(torso_acc_g, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    min_steps = max(int(round(0.003 / dt)), 1)
    if len(arr) < min_steps:
        return float(arr.max())
    windowed_min = minimum_filter1d(arr, size=min_steps, mode='nearest')
    return float(windowed_min.max())


def compute_chest_compression_mm(pos_history: list) -> float:
    """
    흉부 압축량 근사 (mm).
    pos_history 에는 차량 기준 상대 torso 위치가 담겨 있음
    (InjuryDataCollector.physics_callback 에서 vehicle_pos 를 차감하여 저장).

    알려진 한계: 차량의 탄성 반동(bounce)이 발생하면 vehicle_int_pos가 역방향으로
    적분되어 후방 변위까지 압축량으로 잘못 측정될 수 있음.
    보상함수 미사용 및 발표 자료 미포함으로 이번 프로젝트 범위에서는 수정하지 않음.
    """
    if len(pos_history) < 2:
        return 0.0
    positions = np.stack(pos_history)
    disp = positions - positions[0]
    return float(np.abs(disp[:, 0]).max() * 1000.0)


def compute_nij(head_acc_3d: list) -> float:
    """
    Nij = |Fz/Fzc| + |My/Myc|
    Fz = m_head × az (축방향 인장/압축력)
    My = m_head × ax × lever_arm (시상면 굽힘 모멘트)

    NHTSA FMVSS 208 표준. 기준: 1.0 이하.

    현재 보상함수에서는 미사용 (HIC15와 동일 원천 데이터 파생, 중복성 높음).
    로깅/분석용으로만 유지.
    """
    if not head_acc_3d:
        return 0.0
    nij_max = 0.0
    for acc_vec in head_acc_3d:
        az = float(acc_vec[2])
        ax = float(acc_vec[0])
        Fz = HEAD_MASS_KG * az
        My = HEAD_MASS_KG * ax * NECK_LEVER_ARM_M
        Fzc = NIJ_FZC_TENSION   if Fz >= 0 else NIJ_FZC_COMPRESS
        Myc = NIJ_MYC_EXTENSION if My <= 0 else NIJ_MYC_FLEXION
        nij = abs(Fz / Fzc) + abs(My / Myc)
        nij_max = max(nij_max, nij)
    return float(nij_max)


def compute_femur_force_n(thigh_acc_3d: list) -> float:
    """
    대퇴부 축방향 최대 압축력 (N).
    F = m_thigh × |a_thigh|. 기준: 10,000 N.
    """
    if not thigh_acc_3d:
        return 0.0
    return float(max(THIGH_MASS_KG * np.linalg.norm(a) for a in thigh_acc_3d))


# ══════════════════════════════════════════════════════════════════════════
# 4. RL 보상 함수
# ══════════════════════════════════════════════════════════════════════════

def compute_reward(
    hic15:        float,
    chest_g:      float,
    deploy_flags: list = None,
) -> float:
    """
    에피소드 종료 시 전체 이력 기반 터미널 보상.
    base      = -∑(val/safe)²  — 연속 gradient, 기준 근접 시 관대
    violation = -5 × (초과율)² — 기준 초과 시 가속적 패널티 (선형→이차)
    bonus     = +2.0            — 2개 지표 전부 기준 이하
    no_deploy = -2.0            — 에어백 미전개

    안전 지표: HIC15, chest_g (총 2개)
    제외 지표(로깅용 유지, 보상 계산 제외):
      - Nij            : HIC15와 동일 원천 파생, 중복
      - chest_3ms      : chest_g와 동일 원천 파생, 중복
      - chest_compression_mm : 측정 한계 (차량 탄성 반동 측정 오류)
      - femur_force    : 무릎 에어백 미구현
    """
    if not (np.isfinite(hic15) and np.isfinite(chest_g)):
        return -1000.0

    metrics = [
        (hic15,   HIC_SAFE),
        (chest_g, CHEST_G_SAFE),
    ]

    r = sum(-(val / safe) ** 2 for val, safe in metrics)

    for val, safe in metrics:
        if val > safe:
            excess = val / safe - 1.0
            r -= _VIOLATION_COEFF * excess ** 2

    if all(val <= safe for val, safe in metrics):
        r += _SAFETY_BONUS

    if deploy_flags is not None and sum(deploy_flags) == 0:
        r -= _NO_DEPLOY_PEN

    return float(r)


def compute_step_reward(
    head_acc_g:   list,
    torso_acc_g:  list,
    dt:           float,
    deploy_flags: list = None,
    n_steps:      int  = 60,
) -> float:
    """
    컨트롤 스텝 1개(~16ms 윈도우) 기반 중간 보상.
    compute_reward()와 동일한 이차 패널티, 1/n_steps 스케일링.
    전체 누적 시 터미널 보상과 유사한 크기 유지.

    안전 지표: HIC15, chest_g (총 2개)
    제외 지표(compute_reward 참조)는 여기서도 동일하게 미사용.
    """
    if not head_acc_g and not torso_acc_g:
        return 0.0

    hic15   = compute_hic15(head_acc_g, dt)
    chest_g = float(max(torso_acc_g)) if torso_acc_g else 0.0

    if not (np.isfinite(hic15) and np.isfinite(chest_g)):
        return 0.0

    metrics = [
        (hic15,   HIC_SAFE),
        (chest_g, CHEST_G_SAFE),
    ]

    scale = 1.0 / n_steps
    r = sum(-(val / safe) ** 2 for val, safe in metrics) * scale

    for val, safe in metrics:
        if val > safe:
            excess = val / safe - 1.0
            r -= _VIOLATION_COEFF * excess ** 2 * scale

    if deploy_flags is not None and sum(deploy_flags) == 0:
        r -= _NO_DEPLOY_PEN * scale

    return float(r)
