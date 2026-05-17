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
CHEST_3MS_SAFE         = 60.0    # 흉부 3ms 클립 (g)
CHEST_COMPRESSION_SAFE = 50.0    # 흉부 압축량 (mm)
FEMUR_SAFE             = 10_000.0 # 대퇴부 압축력 (N)
NIJ_SAFE               = 1.0     # 목 상해 지수 Nij

# ── Hybrid III 참조값 (NHTSA 표준) ──────────────────────────────────────
HEAD_MASS_KG      = 4.54     # 두부 질량
NECK_LEVER_ARM_M  = 0.105    # 두부 무게중심 → 후두과 거리
NIJ_FZC_TENSION   = 6806.0  # N  (목 인장)
NIJ_FZC_COMPRESS  = 6160.0  # N  (목 압축)
NIJ_MYC_EXTENSION = 310.0   # N·m (신전 모멘트)
NIJ_MYC_FLEXION   = 135.0   # N·m (굴곡 모멘트)

THIGH_MASS_KG = 8.55  # Hybrid III 우측 대퇴부 질량

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

    def __init__(self, human=None, physics_dt: float = 0.001):
        self.human = human
        self.physics_dt = physics_dt
        self.reset()

    def physics_callback(self, step_size: float):
        if self.human is None or self.human.articulation is None:
            return
        thigh_vel = self.human.get_thigh_velocity_3d()
        self.record(
            head_vel  = self.human.get_head_velocity(),
            torso_vel = self.human.get_torso_velocity(),
            torso_pos = self.human.get_torso_position(),
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
        self.head_acc_3d:      list = []
        self.head_acc_g:       list = []
        self.torso_acc_g:      list = []
        self.torso_pos_history: list = []
        self.thigh_acc_3d:     list = []
        self._prev_head_vel   = None
        self._prev_torso_vel  = None
        self._prev_thigh_vel  = None


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
    """전후방향(x축) 최대 변위를 흉부 압축량으로 근사 (mm)."""
    if len(pos_history) < 2:
        return 0.0
    positions = np.stack(pos_history)
    return float(np.abs(positions[:, 0] - positions[0, 0]).max() * 1000.0)


def compute_nij(head_acc_3d: list) -> float:
    """
    Nij = |Fz/Fzc| + |My/Myc|
    Fz = m_head × az (축방향 인장/압축력)
    My = m_head × ax × lever_arm (시상면 굽힘 모멘트)

    NHTSA FMVSS 208 표준. 기준: 1.0 이하.
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
    hic15:                float,
    chest_g:              float,
    chest_3ms:            float = 0.0,
    chest_compression_mm: float = 0.0,
    femur_n:              float = 0.0,
    nij:                  float = 0.0,
    deploy_flags:         list  = None,
) -> float:
    """
    Dense reward 설계:
      base      = -(각 지표 / 기준선)  합산 — 항상 연속 gradient
      violation = -5 × 초과율          기준 초과 항목마다
      bonus     = +2.0                 전 항목 기준 이하
      no_deploy = -2.0                 에어백 미전개

    예시 (기준선 60%, 전 항목 안전):
      HIC=420, chest_g=36 → ≈ +1.3
    """
    metrics = [
        (hic15,                HIC_SAFE),
        (chest_g,              CHEST_G_SAFE),
        (chest_3ms,            CHEST_3MS_SAFE),
        (chest_compression_mm, CHEST_COMPRESSION_SAFE),
        (femur_n,              FEMUR_SAFE),
        (nij,                  NIJ_SAFE),
    ]

    r = sum(-(val / safe) for val, safe in metrics)

    for val, safe in metrics:
        if val > safe:
            r -= _VIOLATION_COEFF * (val / safe - 1.0)

    if all(val <= safe for val, safe in metrics):
        r += _SAFETY_BONUS

    if deploy_flags is not None and sum(deploy_flags) == 0:
        r -= _NO_DEPLOY_PEN

    return float(r)
