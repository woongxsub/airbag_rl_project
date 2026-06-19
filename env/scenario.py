"""
시나리오 샘플러 및 State 벡터 인코더.

State 11차원 설계 (실차 센서로 측정 가능한 값만):
  [0]  충돌 방향   angle / 360          (외부 레이더/카메라)
  [1]  충돌 속도   speed / 120          (차속 센서)
  [2]  안전벨트    0 or 1               (벨트 버클 센서)
  [3]  신장        height / 2.0         (cabin ToF 카메라)
  [4]  앉은키(실측) sitting_height / 1.2 (cabin ToF 카메라)
  [5]  머리 X      head_pos[0] / 1.5    (cabin ToF 카메라)
  [6]  머리 Y      (head_pos[1] + 1.0) / 2.0  (cabin ToF 카메라)
  [7]  머리 Z      head_pos[2] / 2.0    (cabin ToF 카메라)
  [8]  척추 기울기  (spine_tilt_deg + 15) / 35  (cabin ToF 카메라)
  [9]  머리-스티어링 거리  head_to_steering / 1.0  (cabin ToF 카메라)
  [10] 무릎-대시보드 거리  knee_to_dashboard / 0.5 (cabin ToF 카메라)

[4]~[10] 은 에피소드 리셋 시 airbag_env 가 measure_snapshot() 으로 측정하여
scenario dict 에 추가한다. 값이 없으면 신장 기반 추정치로 대체.
충돌 강성(stiffness)은 실차에서 사전 측정 불가 → 제외.
is_rollover / passenger_present 는 sample() 내부에서 유지되지만
State에는 포함하지 않음 — rule_based_policy() 인자로만 사용.
"""

import numpy as np

# State 차원 (ppo.py / train.py 와 동기화)
STATE_DIM = 11

# 착좌 자세 척추 기울기 샘플 범위 (도)
SPINE_TILT_MIN_DEG = -10.0
SPINE_TILT_MAX_DEG =  20.0


class ScenarioSampler:
    """
    충돌 시나리오 샘플러.

    stage 속성으로 커리큘럼 단계 제어 (외부에서 직접 변경):
      0 = Pure PPO — 전 범위 (기본값)
      1 = Stage 1  — 정면±45°, 30-60 km/h, 전복 없음   (관대)
      2 = Stage 2  — 전 각도,   30-90 km/h, 전복 5%    (중간, 누적 확장)
      3 = Stage 3  — 전 각도,  20-120 km/h, 전복 15%   (전 범위, 누적 확장)

    각도는 Stage 1(정면±45°)→ Stage 2/3(전 범위) 이분법적 확장.
    속도는 누적 확장형 (이전 범위 포함하며 상한 확대).
    """

    def __init__(self, seed=None):
        self.rng   = np.random.default_rng(seed)
        self.stage = 0  # 외부(train.py 루프)에서 에피소드마다 갱신

    def sample(self) -> dict:
        """충돌 시나리오 + 체형 파라미터 샘플링."""
        s = self.stage

        # ── 충돌 각도 ────────────────────────────────────────────────────
        if s == 1:
            # 정면 충돌: 315~360° 또는 0~45° (각 50% 확률)
            if self.rng.random() < 0.5:
                angle = float(self.rng.uniform(315.0, 360.0))
            else:
                angle = float(self.rng.uniform(0.0, 45.0))
        else:
            angle = float(self.rng.uniform(0.0, 360.0))

        # ── 충돌 속도 (누적 확장형) ──────────────────────────────────────
        if s == 1:
            speed = float(self.rng.uniform(30.0, 60.0))
        elif s == 2:
            speed = float(self.rng.uniform(30.0, 90.0))   # Stage 1 범위 포함
        else:
            speed = float(self.rng.uniform(20.0, 120.0))  # 전 범위 (Stage 3 & Pure)

        # ── 전복 확률 ────────────────────────────────────────────────────
        # 실차 rollover sensor(자이로/각속도계)에 대응.
        rollover_prob = {0: 0.05, 1: 0.00, 2: 0.05, 3: 0.15}.get(s, 0.05)

        return {
            "angle":             angle,
            "speed":             speed,
            "seatbelt":          bool(self.rng.integers(0, 2)),
            "height":            float(self.rng.uniform(1.55, 1.90)),
            "weight":            float(self.rng.uniform(50.0, 100.0)),
            "is_rollover":       bool(self.rng.random() < rollover_prob),
            # OCS(Occupant Classification System) 시트 센서에 대응.
            "passenger_present": bool(self.rng.integers(0, 2)),
        }

    def to_state_vector(self, scenario: dict) -> np.ndarray:
        """
        scenario dict → 11차원 정규화 벡터 [0, 1].
        measure_snapshot 결과([4]~[10])가 없으면 신장 기반 추정치 사용.
        """
        height = scenario["height"]

        # 실측 앉은키 / 폴백: 신장 × 0.52 (표준비율)
        sitting_height = scenario.get("sitting_height", height * 0.52)

        # 머리 로컬 위치 / 폴백: SEAT_LOCAL + 신장 추정 오프셋
        head_pos = np.asarray(scenario.get("head_pos", [0.3, 0.5, 0.9 + height * 0.52]))

        spine_tilt_deg    = float(scenario.get("spine_tilt_deg",    0.0))
        head_to_steering  = float(scenario.get("head_to_steering",  0.45))
        knee_to_dashboard = float(scenario.get("knee_to_dashboard", 0.30))

        return np.array([
            scenario["angle"] / 360.0,
            scenario["speed"] / 120.0,
            float(scenario["seatbelt"]),
            np.clip(height / 2.0, 0.0, 1.0),
            np.clip(sitting_height / 1.2, 0.0, 1.0),
            np.clip(head_pos[0] / 1.5, 0.0, 1.0),
            np.clip((head_pos[1] + 1.0) / 2.0, 0.0, 1.0),
            np.clip(head_pos[2] / 2.0, 0.0, 1.0),
            np.clip((spine_tilt_deg + 15.0) / 35.0, 0.0, 1.0),
            np.clip(head_to_steering  / 1.0, 0.0, 1.0),
            np.clip(knee_to_dashboard / 0.5, 0.0, 1.0),
        ], dtype=np.float32)
