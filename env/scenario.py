"""
시나리오 샘플러 및 State 벡터 인코더.

State 12차원 설계:
  [0]  충돌 방향   angle / 360
  [1]  충돌 속도   speed / 120
  [2]  충돌 강성   concrete=1.0 / vehicle=0.7 / wood=0.4
  [3]  안전벨트    0 or 1
  [4]  신장        height / 2.0
  [5]  앉은키(실측) sitting_height / 1.2  (착좌 후 head_z - seat_z)
  [6]  머리 X      head_pos[0] / 1.5     (차량 로컬 전후)
  [7]  머리 Y      (head_pos[1] + 1.0) / 2.0  (차량 로컬 좌우)
  [8]  머리 Z      head_pos[2] / 2.0     (차량 로컬 높이)
  [9]  척추 기울기  (spine_tilt_deg + 15) / 35  (-15°~+20° → 0~1)
  [10] 머리-스티어링 거리  head_to_steering / 1.0
  [11] 무릎-대시보드 거리  knee_to_dashboard / 0.5

[5]~[11] 은 에피소드 리셋 시 airbag_env 가 measure_snapshot() 으로 측정하여
scenario dict 에 추가한다. 값이 없으면 신장 기반 추정치로 대체.
"""

import numpy as np

STIFFNESS_OPTIONS = ["concrete", "vehicle", "wood"]
_STIFFNESS_ENC = {"concrete": 1.0, "vehicle": 0.7, "wood": 0.4}

# State 차원 (ppo.py / train.py 와 동기화)
STATE_DIM = 12

# 착좌 자세 척추 기울기 샘플 범위 (도)
SPINE_TILT_MIN_DEG = -10.0
SPINE_TILT_MAX_DEG =  20.0


class ScenarioSampler:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def sample(self) -> dict:
        """충돌 시나리오 + 체형 파라미터 샘플링."""
        return {
            "angle":    float(self.rng.uniform(0, 360)),
            "speed":    float(self.rng.uniform(20, 120)),
            "stiffness": str(self.rng.choice(STIFFNESS_OPTIONS)),
            "seatbelt": bool(self.rng.integers(0, 2)),
            "height":   float(self.rng.uniform(1.55, 1.90)),
            "weight":   float(self.rng.uniform(50.0, 100.0)),
        }

    def to_state_vector(self, scenario: dict) -> np.ndarray:
        """
        scenario dict → 12차원 정규화 벡터 [0, 1].
        measure_snapshot 결과([5]~[11])가 없으면 신장 기반 추정치 사용.
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
            _STIFFNESS_ENC[scenario["stiffness"]],
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
