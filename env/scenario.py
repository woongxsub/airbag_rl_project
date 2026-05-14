import numpy as np


STIFFNESS_OPTIONS = ["concrete", "vehicle", "wood"]


class ScenarioSampler:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def sample(self):
        angle = self.rng.uniform(0, 360)
        speed = self.rng.uniform(20, 120)
        stiffness = self.rng.choice(STIFFNESS_OPTIONS)
        seatbelt = bool(self.rng.integers(0, 2))

        # ToF 체형 변수 (프로토타입: 랜덤 샘플링으로 대체)
        height = self.rng.uniform(1.55, 1.90)
        weight = self.rng.uniform(50.0, 100.0)
        head_offset = self.rng.uniform(-0.05, 0.05)  # 머리 전후 위치 편차

        return {
            "angle": angle,
            "speed": speed,
            "stiffness": stiffness,
            "seatbelt": seatbelt,
            "height": height,
            "weight": weight,
            "head_offset": head_offset,
        }

    def to_state_vector(self, scenario: dict) -> np.ndarray:
        stiffness_enc = {"concrete": 1.0, "vehicle": 0.7, "wood": 0.4}
        return np.array([
            scenario["angle"] / 360.0,
            scenario["speed"] / 120.0,
            stiffness_enc[scenario["stiffness"]],
            float(scenario["seatbelt"]),
            scenario["height"] / 1.90,
            scenario["weight"] / 100.0,
            scenario["head_offset"] / 0.05,
        ], dtype=np.float32)
