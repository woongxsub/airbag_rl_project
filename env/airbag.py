import numpy as np

# 에어백별 보호 부위 및 유효 충돌 각도 (AGENT.md 15섹션)
AIRBAG_PROFILE = {
    0: {"name": "front_driver",    "protects": ["head", "torso"], "angle_range": (315, 45)},
    1: {"name": "front_passenger", "protects": ["head", "torso"], "angle_range": (315, 45)},
    2: {"name": "side_driver",     "protects": ["torso"],         "angle_range": (225, 315)},
    3: {"name": "side_passenger",  "protects": ["torso"],         "angle_range": (45, 135)},
    4: {"name": "curtain",         "protects": ["head"],          "angle_range": (45, 315)},
}

# 압력-감쇠율 역U자 곡선 파라미터 (추후 논문 수치로 교체 예정)
PRESSURE_OPT = 300.0   # kPa, 감쇠 최대 지점
PRESSURE_MAX = 600.0   # kPa, 이 이상이면 감쇠 감소
DAMPING_MAX = 0.75     # 최대 감쇠율


def damping_rate(pressure_kpa: float) -> float:
    """역U자 곡선: 압력 → 감쇠율 [0, DAMPING_MAX]"""
    if pressure_kpa <= 0:
        return 0.0
    x = pressure_kpa / PRESSURE_OPT
    rate = DAMPING_MAX * (2 * x) / (1 + x ** 2)
    return float(np.clip(rate, 0.0, DAMPING_MAX))


def is_effective_angle(airbag_idx: int, collision_angle: float) -> bool:
    lo, hi = AIRBAG_PROFILE[airbag_idx]["angle_range"]
    if lo > hi:  # 0도 걸치는 경우 (예: 315~45)
        return collision_angle >= lo or collision_angle <= hi
    return lo <= collision_angle <= hi


class AirbagSystem:
    def __init__(self, human):
        self.human = human

    def apply(self, actions: np.ndarray, collision_angle: float):
        """
        actions: shape (5, 3) → [deploy(0/1), timing(ms), pressure(kPa)]
        timing은 이 함수 호출 전 시뮬레이션 루프에서 처리.
        여기서는 deploy=1이고 현재 타이밍에 맞는 에어백만 감쇠력 적용.
        """
        for i in range(5):
            deploy = actions[i, 0] > 0.5
            pressure = actions[i, 2]

            if not deploy:
                continue
            if not is_effective_angle(i, collision_angle):
                continue

            rate = damping_rate(pressure)
            self._apply_damping(i, rate)

    def _apply_damping(self, airbag_idx: int, rate: float):
        profile = AIRBAG_PROFILE[airbag_idx]
        for part in profile["protects"]:
            if part == "head":
                vel = self.human.head.get_linear_velocity()
                self.human.head.apply_force_torque(force=-vel * rate * 500.0)
            elif part == "torso":
                vel = self.human.torso.get_linear_velocity()
                self.human.torso.apply_force_torque(force=-vel * rate * 500.0)
