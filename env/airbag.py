"""
에어백 시스템: 팽창 구체 + 역 U 곡선 감쇠 모델.

에어백 sphere 구성:
  - 시각적 구체(VisualSphere) — 충돌 물리 없음, 순수 시각
  - 반지름은 INFLATE_MS(50 ms) 동안 선형 성장 → 이후 max_radius 로 고정
  - 에너지 흡수는 _apply_damping() 단일 경로 (역 U 곡선 기반 속도 감쇠)

prim 경로 /World/vehicle/airbag_<i> → vehicle USD 자식 계층으로 차량 추종.
"""

import numpy as np
import omni.usd
from pxr import UsdGeom
from isaacsim.core.api.objects import VisualSphere

AIRBAG_SPECS = {
    0: {"name": "front_driver",    "volume_L": 60,  "k": 1.0,
        "protects": ["head","torso"], "angle_range": (315,45),  "max_radius": 0.30, "color": np.array([0.95,0.88,0.75])},
    1: {"name": "front_passenger", "volume_L": 120, "k": 0.5,
        "protects": ["head","torso"], "angle_range": (315,45),  "max_radius": 0.35, "color": np.array([0.95,0.88,0.75])},
    2: {"name": "side_driver",     "volume_L": 15,  "k": 4.0,
        "protects": ["torso"],        "angle_range": (225,315), "max_radius": 0.22, "color": np.array([0.90,0.82,0.70])},
    3: {"name": "side_passenger",  "volume_L": 15,  "k": 4.0,
        "protects": ["torso"],        "angle_range": (45,135),  "max_radius": 0.22, "color": np.array([0.90,0.82,0.70])},
    4: {"name": "curtain",         "volume_L": 40,  "k": 1.5,
        "protects": ["head"],         "angle_range": (45,315),  "max_radius": 0.28, "color": np.array([0.92,0.85,0.72])},
}

AIRBAG_LOCAL_POSITIONS = {
    0: np.array([0.65,  0.50, 1.05]),
    1: np.array([0.65, -0.50, 1.05]),
    2: np.array([0.10,  0.95, 1.00]),
    3: np.array([0.10, -0.95, 1.00]),
    4: np.array([0.10,  0.65, 1.50]),
}

PRESSURE_OPT = 300.0
DAMPING_BASE = 0.75
INFLATE_MS   = 50.0
BASE_FORCE   = 500.0


def _reverse_u_curve(pressure_kpa: float) -> float:
    if pressure_kpa <= 0:
        return 0.0
    x = pressure_kpa / PRESSURE_OPT
    return float(np.clip(DAMPING_BASE * (2 * x) / (1 + x ** 2), 0.0, DAMPING_BASE))


def _is_effective_angle(airbag_idx: int, collision_angle: float) -> bool:
    lo, hi = AIRBAG_SPECS[airbag_idx]["angle_range"]
    if lo > hi:
        return collision_angle >= lo or collision_angle <= hi
    return lo <= collision_angle <= hi


class AirbagSystem:
    def __init__(self, world, human):
        self.human = human
        self._sphere_prims   = {}
        self._inflate_start  = {}
        self._fully_inflated = set()

        stage = omni.usd.get_context().get_stage()

        for i, spec in AIRBAG_SPECS.items():
            path = f"/World/vehicle/airbag_{i}"
            name = f"airbag_{i}"
            if not stage.GetPrimAtPath(path).IsValid() and not world.scene.object_exists(name):
                world.scene.add(
                    VisualSphere(
                        prim_path=path,
                        name=name,
                        position=AIRBAG_LOCAL_POSITIONS[i],
                        radius=0.001,
                        color=spec["color"],
                    )
                )
            self._sphere_prims[i] = UsdGeom.Sphere(stage.GetPrimAtPath(path))

    def apply(self, actions: np.ndarray, collision_angle: float, current_ms: float):
        for i, spec in AIRBAG_SPECS.items():
            deploy    = actions[i, 0] > 0.5
            timing_ms = actions[i, 1]
            pressure  = actions[i, 2]

            if not deploy or current_ms < timing_ms or not _is_effective_angle(i, collision_angle):
                if i not in self._fully_inflated:
                    self._set_radius(i, 0.001)
                    self._inflate_start.pop(i, None)
                continue

            if i not in self._inflate_start:
                self._inflate_start[i] = current_ms

            if i in self._fully_inflated:
                damping = spec["k"] * _reverse_u_curve(pressure)
                self._apply_damping(i, damping, spec)
                continue

            elapsed       = current_ms - self._inflate_start[i]
            inflate_ratio = min(elapsed / INFLATE_MS, 1.0)
            self._set_radius(i, spec["max_radius"] * inflate_ratio)

            if inflate_ratio >= 1.0:
                self._fully_inflated.add(i)

            damping = spec["k"] * _reverse_u_curve(pressure)
            if inflate_ratio >= 1.0:
                self._apply_damping(i, damping, spec)

    def reset(self):
        self._inflate_start.clear()
        self._fully_inflated.clear()
        for i in self._sphere_prims:
            self._set_radius(i, 0.001)

    def _set_radius(self, idx: int, radius: float):
        self._sphere_prims[idx].GetRadiusAttr().Set(max(float(radius), 0.001))

    def _apply_damping(self, idx: int, damping: float, spec: dict):
        human = self.human
        if human is None:
            return
        scale = damping * BASE_FORCE
        for part in spec["protects"]:
            if part == "head":
                vel      = human.get_head_velocity()
                link_idx = human._head_idx
            elif part == "torso":
                vel      = human.get_torso_velocity()
                link_idx = human._torso_idx
            else:
                continue
            force = np.array([-vel[0] * scale, -vel[1] * scale, -vel[2] * scale],
                             dtype=np.float32)
            human._apply_link_force(link_idx, force)
