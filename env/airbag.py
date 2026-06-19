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
from isaacsim.core.utils.rotations import quat_to_rot_matrix

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

# 에어백 감쇠력 상한선
# 출처: 운동량-충격량 원리 기반 추정 (Quora, 70kg 탑승자 기준
#   가슴 속도변화 0.5~1.0m/s, 충돌시간 0.03s → 평균 반력 1,200~2,300N,
#   피크 반력 2,000~6,000N). 고속 충돌 안전 마진 포함 중상단값 적용.
#   안전벨트 로드 리미터(15,000N)보다 낮게 설정 (에어백 쿠션 특성 반영).
FORCE_CAP = 9_000.0  # N


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
    def __init__(self, world, human, vehicle=None):
        self.human   = human
        self.vehicle = vehicle   # 에어백 local→world 변환 및 거리 게이트용
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
        """Control step (60Hz): 팽창 타이밍 추적 + 반지름 visual 업데이트만."""
        for i, spec in AIRBAG_SPECS.items():
            deploy    = actions[i, 0] > 0.5
            timing_ms = actions[i, 1]

            if not deploy or current_ms < timing_ms or not _is_effective_angle(i, collision_angle):
                if i not in self._fully_inflated:
                    self._set_radius(i, 0.001)
                    self._inflate_start.pop(i, None)
                continue

            if i not in self._inflate_start:
                self._inflate_start[i] = current_ms

            if i in self._fully_inflated:
                continue

            elapsed       = current_ms - self._inflate_start[i]
            inflate_ratio = min(elapsed / INFLATE_MS, 1.0)
            self._set_radius(i, spec["max_radius"] * inflate_ratio)

            if inflate_ratio >= 1.0:
                self._fully_inflated.add(i)

    def apply_forces(self, actions: np.ndarray, collision_angle: float, current_ms: float):
        """Physics callback (1000Hz): 감쇠력 인가만. USD 업데이트 없음."""
        # 차량 world pose 취득 — 에어백 local→world 변환 (1ms마다 1회)
        rot        = None
        veh_origin = None
        if self.vehicle is not None:
            try:
                veh_pos, veh_quat = self.vehicle.body.get_world_pose()
                rot        = quat_to_rot_matrix(np.asarray(veh_quat))
                veh_origin = np.asarray(veh_pos)
            except Exception:
                pass

        for i, spec in AIRBAG_SPECS.items():
            deploy    = actions[i, 0] > 0.5
            timing_ms = actions[i, 1]
            pressure  = actions[i, 2]

            if not deploy or current_ms < timing_ms or not _is_effective_angle(i, collision_angle):
                continue
            if i not in self._inflate_start:
                continue

            elapsed       = current_ms - self._inflate_start[i]
            inflate_ratio = min(elapsed / INFLATE_MS, 1.0)
            if inflate_ratio <= 0:
                continue

            damping = spec["k"] * _reverse_u_curve(pressure) * inflate_ratio

            # 에어백 월드 좌표 (vehicle=None이면 게이트 비활성 — 폴백)
            airbag_world = None
            if rot is not None:
                airbag_world = veh_origin + rot @ AIRBAG_LOCAL_POSITIONS[i]

            self._apply_damping(i, damping, spec, airbag_world, inflate_ratio)

    def reset(self):
        self._inflate_start.clear()
        self._fully_inflated.clear()
        for i in self._sphere_prims:
            self._set_radius(i, 0.001)

    def _set_radius(self, idx: int, radius: float):
        self._sphere_prims[idx].GetRadiusAttr().Set(max(float(radius), 0.001))

    def _apply_damping(
        self,
        idx: int,
        damping: float,
        spec: dict,
        airbag_world: np.ndarray = None,
        inflate_ratio: float = 1.0,
    ):
        """
        에어백 감쇠력 인가.

        airbag_world : 에어백 중심의 월드 좌표 (None이면 거리 게이트 비활성).
        inflate_ratio: 현재 팽창 비율 [0,1] — 현재 에어백 반경 계산에 사용.

        거리 게이트 — 부위별 독립 판정:
          head/torso 각각 '현재 에어백 반경(max_radius * inflate_ratio) 이내'
          일 때만 해당 부위에 힘 인가. 두 부위가 서로 다른 타이밍에 접촉해도
          자연스럽게 처리됨.

        Force cap — apply_seatbelt의 F_BELT_CAP 패턴과 동일:
          force 벡터의 norm을 FORCE_CAP(9,000N)으로 비율 스케일 다운.
        """
        human = self.human
        if human is None:
            return
        scale = damping * BASE_FORCE

        for part in spec["protects"]:
            if part == "head":
                vel      = human.get_head_velocity()
                link_idx = human._head_idx
                body_pos = human.get_head_position()
            elif part == "torso":
                vel      = human.get_torso_velocity()
                link_idx = human._torso_idx
                body_pos = human.get_torso_position()
            else:
                continue

            # NaN 가드: 초기화 직후 physics_view 값이 유효하지 않을 때
            if not np.all(np.isfinite(vel)):
                continue

            # ── 거리 기반 게이트 ─────────────────────────────────────────
            # 신체 부위(world)와 에어백 중심(world) 간 거리가
            # 현재 에어백 반경(max_radius * inflate_ratio) 초과 시 스킵.
            # 모든 좌표 world space — head/torso는 physics tensors 기반,
            # airbag_world는 vehicle pose로 변환된 값.
            if airbag_world is not None:
                dist = float(np.linalg.norm(np.asarray(body_pos) - airbag_world))
                if dist > spec["max_radius"] * inflate_ratio:
                    continue

            force = np.array(
                [-vel[0] * scale, -vel[1] * scale, -vel[2] * scale],
                dtype=np.float32,
            )

            # ── Force cap ────────────────────────────────────────────────
            norm = float(np.linalg.norm(force))
            if norm > FORCE_CAP:
                force = force * (FORCE_CAP / norm)

            human._apply_link_force(link_idx, force)
