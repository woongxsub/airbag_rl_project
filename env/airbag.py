"""
에어백 시스템: 팽창 구체 + 물리 충돌(kinematic soft-bag) + 체적 기반 감쇠.

에어백 sphere 구성:
  - 시각적 구체(VisualSphere) 위에 USD 물리 API를 직접 부여
  - kinematic rigid body → 에어백 자체는 힘에 의해 이동하지 않음
  - PhysxMaterialAPI compliant contact → 낮은 강성 + 높은 감쇠 ("말랑한 주머니")
  - 반지름은 INFLATE_MS(50 ms) 동안 선형 성장 → 이후 max_radius 로 고정

prim 경로 /World/vehicle/airbag_<i> → vehicle USD 자식 계층으로 차량 추종.
"""

import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, UsdShade, Sdf, PhysxSchema
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

PRESSURE_OPT  = 300.0
DAMPING_BASE  = 0.75
INFLATE_MS    = 50.0    # 50ms로 팽창 완료, 이후 반지름 고정 (FMVSS 참고값)
BASE_FORCE    = 500.0

# ── Soft-bag compliant contact 파라미터 ──────────────────────────────────
#   에어백 내부 압력 모델: F_contact = K * penetration + D * penetration_rate
#   K = 150,000 N/m  (팽팽한 쿠션, rigid wall의 1/100 이하)
#   D = 15,000 Ns/m  (과감쇠 → 반동 없이 에너지 흡수)
_CONTACT_STIFFNESS = 150_000.0   # N/m
_CONTACT_DAMPING   =  15_000.0   # Ns/m

# 재질 바인딩에 쓸 공유 material prim 경로 (에어백 공통)
_BAG_MATERIAL_PATH = "/World/airbag_soft_material"


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


def _create_soft_material(stage) -> bool:
    """
    에어백 공용 soft-bag 물리 재질을 USD stage에 등록.
    UsdPhysics.MaterialAPI + PhysxSchema.PhysxMaterialAPI compliant contact.
    반환: True=성공, False=PhysxSchema 없음(fallback)
    """
    if stage.GetPrimAtPath(_BAG_MATERIAL_PATH).IsValid():
        return True

    mat_prim = stage.DefinePrim(Sdf.Path(_BAG_MATERIAL_PATH), "Material")

    # 기본 USD 물리 재질 (반발 없음, 저마찰)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat.CreateRestitutionAttr().Set(0.0)        # 반발계수 0 (반동 없음)
    phys_mat.CreateStaticFrictionAttr().Set(0.05)
    phys_mat.CreateDynamicFrictionAttr().Set(0.05)

    # PhysX compliant contact (soft spring-damper contact)
    try:
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
        physx_mat.CreateCompliantContactStiffnessAttr().Set(_CONTACT_STIFFNESS)
        physx_mat.CreateCompliantContactDampingAttr().Set(_CONTACT_DAMPING)
        return True
    except Exception as e:
        print(f"[Airbag] PhysxMaterialAPI compliant contact unavailable: {e}")
        return False


def _apply_physics_to_sphere(stage, sphere_path: str):
    """
    기존 sphere prim에 kinematic rigid body + soft collision을 부여.
    - kinematic: 에어백 자체는 외력에 반응하지 않음
    - CollisionAPI: 인체와 물리 접촉 가능
    - 재질 바인딩: soft-bag compliant contact
    """
    prim = stage.GetPrimAtPath(sphere_path)
    if not prim.IsValid():
        return

    # 충돌 활성화
    UsdPhysics.CollisionAPI.Apply(prim)

    # Kinematic rigid body (에어백이 힘에 의해 날아가지 않음)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(True)

    # Soft material 바인딩
    mat_prim = stage.GetPrimAtPath(_BAG_MATERIAL_PATH)
    if mat_prim.IsValid():
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(
            UsdShade.Material(mat_prim),
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )


class AirbagSystem:
    def __init__(self, world, human):
        self.human = human
        self._sphere_prims  = {}
        self._inflate_start = {}
        self._fully_inflated = set()   # 반지름 고정 완료 인덱스

        stage = omni.usd.get_context().get_stage()

        # 공용 soft material 등록 (1회)
        _create_soft_material(stage)

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
            # 물리 충돌 + kinematic + soft material 부여
            _apply_physics_to_sphere(stage, path)
            self._sphere_prims[i] = UsdGeom.Sphere(stage.GetPrimAtPath(path))

    def apply(self, actions: np.ndarray, collision_angle: float, current_ms: float):
        for i, spec in AIRBAG_SPECS.items():
            deploy    = actions[i, 0] > 0.5
            timing_ms = actions[i, 1]
            pressure  = actions[i, 2]

            if not deploy or current_ms < timing_ms or not _is_effective_angle(i, collision_angle):
                # 미전개: 반지름 0으로 초기화
                if i not in self._fully_inflated:
                    self._set_radius(i, 0.001)
                    self._inflate_start.pop(i, None)
                continue

            if i not in self._inflate_start:
                self._inflate_start[i] = current_ms

            # 반지름 고정 이후에는 visual 업데이트 불필요
            if i in self._fully_inflated:
                damping = spec["k"] * _reverse_u_curve(pressure)
                self._apply_damping(i, damping, spec)
                continue

            elapsed       = current_ms - self._inflate_start[i]
            inflate_ratio = min(elapsed / INFLATE_MS, 1.0)
            self._set_radius(i, spec["max_radius"] * inflate_ratio)

            if inflate_ratio >= 1.0:
                self._fully_inflated.add(i)   # 이후 반지름 고정

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
        """프로그래밍 감쇠력: 물리 충돌 외에도 추가 에너지 흡수."""
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
