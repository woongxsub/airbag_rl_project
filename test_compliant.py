"""
compliant contact 검증 스크립트.
1. PhysxMaterialAPI 속성 확인
2. 차량(60km/h) → 벽 충돌 with compliant contact → NaN 없는지 확인
"""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import carb
carb.settings.get_settings().set("/physics/cudaDevice", 0)

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdPhysics, PhysxSchema, UsdShade, Sdf
import omni.usd

world = World(physics_dt=0.001, rendering_dt=1/60, stage_units_in_meters=1.0)

# ── 1. PhysxMaterialAPI 속성 확인 ─────────────────────────────────────────
print("\n=== [1] PhysxMaterialAPI 속성 목록 ===")
all_attrs = [a for a in dir(PhysxSchema.PhysxMaterialAPI) if not a.startswith('_')]
compliant = [a for a in all_attrs if any(k in a.lower() for k in ['compliant', 'stiffness', 'damping', 'restitution'])]
print("compliant 관련:", compliant)

# ── 2. compliant material 생성 ────────────────────────────────────────────
print("\n=== [2] compliant material 생성 ===")
stage = omni.usd.get_context().get_stage()

mat_prim = stage.DefinePrim("/World/VehicleMat", "Material")
phys_mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
phys_mat_api.CreateRestitutionAttr(0.0)
phys_mat_api.CreateStaticFrictionAttr(0.3)
phys_mat_api.CreateDynamicFrictionAttr(0.3)

physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)

# compliant contact 속성 적용 시도
stiffness_set = False
damping_set   = False

# 방법 1: Create* 메서드
for method, val, flag_name in [
    ("CreateCompliantContactStiffnessAttr", 2e6,  "stiffness"),
    ("CreateCompliantContactDampingAttr",   1e4,  "damping"),
]:
    fn = getattr(physx_mat_api, method, None)
    if fn:
        fn(val)
        print(f"  {method}({val:.0e}) → 성공")
        if flag_name == "stiffness": stiffness_set = True
        else: damping_set = True
    else:
        print(f"  {method} 없음")

# 방법 2: USD 직접 속성 (fallback)
if not stiffness_set:
    try:
        a = physx_mat_api.GetPrim().CreateAttribute(
            "physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float)
        a.Set(2e6)
        stiffness_set = True
        print("  compliantContactStiffness (direct USD) 설정 성공")
    except Exception as e:
        print(f"  direct stiffness 실패: {e}")

if not damping_set:
    try:
        a = physx_mat_api.GetPrim().CreateAttribute(
            "physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float)
        a.Set(1e4)
        damping_set = True
        print("  compliantContactDamping (direct USD) 설정 성공")
    except Exception as e:
        print(f"  direct damping 실패: {e}")

print(f"stiffness 설정: {stiffness_set}, damping 설정: {damping_set}")

# ── 3. 씬 구성: 차량 박스 + 벽 ─────────────────────────────────────────
print("\n=== [3] 씬 구성 ===")
world.scene.add(GroundPlane(prim_path="/World/Ground", name="ground", z_position=0.0))

wall = world.scene.add(FixedCuboid(
    prim_path="/World/wall",
    name="wall",
    position=np.array([3.5, 0.0, 0.75]),
    scale=np.array([0.3, 3.0, 1.5]),
    color=np.array([0.8, 0.2, 0.2]),
))

# 차량 대신 동적 박스 사용 (질량 1500kg)
from pxr import UsdPhysics as UP, Gf
box_prim = stage.DefinePrim("/World/vehicle_box", "Cube")
UP.CollisionAPI.Apply(box_prim)
rb = UP.RigidBodyAPI.Apply(box_prim)
mass_api = UP.MassAPI.Apply(box_prim)
mass_api.CreateMassAttr(1500.0)

from pxr import UsdGeom
UsdGeom.Xformable(box_prim).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.75))
UsdGeom.Xformable(box_prim).AddScaleOp().Set(Gf.Vec3d(1.0, 0.8, 0.75))

# vehicle_box에 compliant material 바인딩
binding_api = UsdShade.MaterialBindingAPI.Apply(box_prim)
binding_api.Bind(
    UsdShade.Material(mat_prim),
    UsdShade.Tokens.weakerThanDescendants,
    "physics",
)
print("material binding 완료")

# SingleRigidPrim으로 wrapping
vehicle = world.scene.add(
    SingleRigidPrim(prim_path="/World/vehicle_box", name="vehicle_box")
)

# ── 4. 충돌 시뮬레이션 ────────────────────────────────────────────────────
print("\n=== [4] 60km/h 충돌 시뮬레이션 ===")
world.reset()

v0 = 60.0 / 3.6  # 16.67 m/s
vehicle.set_linear_velocity(np.array([v0, 0.0, 0.0], dtype=np.float32))

max_force = 0.0
nan_count = 0
max_speed = v0

for step in range(300):  # 300ms
    world.step(render=False)
    pos, _ = vehicle.get_world_pose()
    vel = vehicle.get_linear_velocity()

    if not all(np.isfinite(v) for v in pos) or not all(np.isfinite(v) for v in vel):
        nan_count += 1
        if nan_count == 1:
            print(f"  step {step}: NaN 발생! pos={pos}, vel={vel}")
        continue

    speed = float(np.linalg.norm(vel))
    max_speed = max(max_speed, speed)
    px = float(pos[0])

    if step % 50 == 0:
        print(f"  step {step:3d} | pos_x={px:.3f}m | speed={speed:.2f}m/s")

    if speed < 0.1 and px > 1.0:
        print(f"  → 완전 정지 (step {step}, pos_x={px:.3f}m)")
        break

print(f"\n결과: NaN {nan_count}회 | 최대속도 {max_speed:.1f}m/s")
if nan_count == 0:
    print("✓ compliant contact 정상 작동 — NaN 없음")
else:
    print("✗ NaN 발생 — 추가 조정 필요")

sim_app.close()
