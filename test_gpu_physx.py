"""
GPU PhysX 활성화 + PhysxMaterialAPI compliant contact 검증.

실행 전 DISPLAY 환경변수가 설정되어 있어야 함:
  Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX &
  DISPLAY=:1 python test_gpu_physx.py
"""
import os, sys

# Xvfb가 없으면 자동 시작
if not os.environ.get("DISPLAY"):
    import subprocess, time
    subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24",
                      "-ac", "+extension", "GLX"])
    time.sleep(2.0)
    os.environ["DISPLAY"] = ":1"
    print(f"[test] Xvfb started, DISPLAY=:1")

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import carb
carb.settings.get_settings().set("/physics/cudaDevice", 0)

# ── 1. GPU PhysX 확인 ─────────────────────────────────────────────────────
print("\n=== 1. GPU PhysX ===")
from isaacsim.core.api import World
world = World(physics_dt=0.001, rendering_dt=1/60, stage_units_in_meters=1.0)
try:
    ctx = world.get_physics_context()
    ctx.enable_gpu_dynamics(True)
    print("enable_gpu_dynamics(True) 호출 성공")
except Exception as e:
    print(f"enable_gpu_dynamics 실패: {e}")

# ── 2. PhysxMaterialAPI 속성 확인 ────────────────────────────────────────
print("\n=== 2. PhysxMaterialAPI 속성 ===")
try:
    from pxr import PhysxSchema, UsdPhysics, Gf
    attrs = [a for a in dir(PhysxSchema.PhysxMaterialAPI) if not a.startswith('_')]
    compliant = [a for a in attrs if 'ompliant' in a.lower() or 'tiffness' in a.lower() or 'amping' in a.lower()]
    print("compliant/stiffness/damping 관련:", compliant)
    print("전체 속성 수:", len(attrs))
except Exception as e:
    print(f"PhysxMaterialAPI 확인 실패: {e}")

# ── 3. 실제 vehicle prim에 compliant contact 적용 테스트 ──────────────────
print("\n=== 3. compliant contact 적용 테스트 ===")
try:
    from pxr import PhysxSchema, UsdPhysics, Gf, UsdShade, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # 테스트용 prim 생성
    from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
    import numpy as np

    wall = world.scene.add(FixedCuboid(
        prim_path="/World/test_wall",
        name="test_wall",
        position=np.array([3.0, 0.0, 0.5]),
        scale=np.array([0.5, 2.0, 1.0]),
    ))

    # physics material 생성
    mat_path = "/World/test_mat"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat.CreateRestitutionAttr(0.0)
    phys_mat.CreateStaticFrictionAttr(0.5)

    # PhysxMaterialAPI 적용
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)

    # compliant contact 속성 시도
    applied = []
    for attr_name, val in [
        ("compliantContactStiffness", 1e6),   # N/m  (차량 크럼플존 등가 강성)
        ("compliantContactDamping",   5e3),    # N·s/m
    ]:
        try:
            attr = getattr(physx_mat, f"Create{attr_name}Attr", None)
            if attr:
                attr(val)
                applied.append(attr_name)
            else:
                # 직접 속성 이름으로 시도
                a = physx_mat.GetPrim().CreateAttribute(f"physxMaterial:{attr_name}", Sdf.ValueTypeNames.Float)
                a.Set(val)
                applied.append(f"{attr_name}(direct)")
        except Exception as ex:
            print(f"  {attr_name}: 실패 - {ex}")

    print(f"적용된 속성: {applied}")

    # material binding
    from pxr import UsdShade
    wall_prim = stage.GetPrimAtPath("/World/test_wall")
    if wall_prim.IsValid():
        binding = UsdShade.MaterialBindingAPI.Apply(wall_prim)
        binding.Bind(UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
        print("Material binding 성공")

except Exception as e:
    import traceback
    print(f"compliant contact 실패: {e}")
    traceback.print_exc()

# ── 4. 간단한 물리 스텝 실행 (NaN 여부) ──────────────────────────────────
print("\n=== 4. 물리 스텝 NaN 체크 ===")
try:
    from isaacsim.core.api.objects import DynamicSphere
    import numpy as np
    ball = world.scene.add(DynamicSphere(
        prim_path="/World/test_ball",
        name="test_ball",
        position=np.array([0.0, 0.0, 2.0]),
        radius=0.1,
        mass=1.0,
    ))
    world.reset()
    nan_count = 0
    for i in range(50):
        world.step(render=False)
        pos, _ = ball.get_world_pose()
        if not all(v == v for v in pos):  # NaN check
            nan_count += 1
    print(f"50 스텝 완료 (NaN 발생: {nan_count}회)")
    pos, _ = ball.get_world_pose()
    print(f"최종 위치: {pos}")
except Exception as e:
    import traceback
    print(f"물리 스텝 실패: {e}")
    traceback.print_exc()

sim_app.close()
print("\n=== 완료 ===")
