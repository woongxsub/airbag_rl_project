import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
world = World(physics_dt=0.001, rendering_dt=1/60, stage_units_in_meters=1.0)

print("=== PhysxSchema compliant contact ===")
try:
    from pxr import PhysxSchema
    attrs = [a for a in dir(PhysxSchema.PhysxMaterialAPI) if 'ompliant' in a.lower()]
    print("compliant attrs:", attrs if attrs else "없음 — CPU PhysX에서 미지원")
    all_attrs = [a for a in dir(PhysxSchema.PhysxMaterialAPI) if not a.startswith('_')]
    print("전체 attrs:", all_attrs)
except Exception as e:
    print("오류:", e)

print("\n=== SingleRigidPrim 힘/속도 API ===")
from isaacsim.core.prims import SingleRigidPrim
methods = [m for m in dir(SingleRigidPrim) if any(k in m.lower() for k in ['force','vel','impulse','apply'])]
print(methods)

sim_app.close()
