"""
차량 USD 생성 스크립트 — 한 번만 실행하면 vehicle.usd 생성됨.

실행 방법:
    python assets/create_car_usd.py

생성 파일: assets/vehicle.usd
차체 형태: 박스 바디 + 루프 + 바퀴 4개 (SUV 비율)
물리 설정: RigidBody(1500kg) + 각 부위 Box/Cylinder 콜라이더
"""

import os
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicle.usd")


def _add_box_collider(stage, path, half_extents, translate):
    """Box 형태 콜라이더 추가 (UsdGeom.Cube + scale + translate)."""
    prim = UsdGeom.Cube.Define(stage, path)
    prim.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(prim.GetPrim())
    xf.AddScaleOp().Set(Gf.Vec3f(*[e * 2 for e in half_extents]))  # size=1 → half_ext*2
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    return prim


def _add_cylinder_collider(stage, path, radius, height, translate):
    """원통 바퀴 콜라이더 추가."""
    prim = UsdGeom.Cylinder.Define(stage, path)
    prim.GetRadiusAttr().Set(radius)
    prim.GetHeightAttr().Set(height)
    prim.GetAxisAttr().Set("Y")
    xf = UsdGeom.Xformable(prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    return prim


def create_vehicle_usd(output_path: str):
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)

    # ── 루트 Xform (RigidBody 물리 설정) ──
    root = UsdGeom.Xform.Define(stage, "/vehicle")
    stage.SetDefaultPrim(root.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(1500.0)
    # 무게중심을 바닥 기준 약 0.6m 위에 설정 (SUV 기준)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.6))

    # ── 차체 (메인 바디) ──
    # half_extents: (2.1m, 0.95m, 0.6m) → 전체 4.2m × 1.9m × 1.2m
    _add_box_collider(stage, "/vehicle/body",
                      half_extents=(2.1, 0.95, 0.6),
                      translate=(0.0, 0.0, 0.7))   # z=0.7m: 타이어 위

    # ── 루프 ──
    # 전체 2.0m × 1.6m × 0.7m, 차체 위에 얹힘
    _add_box_collider(stage, "/vehicle/roof",
                      half_extents=(1.0, 0.8, 0.35),
                      translate=(0.1, 0.0, 1.65))  # z=1.65m: 바디 위

    # ── 바퀴 4개 (반경 0.35m, 두께 0.22m) ──
    wheel_cfg = [
        ("wheel_FL", ( 1.15,  1.0, 0.35)),
        ("wheel_FR", ( 1.15, -1.0, 0.35)),
        ("wheel_RL", (-1.15,  1.0, 0.35)),
        ("wheel_RR", (-1.15, -1.0, 0.35)),
    ]
    for name, pos in wheel_cfg:
        _add_cylinder_collider(stage, f"/vehicle/{name}",
                               radius=0.35, height=0.22, translate=pos)

    stage.Save()
    print(f"vehicle.usd 생성 완료 → {output_path}")


if __name__ == "__main__":
    create_vehicle_usd(OUTPUT_PATH)
