"""
Newton humanoid.usda 기반 인체 모델.

착좌 자세:
  - 에피소드 리셋 시 hip_y / knee 관절 90° 적용 → 운전석 착좌
  - spine(abdomen_y) 에 랜덤 미세 각도(-10°~+20°) 인가 → 자세 다양성

ToF 사전측정 (pre-crash snapshot):
  - head_pos    : ArticulationView link transform으로 머리 월드 좌표
  - sitting_height : 착좌 후 실측 앉은키 (head_z - seat_z)
  - spine_tilt_deg : 관절 위치 기반 척추 기울기
  - head_to_steering  : 머리 중심 → 스티어링휠 직선거리
  - knee_to_dashboard : 무릎(right_thigh) → 대시보드 잔여 거리
"""

import importlib.util
import os
import numpy as np
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.objects import VisualCapsule

# ── prim 경로 ────────────────────────────────────────────────────────────
HUMAN_PRIM_PATH = "/World/human"
_ARTIC_PATH     = f"{HUMAN_PRIM_PATH}/torso"   # Newton humanoid articulation root

SEAT_LOCAL           = np.array([0.3,  0.5,  0.9])
STEERING_LOCAL       = np.array([0.65, 0.50, 1.05])
DASHBOARD_KNEE_LOCAL = np.array([0.65, 0.50, 0.72])

_BELT_SHOULDER = SEAT_LOCAL + np.array([0.0,  0.20,  0.42])
_BELT_HIP      = SEAT_LOCAL + np.array([0.0, -0.18, -0.22])
_BELT_HIP_L    = SEAT_LOCAL + np.array([0.0,  0.22, -0.22])
_BELT_HIP_R    = SEAT_LOCAL + np.array([0.0, -0.22, -0.22])

_HIP_PITCH_HINTS = ["hip_y", "hip_pitch", "hip_flex"]
_KNEE_HINTS      = ["knee"]
_SPINE_HINTS     = ["abdomen_y", "lower_waist", "spine", "lumbar", "torso_pitch"]

SITTING_HIP_PITCH_RAD = -1.5708
SITTING_KNEE_FLEX_RAD =  1.5708


def _find_humanoid_usd() -> str:
    spec = importlib.util.find_spec("newton")
    if spec is None:
        raise RuntimeError("Newton 패키지 없음. Isaac Sim 설치 확인 요망.")
    path = os.path.join(os.path.dirname(spec.origin), "examples", "assets", "humanoid.usda")
    if not os.path.exists(path):
        raise FileNotFoundError(f"humanoid.usda 없음: {path}")
    return path


class Human:
    def __init__(self, world, base_position: np.ndarray = None, height: float = 1.75, weight: float = 70.0):
        self.height   = height
        self.weight   = weight
        self._position = np.array(base_position) if base_position is not None else SEAT_LOCAL.copy()
        self._world   = world

        self.articulation  = None
        self._physics_view = None   # omni.physics.tensors ArticulationView
        self._link_names:  list = []
        self._torso_idx:   int  = 0
        self._head_idx:    int  = 0
        self._thigh_idx:   int  = 0
        self._dof_name_to_idx: dict = {}

        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(HUMAN_PRIM_PATH).IsValid():
            add_reference_to_stage(usd_path=_find_humanoid_usd(), prim_path=HUMAN_PRIM_PATH)

        artic_path = self._resolve_artic_path()
        if world.scene.object_exists("human"):
            self.articulation = world.scene.get_object("human")
        else:
            try:
                self.articulation = world.scene.add(
                    SingleArticulation(
                        prim_path=artic_path,
                        name="human",
                        position=self._position,
                    )
                )
            except Exception as e:
                print(f"[Human] articulation scene.add failed ({artic_path}): {e}")

        self._add_seatbelt_visual(world, stage)

    # ── 초기화 ──────────────────────────────────────────────────────────

    def initialize(self):
        """world.reset() 이후 physics tensors view + DOF 이름 매핑 획득."""
        if self.articulation is None:
            return

        # DOF 이름 매핑
        try:
            names = self.articulation.dof_names
            self._dof_name_to_idx = {n: i for i, n in enumerate(names)}
        except Exception:
            self._dof_name_to_idx = {}

        # physics tensors ArticulationView 획득
        try:
            self._physics_view = self.articulation._articulation_view._physics_view
        except AttributeError:
            self._physics_view = None

        # 링크 이름 → 인덱스 매핑
        if self._physics_view is not None:
            try:
                meta = self._physics_view.get_metatype(0)
                self._link_names = list(meta.link_names)
                torso_found = head_found = thigh_found = False
                for i, name in enumerate(self._link_names):
                    nl = name.lower()
                    if not torso_found and any(k in nl for k in ["torso", "chest", "pelvis"]):
                        self._torso_idx = i
                        torso_found = "torso" in nl or "chest" in nl  # pelvis는 fallback만
                    if not head_found and "head" in nl:
                        self._head_idx  = i
                        head_found = True
                    if not thigh_found and ("right_thigh" in nl or ("thigh" in nl and "right" in nl)):
                        self._thigh_idx = i
                        thigh_found = True
                print(f"[Human] links={self._link_names}")
                print(f"[Human] torso={self._torso_idx} head={self._head_idx} thigh={self._thigh_idx}")
            except Exception as e:
                print(f"[Human] link index mapping failed: {e}")

    # ── 착좌 자세 ────────────────────────────────────────────────────────

    def set_sitting_posture(self, spine_tilt_deg: float = 0.0):
        if self.articulation is None or not self._dof_name_to_idx:
            return
        n = self.articulation.num_dof
        positions = np.zeros(n, dtype=np.float32)

        for hint in _HIP_PITCH_HINTS:
            for name, idx in self._dof_name_to_idx.items():
                if hint in name.lower() and idx < n:
                    positions[idx] = SITTING_HIP_PITCH_RAD

        for name, idx in self._dof_name_to_idx.items():
            if any(h in name.lower() for h in _KNEE_HINTS) and idx < n:
                positions[idx] = SITTING_KNEE_FLEX_RAD

        spine_rad = np.deg2rad(spine_tilt_deg)
        for hint in _SPINE_HINTS:
            applied = False
            for name, idx in self._dof_name_to_idx.items():
                if hint in name.lower() and idx < n:
                    positions[idx] = spine_rad
                    applied = True
                    break
            if applied:
                break

        try:
            self.articulation.set_joint_positions(positions)
        except Exception:
            pass

    def get_spine_tilt_deg(self) -> float:
        if self.articulation is None or not self._dof_name_to_idx:
            return 0.0
        try:
            jpos = self.articulation.get_joint_positions()
            for hint in _SPINE_HINTS:
                for name, idx in self._dof_name_to_idx.items():
                    if hint in name.lower() and idx < len(jpos):
                        return float(np.rad2deg(jpos[idx]))
        except Exception:
            pass
        return 0.0

    # ── Pre-crash snapshot ────────────────────────────────────────────────

    def measure_snapshot(self, vehicle_body=None) -> dict:
        head_world  = self.get_head_position()
        thigh_world = self.get_thigh_position()

        if vehicle_body is not None:
            veh_pos, veh_quat = vehicle_body.get_world_pose()
            from isaacsim.core.utils.rotations import quat_to_rot_matrix
            rot = quat_to_rot_matrix(veh_quat)
            veh_pos_arr = np.asarray(veh_pos)
            head_local  = rot.T @ (head_world - veh_pos_arr)
            steer_world = veh_pos_arr + rot @ STEERING_LOCAL
            dash_world  = veh_pos_arr + rot @ DASHBOARD_KNEE_LOCAL
        else:
            head_local  = head_world.copy()
            steer_world = STEERING_LOCAL.copy()
            dash_world  = DASHBOARD_KNEE_LOCAL.copy()

        sitting_height    = float(max(head_local[2] - SEAT_LOCAL[2], 0.3))
        head_to_steering  = float(np.linalg.norm(head_world - steer_world))
        knee_to_dashboard = float(max(np.linalg.norm(thigh_world - dash_world), 0.0))

        return {
            "head_pos":          head_local,
            "sitting_height":    sitting_height,
            "spine_tilt_deg":    self.get_spine_tilt_deg(),
            "head_to_steering":  head_to_steering,
            "knee_to_dashboard": knee_to_dashboard,
        }

    # ── 초기 속도 부여 ────────────────────────────────────────────────────

    def set_initial_velocity(self, vel: np.ndarray):
        if self.articulation is None:
            return
        try:
            self.articulation.set_linear_velocity(np.asarray(vel, dtype=np.float32))
        except Exception as e:
            print(f"[Human] set_initial_velocity failed: {e}")

    # ── 안전벨트 ─────────────────────────────────────────────────────────
    # FMVSS 209 / ECE R16 현업 스펙:
    #   k_belt  = 8,000 N/(m/s)  — 벨트 강성·감쇠 계수
    #   F_cap   = 15,000 N       — 로드 리미터 (프리텐셔너 포함 최대 하중)
    _K_BELT = 8_000.0
    _F_BELT_CAP = 15_000.0

    def apply_seatbelt(self, wearing: bool, vehicle_body=None):
        if not wearing or self.articulation is None:
            return
        torso_vel = self.get_torso_velocity()
        if vehicle_body is not None:
            try:
                veh_vel = np.asarray(vehicle_body.get_linear_velocity())
                rel_vel = torso_vel - veh_vel
            except Exception:
                rel_vel = torso_vel
        else:
            rel_vel = torso_vel

        raw_force = -rel_vel * self._K_BELT
        # 로드 리미터: 절댓값 캡
        norm = float(np.linalg.norm(raw_force))
        if norm > self._F_BELT_CAP:
            raw_force = raw_force * (self._F_BELT_CAP / norm)
        self._apply_link_force(self._torso_idx, raw_force.astype(np.float32))

    # ── 센서 getter ───────────────────────────────────────────────────────

    def get_head_velocity(self) -> np.ndarray:
        vels = self._get_link_velocities()
        if vels is not None:
            return np.asarray(vels[0, self._head_idx, :3])
        return np.zeros(3)

    def get_torso_velocity(self) -> np.ndarray:
        if self.articulation is not None:
            try:
                v = self.articulation.get_linear_velocity()
                if v is not None:
                    return np.asarray(v)
            except Exception:
                pass
        return np.zeros(3)

    def get_torso_position(self) -> np.ndarray:
        if self.articulation is not None:
            try:
                pos, _ = self.articulation.get_world_pose()
                return np.asarray(pos)
            except Exception:
                pass
        return np.zeros(3)

    def get_head_position(self) -> np.ndarray:
        transforms = self._get_link_transforms()
        if transforms is not None:
            return np.asarray(transforms[0, self._head_idx, :3])
        return self._fallback_prim_pos(_ARTIC_PATH)

    def get_thigh_position(self) -> np.ndarray:
        transforms = self._get_link_transforms()
        if transforms is not None:
            return np.asarray(transforms[0, self._thigh_idx, :3])
        return self._position.copy()

    def get_thigh_velocity_3d(self) -> np.ndarray:
        vels = self._get_link_velocities()
        if vels is not None:
            return np.asarray(vels[0, self._thigh_idx, :3])
        return np.zeros(3)

    # ── 리셋 ─────────────────────────────────────────────────────────────

    def reset(self):
        if self.articulation is not None:
            try:
                self.articulation.set_world_pose(position=self._position)
            except Exception:
                pass

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _get_link_velocities(self):
        """(1, max_links, 6) numpy array or None."""
        if self._physics_view is None:
            return None
        try:
            vels = np.asarray(self._physics_view.get_link_velocities())
            return vels.reshape(1, -1, 6)
        except Exception:
            return None

    def _get_link_transforms(self):
        """(1, max_links, 7) numpy array [x,y,z, qx,qy,qz,qw] or None."""
        if self._physics_view is None:
            return None
        try:
            t = np.asarray(self._physics_view.get_link_transforms())
            return t.reshape(1, -1, 7)
        except Exception:
            return None

    def _apply_link_force(self, link_idx: int, force: np.ndarray):
        """physics_view 로 특정 링크에 힘을 인가."""
        if self._physics_view is None:
            return
        try:
            n_links = self._physics_view.max_links
            forces  = np.zeros((1, n_links, 3), dtype=np.float32)
            torques = np.zeros((1, n_links, 3), dtype=np.float32)
            pos     = np.zeros((1, n_links, 3), dtype=np.float32)
            forces[0, link_idx] = force
            indices = np.array([0], dtype=np.int32)
            self._physics_view.apply_forces_and_torques_at_position(
                forces, torques, pos, indices, True
            )
        except Exception:
            pass

    def _fallback_prim_pos(self, prim_path: str) -> np.ndarray:
        try:
            from pxr import UsdGeom, Usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return self._position.copy()
            mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = mat.ExtractTranslation()
            return np.array([t[0], t[1], t[2]])
        except Exception:
            return self._position.copy()

    def _resolve_artic_path(self) -> str:
        """USD 로드 후 실제 articulation root prim 경로를 탐지."""
        stage = omni.usd.get_context().get_stage()

        if stage.GetPrimAtPath(_ARTIC_PATH).IsValid():
            return _ARTIC_PATH

        from pxr import UsdPhysics
        human_prim = stage.GetPrimAtPath(HUMAN_PRIM_PATH)
        if human_prim.IsValid():
            for prim in human_prim.GetChildren():
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    print(f"[Human] articulation root found: {prim.GetPath()}")
                    return str(prim.GetPath())
            for prim in human_prim.GetChildren():
                for child in prim.GetChildren():
                    if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                        print(f"[Human] articulation root found: {child.GetPath()}")
                        return str(child.GetPath())

        print(f"[Human] articulation root not found, using default: {_ARTIC_PATH}")
        return _ARTIC_PATH

    def _add_seatbelt_visual(self, world, stage):
        belt_color   = np.array([0.15, 0.15, 0.15])
        shoulder_mid = (_BELT_SHOULDER + _BELT_HIP) / 2.0
        shoulder_len = float(np.linalg.norm(_BELT_HIP - _BELT_SHOULDER))
        lap_mid      = (_BELT_HIP_L + _BELT_HIP_R) / 2.0
        lap_len      = float(np.linalg.norm(_BELT_HIP_R - _BELT_HIP_L))

        for path, name, mid, length in [
            ("/World/seatbelt_shoulder", "seatbelt_shoulder", shoulder_mid, shoulder_len),
            ("/World/seatbelt_lap",      "seatbelt_lap",      lap_mid,      lap_len),
        ]:
            if stage.GetPrimAtPath(path).IsValid():
                continue
            if world.scene.object_exists(name):
                continue
            world.scene.add(
                VisualCapsule(
                    prim_path=path,
                    name=name,
                    position=mid,
                    radius=0.015,
                    height=length,
                    color=belt_color,
                )
            )
