"""
에어백 RL 환경 — Isaac Sim 6.0 / Python 3.12.

물리 충돌 모델: rigid wall + compliant contact material
  차량 전면에 crumple zone 등가 재질 적용
  stiffness=4.5e5 N/m, damping=1e5 N·s/m
  → 1ms 스텝 내 충격력 유한화, NaN 폭발 방지
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from pxr import UsdPhysics, PhysxSchema, UsdShade, Sdf
from isaacsim.core.utils.rotations import euler_angles_to_quat

from env.vehicle import Vehicle
from env.human import Human, SEAT_LOCAL
from env.airbag import AirbagSystem
from env.scenario import ScenarioSampler, STATE_DIM, SPINE_TILT_MIN_DEG, SPINE_TILT_MAX_DEG
from rl.reward import (
    InjuryDataCollector,
    compute_hic15, compute_chest_g, compute_chest_3ms_clip,
    compute_chest_compression_mm, compute_femur_force_n, compute_nij,
    compute_reward, compute_step_reward,
)

PHYSICS_DT       = 0.001
CONTROL_DT       = 1.0 / 60.0
COLLISION_STEPS  = 60
PHYSICS_SUBSTEPS = max(1, round(CONTROL_DT / PHYSICS_DT))  # 17
TIMING_MAX_MS    = 30.0

WALL_DIST_M = 3.5
WALL_SIZE   = np.array([0.5, 5.0, 3.0])
WALL_POS_Z  = 1.5

# compliant contact (crumple zone 등가 재질)
_CONTACT_STIFFNESS = 4.5e5  # N/m  — 차량 크럼플존 등가 스프링 강성 (2e5→4.5e5 조정)
_CONTACT_DAMPING   = 1e5    # N·s/m — 임계감쇠 근처 (바운싱 억제)


class AirbagEnv(gym.Env):
    """
    State  : 11차원 (실차 센서 측정 가능한 값만, scenario.STATE_DIM=11)
    Action : 15차원 (에어백 5개 × [deploy, timing, pressure])
    Reward : Dense (스텝마다) + Terminal (에피소드 종료 시 bonus/penalty)
             안전 지표 5개: HIC15, Nij, chest_g, chest_3ms, chest_compression_mm
    """

    def __init__(self, headless: bool = True, debug: bool = False,
                 violation_coeff: float = 5.0):
        super().__init__()
        self.world = World(
            physics_dt=PHYSICS_DT,
            rendering_dt=CONTROL_DT,
            stage_units_in_meters=1.0,
        )
        self.world.get_physics_context().enable_gpu_dynamics(True)
        self.sampler          = ScenarioSampler()
        self._rng             = np.random.default_rng()
        self.scenario         = None
        self.collector        = InjuryDataCollector(human=None, vehicle_body=None, physics_dt=PHYSICS_DT)
        self._wall            = None
        self.last_raw_actions = None

        self._cb_actions  = np.zeros((5, 3), dtype=np.float32)
        self._cb_seatbelt = False
        self._physics_ms  = 0.0

        self.debug           = debug
        self.violation_coeff = violation_coeff  # 커리큘럼 단계에서 외부 갱신 가능
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self.action_space      = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)

    # ── reset ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # 이전 에피소드 콜백을 world.reset() 이전에 먼저 제거
        # → world.reset() 내부 물리 스텝에서 구 콜백이 실행되는 것을 차단 (addTorque NaN 방지)
        for cb in ("collect_injury", "apply_forces"):
            try:
                self.world.remove_physics_callback(cb)
            except Exception:
                pass

        self.world.reset()
        self.scenario = self.sampler.sample()

        angle_deg = self.scenario["angle"]
        speed_kmh = self.scenario["speed"]

        self.vehicle = Vehicle(self.world, position=(0.0, 0.0, 0.0))
        self._place_wall(angle_deg)
        self._apply_vehicle_compliant_contact()

        human_world_pos = np.zeros(3) + SEAT_LOCAL
        self.human = Human(
            self.world,
            base_position=human_world_pos,
            height=self.scenario["height"],
            weight=self.scenario["weight"],
        )
        self.airbag_sys = AirbagSystem(self.world, self.human, self.vehicle)

        if not self.world.scene.object_exists("ground_plane"):
            self.world.scene.add(GroundPlane(prim_path="/World/GroundPlane",
                                             name="ground_plane",
                                             z_position=0.0))

        self.world.reset()

        self.human.initialize()
        self._filter_human_collisions()
        self.airbag_sys.reset()

        spine_tilt_deg = float(self._rng.uniform(SPINE_TILT_MIN_DEG, SPINE_TILT_MAX_DEG))
        self.human.set_sitting_posture(spine_tilt_deg=spine_tilt_deg)

        self.world.step(render=False)

        snapshot = self.human.measure_snapshot(vehicle_body=self.vehicle.body)
        self.scenario.update({
            "sitting_height":    snapshot["sitting_height"],
            "head_pos":          snapshot["head_pos"],
            "spine_tilt_deg":    snapshot["spine_tilt_deg"],
            "head_to_steering":  snapshot["head_to_steering"],
            "knee_to_dashboard": snapshot["knee_to_dashboard"],
        })

        speed_ms  = speed_kmh / 3.6
        angle_rad = np.deg2rad(angle_deg)
        init_vel  = np.array([speed_ms * np.cos(angle_rad),
                               speed_ms * np.sin(angle_rad), 0.0])
        self.vehicle.body.set_linear_velocity(init_vel)
        self.human.set_initial_velocity(init_vel)

        self.collector.human        = self.human
        self.collector.vehicle_body = self.vehicle.body  # chest_compression 상대 변위 계산용
        self.collector.reset()
        self.world.add_physics_callback("collect_injury", self.collector.physics_callback)

        self.human.reset_belt_reference()  # 안전벨트 스프링 기준위치 초기화

        self._cb_actions  = np.zeros((5, 3), dtype=np.float32)
        self._cb_seatbelt = bool(self.scenario["seatbelt"])
        self._physics_ms  = 0.0

        self.world.add_physics_callback("apply_forces", self._force_physics_callback)

        self._step              = 0
        self._prev_sample_count = 0

        obs = self.sampler.to_state_vector(self.scenario)
        return obs, {}

    # ── step ───────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        raw_actions = self._parse_action(action)
        self.last_raw_actions = raw_actions.copy()

        self._cb_actions  = raw_actions.copy()
        self._cb_seatbelt = bool(self.scenario["seatbelt"])

        current_ms = self._step * CONTROL_DT * 1000.0
        self.airbag_sys.apply(raw_actions, self.scenario["angle"], current_ms)

        prev_count = self._prev_sample_count

        for _ in range(PHYSICS_SUBSTEPS):
            self.world.step(render=False)
        self._step += 1

        curr_count = len(self.collector.head_acc_g)
        self._prev_sample_count = curr_count

        if self.debug:
            print(
                f"[step {self._step:02d}] "
                f"head={curr_count:4d}샘플  "
                f"torso={len(self.collector.torso_acc_g):4d}샘플  "
                f"thigh={len(self.collector.thigh_acc_3d):4d}샘플",
                flush=True,
            )

        deploy_flags = [raw_actions[i, 0] > 0.5 for i in range(5)]

        # Dense reward: 이번 스텝 윈도우 (~17ms)
        reward = compute_step_reward(
            head_acc_g      = self.collector.head_acc_g[prev_count:curr_count],
            torso_acc_g     = self.collector.torso_acc_g[prev_count:curr_count],
            dt              = PHYSICS_DT,
            deploy_flags    = deploy_flags,
            n_steps         = COLLISION_STEPS,
            violation_coeff = self.violation_coeff,
        )

        done = self._step >= COLLISION_STEPS
        info = {}

        if done:
            dt          = PHYSICS_DT
            hic15       = compute_hic15(self.collector.head_acc_g, dt)
            chest_g     = compute_chest_g(self.collector.torso_acc_g)
            chest_3ms   = compute_chest_3ms_clip(self.collector.torso_acc_g, dt)
            compression = compute_chest_compression_mm(
                self.collector.torso_pos_history,
            )
            femur_n     = compute_femur_force_n(self.collector.thigh_acc_3d)
            nij         = compute_nij(self.collector.head_acc_3d)

            terminal_reward = compute_reward(
                hic15=hic15, chest_g=chest_g,
                deploy_flags=deploy_flags,
                violation_coeff=self.violation_coeff,
            )
            reward += terminal_reward
            info = {
                "hic15":                hic15,
                "chest_g":              chest_g,
                "chest_3ms":            chest_3ms,
                "chest_compression_mm": compression,
                "femur_n":              femur_n,
                "nij":                  nij,
                "deploy_count":         int(sum(deploy_flags)),
            }

        obs = self.sampler.to_state_vector(self.scenario)
        return obs, reward, done, False, info

    # ── 종료 ───────────────────────────────────────────────────────────

    def close(self):
        for cb in ("collect_injury", "apply_forces"):
            try:
                self.world.remove_physics_callback(cb)
            except Exception:
                pass
        self.world.stop()

    # ── compliant contact 적용 ─────────────────────────────────────────

    def _apply_vehicle_compliant_contact(self):
        """
        차량 물리 재질에 compliant contact 속성 부여.
        stiffness: 차량 크럼플존 등가 스프링 강성 (N/m)
        damping  : 에너지 흡수 댐퍼 계수 (N·s/m)
        → rigid wall 충돌 시 힘이 시간에 분산되어 NaN 방지.
        """
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        mat_path = "/World/VehicleContactMat"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            mat_prim = stage.DefinePrim(mat_path, "Material")

            phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
            phys_mat.CreateRestitutionAttr(0.0)
            phys_mat.CreateStaticFrictionAttr(0.3)
            phys_mat.CreateDynamicFrictionAttr(0.3)

            physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
            physx_mat.CreateCompliantContactStiffnessAttr(_CONTACT_STIFFNESS)
            physx_mat.CreateCompliantContactDampingAttr(_CONTACT_DAMPING)
        else:
            mat_prim = stage.GetPrimAtPath(mat_path)

        # 차량 전체 collision prim에 바인딩
        from pxr import Usd
        vehicle_prim = stage.GetPrimAtPath("/World/vehicle")
        if not vehicle_prim.IsValid():
            return
        count = 0
        for prim in Usd.PrimRange(vehicle_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                binding.Bind(
                    UsdShade.Material(mat_prim),
                    UsdShade.Tokens.weakerThanDescendants,
                    "physics",
                )
                count += 1
        if count:
            print(f"[Env] compliant contact material applied ({count} collision prims)")

    def _filter_human_collisions(self):
        """
        인체와 충돌해서는 안 되는 모든 prim에 FilteredPairs 필터 적용.

        대상:
          1. 차량 (/World/vehicle) — 인체가 차량 내부에서 시작하므로
             겹침 해소 충격력 방지 (기존 동작 유지)
          2. 벽 (/World/collision_wall) — 안전벨트·에어백이 수식 기반
             힘으로만 구현되어 있어 rigid body 직접 충돌 시 비현실적
             충격력 발생 → 필터로 차단

        이 모델에서 모든 신체 운동은 안전벨트·에어백의 수식 힘으로만
        결정된다. 수치는 절댓값이 아닌 Rule-Based 대비 상대적 개선으로
        해석해야 한다.
        """
        import omni.usd
        from pxr import Usd
        stage = omni.usd.get_context().get_stage()
        human_path = Sdf.Path("/World/human")

        # ── 1. 차량 → human 필터 ──────────────────────────────────────
        vehicle_prim = stage.GetPrimAtPath("/World/vehicle")
        veh_count = 0
        if vehicle_prim.IsValid():
            for prim in Usd.PrimRange(vehicle_prim):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    api = UsdPhysics.FilteredPairsAPI.Apply(prim)
                    rel = api.GetFilteredPairsRel()
                    if human_path not in rel.GetTargets():
                        rel.AddTarget(human_path)
                        veh_count += 1
        if veh_count:
            print(f"[Env] vehicle-human collision filter applied ({veh_count} prims)")

        # ── 2. 벽 → human 필터 ────────────────────────────────────────
        wall_prim = stage.GetPrimAtPath("/World/collision_wall")
        if wall_prim.IsValid():
            api = UsdPhysics.FilteredPairsAPI.Apply(wall_prim)
            rel = api.GetFilteredPairsRel()
            if human_path not in rel.GetTargets():
                rel.AddTarget(human_path)
                print("[Env] wall-human collision filter applied")

    def _force_physics_callback(self, step_size: float):
        """1ms 물리 스텝마다 호출: 안전벨트 + 에어백 감쇠력 인가."""
        self._physics_ms += step_size * 1000.0
        if self.human is None:
            return
        self.human.apply_seatbelt(self._cb_seatbelt, self.vehicle.body,
                                   step_dt=step_size)
        if self.airbag_sys is not None and self.scenario is not None:
            self.airbag_sys.apply_forces(
                self._cb_actions,
                self.scenario["angle"],
                self._physics_ms,
                step_dt=step_size,
            )

    # ── 내부 ───────────────────────────────────────────────────────────

    def _place_wall(self, angle_deg: float):
        angle_rad = np.deg2rad(angle_deg)
        wall_pos  = np.array([WALL_DIST_M * np.cos(angle_rad),
                               WALL_DIST_M * np.sin(angle_rad), WALL_POS_Z])
        wall_quat = euler_angles_to_quat(np.array([0.0, 0.0, np.deg2rad(angle_deg)]))

        if self._wall is None and not self.world.scene.object_exists("collision_wall"):
            self._wall = self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/collision_wall",
                    name="collision_wall",
                    position=wall_pos,
                    orientation=wall_quat,
                    scale=WALL_SIZE,
                    color=np.array([0.55, 0.55, 0.55]),
                )
            )
        elif self._wall is None:
            self._wall = self.world.scene.get_object("collision_wall")

        if self._wall is not None:
            self._wall.set_world_pose(position=wall_pos, orientation=wall_quat)

    def _parse_action(self, action: np.ndarray) -> np.ndarray:
        # 압력 Action 범위: 0~1 → 0~250 kPa (실측 기반 현실화)
        # 출처: US Patent 9623831 (80~250 kPa), ResearchGate 충돌 실험 (47~53 kPa)
        result = np.zeros((5, 3), dtype=np.float32)
        for i in range(5):
            deploy = float(action[i] > 0.5)
            result[i, 0] = deploy
            if deploy:
                result[i, 1] = action[5  + i] * TIMING_MAX_MS
                result[i, 2] = action[10 + i] * 250.0
        return result
