"""
에어백 RL 환경 — Isaac Sim 6.0 / Python 3.12.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat

from env.vehicle import Vehicle
from env.human import Human, SEAT_LOCAL
from env.airbag import AirbagSystem
from env.scenario import ScenarioSampler, STATE_DIM, SPINE_TILT_MIN_DEG, SPINE_TILT_MAX_DEG
from rl.reward import (
    InjuryDataCollector,
    compute_hic15, compute_chest_g, compute_chest_3ms_clip,
    compute_chest_compression_mm, compute_femur_force_n, compute_nij,
    compute_reward,
)

PHYSICS_DT      = 0.001
CONTROL_DT      = 1.0 / 60.0
COLLISION_STEPS = 60
TIMING_MAX_MS   = 30.0

WALL_DIST_M = 3.5
WALL_SIZE   = np.array([0.5, 5.0, 3.0])
WALL_POS_Z  = 1.5


class AirbagEnv(gym.Env):
    """
    State  : 12차원
    Action : 15차원 (에어백 5개 × [deploy, timing, pressure])
    """

    def __init__(self, headless: bool = True, debug: bool = False):
        super().__init__()
        self.world = World(
            physics_dt=PHYSICS_DT,
            rendering_dt=CONTROL_DT,
            stage_units_in_meters=1.0,
        )
        self.sampler   = ScenarioSampler()
        self._rng      = np.random.default_rng()
        self.scenario  = None
        self.collector = InjuryDataCollector(human=None, physics_dt=PHYSICS_DT)
        self._wall     = None

        self.debug = debug
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self.action_space      = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)

    # ── reset ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.world.reset()
        self.scenario = self.sampler.sample()

        angle_deg = self.scenario["angle"]
        speed_kmh = self.scenario["speed"]

        self.vehicle = Vehicle(self.world, position=(0.0, 0.0, 0.0))
        self._place_wall(angle_deg)

        human_world_pos = np.zeros(3) + SEAT_LOCAL
        self.human = Human(
            self.world,
            base_position=human_world_pos,
            height=self.scenario["height"],
            weight=self.scenario["weight"],
        )
        self.airbag_sys = AirbagSystem(self.world, self.human)

        self.world.reset()

        self.human.initialize()
        self.airbag_sys.reset()

        # 착좌 자세 인가
        spine_tilt_deg = float(self._rng.uniform(SPINE_TILT_MIN_DEG, SPINE_TILT_MAX_DEG))
        self.human.set_sitting_posture(spine_tilt_deg=spine_tilt_deg)

        # 관절 위치 PhysX 전파
        self.world.step(render=False)

        # Pre-crash snapshot
        snapshot = self.human.measure_snapshot(vehicle_body=self.vehicle.body)
        self.scenario.update({
            "sitting_height":    snapshot["sitting_height"],
            "head_pos":          snapshot["head_pos"],
            "spine_tilt_deg":    snapshot["spine_tilt_deg"],
            "head_to_steering":  snapshot["head_to_steering"],
            "knee_to_dashboard": snapshot["knee_to_dashboard"],
        })

        # 초기 속도 부여
        speed_ms  = speed_kmh / 3.6
        angle_rad = np.deg2rad(angle_deg)
        init_vel  = np.array([speed_ms * np.cos(angle_rad),
                               speed_ms * np.sin(angle_rad), 0.0])
        self.vehicle.body.set_linear_velocity(init_vel)
        self.human.set_initial_velocity(init_vel)

        # 센서 콜백 등록
        self.collector.human = self.human
        self.collector.reset()
        try:
            self.world.remove_physics_callback("collect_injury")
        except Exception:
            pass
        self.world.add_physics_callback("collect_injury", self.collector.physics_callback)

        self._step = 0
        obs = self.sampler.to_state_vector(self.scenario)
        return obs, {}

    # ── step ───────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        raw_actions = self._parse_action(action)

        self.human.apply_seatbelt(self.scenario["seatbelt"], self.vehicle.body)
        current_ms = self._step * CONTROL_DT * 1000.0
        self.airbag_sys.apply(raw_actions, self.scenario["angle"], current_ms)

        self.world.step(render=False)
        self._step += 1

        if self.debug:
            print(
                f"[step {self._step:02d}] "
                f"head={len(self.collector.head_acc_g):4d}샘플  "
                f"torso={len(self.collector.torso_acc_g):4d}샘플  "
                f"thigh={len(self.collector.thigh_acc_3d):4d}샘플",
                flush=True,
            )

        done   = self._step >= COLLISION_STEPS
        reward = 0.0

        if done:
            dt          = PHYSICS_DT
            hic15       = compute_hic15(self.collector.head_acc_g, dt)
            chest_g     = compute_chest_g(self.collector.torso_acc_g)
            chest_3ms   = compute_chest_3ms_clip(self.collector.torso_acc_g, dt)
            compression = compute_chest_compression_mm(self.collector.torso_pos_history)
            femur_n     = compute_femur_force_n(self.collector.thigh_acc_3d)
            nij         = compute_nij(self.collector.head_acc_3d)
            deploy_flags = [raw_actions[i, 0] > 0.5 for i in range(5)]

            reward = compute_reward(
                hic15=hic15, chest_g=chest_g, chest_3ms=chest_3ms,
                chest_compression_mm=compression, femur_n=femur_n,
                nij=nij, deploy_flags=deploy_flags,
            )

        obs = self.sampler.to_state_vector(self.scenario)
        return obs, reward, done, False, {}

    # ── 종료 ───────────────────────────────────────────────────────────

    def close(self):
        try:
            self.world.remove_physics_callback("collect_injury")
        except Exception:
            pass
        self.world.stop()

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
        result = np.zeros((5, 3), dtype=np.float32)
        for i in range(5):
            deploy = float(action[i] > 0.5)
            result[i, 0] = deploy
            if deploy:
                result[i, 1] = action[5  + i] * TIMING_MAX_MS
                result[i, 2] = action[10 + i] * 600.0
        return result
