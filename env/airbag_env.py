import numpy as np
import gymnasium as gym
from gymnasium import spaces

from omni.isaac.core import World

from env.vehicle import Vehicle
from env.human import Human
from env.airbag import AirbagSystem
from env.scenario import ScenarioSampler
from rl.reward import compute_hic, compute_reward


SIM_DT = 1.0 / 60.0        # Isaac Sim 기본 스텝
COLLISION_STEPS = 10        # 충돌 힘 적용 스텝 수
EPISODE_STEPS = 60          # 에피소드 총 스텝 수
TIMING_MAX_MS = 30.0


class AirbagEnv(gym.Env):
    """
    State  : 7차원 (scenario 벡터)
    Action : 15차원 (에어백 5개 × [deploy, timing, pressure])
             deploy  → 이산 (0/1), multi-head PPO에서 Categorical 처리
             timing  → 연속 [0, 30] ms
             pressure→ 연속 [0, 600] kPa
    """

    def __init__(self, headless=True):
        super().__init__()
        self.world = World(physics_dt=SIM_DT, stage_units_in_meters=1.0)
        self.sampler = ScenarioSampler()
        self.scenario = None

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        # deploy(5) + timing(5) + pressure(5) 모두 [0,1]로 정규화해서 넘김
        # multi-head PPO가 내부에서 분리해서 처리
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )

        self._head_acc_history = []

    def reset(self, seed=None, options=None):
        self.world.reset()
        self.scenario = self.sampler.sample()

        self.vehicle = Vehicle(self.world)
        self.human = Human(
            self.world,
            height=self.scenario["height"],
            weight=self.scenario["weight"],
        )
        self.airbag_sys = AirbagSystem(self.human)
        self.world.reset()

        self._head_acc_history = []
        self._step = 0

        obs = self.sampler.to_state_vector(self.scenario)
        return obs, {}

    def step(self, action: np.ndarray):
        # action 파싱 및 역정규화
        raw_actions = self._parse_action(action)

        # 충돌 힘 적용 (초반 스텝만)
        if self._step < COLLISION_STEPS:
            self.vehicle.apply_collision_force(
                self.scenario["angle"],
                self.scenario["speed"],
                self.scenario["stiffness"],
            )

        # 안전벨트
        self.human.apply_seatbelt(self.scenario["seatbelt"])

        # 타이밍에 맞는 에어백 전개
        current_ms = self._step * SIM_DT * 1000.0
        timed_actions = self._apply_timing_mask(raw_actions, current_ms)
        self.airbag_sys.apply(timed_actions, self.scenario["angle"])

        self.world.step(render=False)
        self._step += 1

        head_vel = self.human.get_head_acceleration()
        self._head_acc_history.append(head_vel)

        done = self._step >= EPISODE_STEPS
        reward = 0.0

        if done:
            hic = compute_hic(self._head_acc_history, SIM_DT)
            deploy_flags = [raw_actions[i, 0] > 0.5 for i in range(5)]
            reward = compute_reward(
                hic=hic,
                chest_g=0.0,        # TODO: 흉부 가속도 측정 추가
                chest_compression_mm=0.0,  # TODO: 압축량 측정 추가
                femur_n=0.0,        # TODO: 대퇴부 힘 측정 추가
                neck_n=0.0,         # TODO: 목 전단력 측정 추가
                deploy_flags=deploy_flags,
            )

        obs = self.sampler.to_state_vector(self.scenario)
        return obs, reward, done, False, {}

    def _parse_action(self, action: np.ndarray) -> np.ndarray:
        """
        action[0:5]  → deploy (0/1 threshold 0.5)
        action[5:10] → timing [0, 30] ms
        action[10:15]→ pressure [0, 600] kPa
        반환: shape (5, 3)
        """
        result = np.zeros((5, 3), dtype=np.float32)
        for i in range(5):
            deploy = float(action[i] > 0.5)
            result[i, 0] = deploy
            if deploy:
                result[i, 1] = action[5 + i] * TIMING_MAX_MS
                result[i, 2] = action[10 + i] * 600.0
        # deploy=0이면 timing·pressure는 0으로 유지 (gradient 마스킹은 PPO단에서 처리)
        return result

    def _apply_timing_mask(self, actions: np.ndarray, current_ms: float) -> np.ndarray:
        masked = actions.copy()
        for i in range(5):
            if actions[i, 0] > 0.5 and current_ms < actions[i, 1]:
                masked[i, 0] = 0.0  # 아직 타이밍 안 됐으면 전개 억제
        return masked

    def close(self):
        self.world.stop()
