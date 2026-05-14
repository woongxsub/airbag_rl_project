from omni.isaac.core.objects import DynamicCuboid
import numpy as np


class Human:
    """
    머리/몸통/하체 박스 3개를 관절 없이 독립 강체로 구성한 러프 프로토타입.
    관절 연결은 Isaac Sim 세팅 확인 후 추가 예정.
    """

    def __init__(self, world, base_position=(0.3, 0.0, 0.5), height=1.75, weight=70.0):
        self.base_pos = np.array(base_position)
        self.height = height
        self.weight = weight
        self._build(world)

    def _build(self, world):
        head_h = self.height * 0.15
        torso_h = self.height * 0.40
        lower_h = self.height * 0.45

        head_z = self.base_pos[2] + lower_h + torso_h + head_h / 2
        torso_z = self.base_pos[2] + lower_h + torso_h / 2
        lower_z = self.base_pos[2] + lower_h / 2

        xy = self.base_pos[:2]

        self.head = world.scene.add(
            DynamicCuboid(
                prim_path="/World/human/head",
                name="head",
                position=np.array([*xy, head_z]),
                size=np.array([0.2, 0.2, head_h]),
                mass=self.weight * 0.08,
            )
        )
        self.torso = world.scene.add(
            DynamicCuboid(
                prim_path="/World/human/torso",
                name="torso",
                position=np.array([*xy, torso_z]),
                size=np.array([0.35, 0.25, torso_h]),
                mass=self.weight * 0.55,
            )
        )
        self.lower = world.scene.add(
            DynamicCuboid(
                prim_path="/World/human/lower",
                name="lower",
                position=np.array([*xy, lower_z]),
                size=np.array([0.30, 0.25, lower_h]),
                mass=self.weight * 0.37,
            )
        )

    def apply_seatbelt(self, wearing: bool):
        # 안전벨트 → 몸통에 복원력 적용 (러프 구현, 수치는 추후 조정)
        if wearing:
            vel = self.torso.get_linear_velocity()
            restraint_force = -vel * 800.0
            self.torso.apply_force_torque(force=restraint_force)

    def get_head_acceleration(self):
        return self.head.get_linear_velocity()

    def reset(self):
        self._reset_box(self.head)
        self._reset_box(self.torso)
        self._reset_box(self.lower)

    def _reset_box(self, box):
        pos, _ = box.get_world_pose()
        box.set_world_pose(position=pos)
        box.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        box.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
