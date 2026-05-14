from omni.isaac.core.objects import DynamicCuboid
import numpy as np


class Vehicle:
    def __init__(self, world, position=(0.0, 0.0, 0.5)):
        self.body = world.scene.add(
            DynamicCuboid(
                prim_path="/World/vehicle",
                name="vehicle",
                position=np.array(position),
                size=np.array([2.0, 1.0, 0.8]),
                mass=1500.0,
            )
        )

    def apply_collision_force(self, angle_deg, speed_kmh, stiffness="concrete"):
        stiffness_map = {"concrete": 1.0, "vehicle": 0.7, "wood": 0.4}
        k = stiffness_map.get(stiffness, 1.0)
        speed_ms = speed_kmh / 3.6
        force_magnitude = 1500.0 * speed_ms * k * 10.0
        angle_rad = np.deg2rad(angle_deg)
        fx = -np.cos(angle_rad) * force_magnitude
        fy = -np.sin(angle_rad) * force_magnitude
        self.body.apply_force_torque(force=np.array([fx, fy, 0.0]))

    def reset(self, position=(0.0, 0.0, 0.5)):
        self.body.set_world_pose(position=np.array(position))
        self.body.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.body.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
