import os
import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleRigidPrim
import omni.usd

VEHICLE_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/vehicle.usd")
VEHICLE_PRIM_PATH = "/World/vehicle"


class Vehicle:
    def __init__(self, world, position=(0.0, 0.0, 0.0)):
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(VEHICLE_PRIM_PATH).IsValid():
            add_reference_to_stage(
                usd_path=os.path.abspath(VEHICLE_USD),
                prim_path=VEHICLE_PRIM_PATH,
            )

        if world.scene.object_exists("vehicle"):
            self.body = world.scene.get_object("vehicle")
        else:
            self.body = world.scene.add(
                SingleRigidPrim(
                    prim_path=VEHICLE_PRIM_PATH,
                    name="vehicle",
                    position=np.array(position),
                )
            )

    def reset(self, position=(0.0, 0.0, 0.0)):
        self.body.set_world_pose(position=np.array(position))
        self.body.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.body.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
