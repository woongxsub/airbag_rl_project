import ctypes
import sys

ctypes.PyDLL('/workspace/isaac_env/lib/python3.10/site-packages/omni/libcarb.so', mode=ctypes.RTLD_GLOBAL)
sys.path.insert(0, '/workspace/isaac_env/lib/python3.10/site-packages/omni/kernel/py')
sys.path.insert(0, '/workspace/isaac_env/lib/python3.10/site-packages')

import carb
print('carb ok:', carb.__file__)

from isaacsim import SimulationApp
print('SimulationApp import ok')
