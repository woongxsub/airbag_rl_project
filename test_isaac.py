import ctypes
import sys

OMNI = '/workspace/isaac_env/lib/python3.10/site-packages/omni'
KERNEL_PY = f'{OMNI}/kernel/py'
SITE = '/workspace/isaac_env/lib/python3.10/site-packages'

ctypes.PyDLL(f'{OMNI}/libcarb.so', mode=ctypes.RTLD_GLOBAL)
sys.path.insert(0, KERNEL_PY)
sys.path.insert(0, SITE)

print("1. carb import...")
import carb
print("   OK")

print("2. SimulationApp import...")
from isaacsim import SimulationApp
print("   OK")

print("3. SimulationApp 초기화...")
app = SimulationApp({"headless": True})
print("   OK - Isaac Sim 실행됨")

app.close()
print("완료")
