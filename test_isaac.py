import ctypes
import sys

OMNI = '/workspace/isaac_env/lib/python3.10/site-packages/omni'
KERNEL_PY = f'{OMNI}/kernel/py'
SITE = '/workspace/isaac_env/lib/python3.10/site-packages'
CARB_SO = f'{KERNEL_PY}/carb/_carb.cpython-310-x86_64-linux-gnu.so'

# libcarb.so 직접 로딩 시도
print("1. libcarb.so 로딩 시도...")
try:
    ctypes.PyDLL(f'{OMNI}/libcarb.so', mode=ctypes.RTLD_GLOBAL)
    print("   OK")
except Exception as e:
    print(f"   FAIL: {e}")

# _carb.so 직접 로딩 시도 (실제 에러 확인)
print("2. _carb.so 직접 로딩 시도...")
try:
    ctypes.CDLL(CARB_SO)
    print("   OK")
except Exception as e:
    print(f"   FAIL: {e}")

# sys.path 추가
sys.path.insert(0, KERNEL_PY)
sys.path.insert(0, SITE)

# carb import 시도
print("3. carb import 시도...")
try:
    import carb
    print(f"   OK: {carb.__file__}")
except Exception as e:
    print(f"   FAIL: {e}")
