import ctypes
import sys
import os
import importlib.util

OMNI = '/workspace/isaac_env/lib/python3.10/site-packages/omni'
KERNEL_PY = f'{OMNI}/kernel/py'
SITE = '/workspace/isaac_env/lib/python3.10/site-packages'
CARB_SO = f'{KERNEL_PY}/carb/_carb.cpython-310-x86_64-linux-gnu.so'

# 1. libcarb.so RTLD_GLOBAL 로딩
print("1. libcarb.so 로딩...")
ctypes.PyDLL(f'{OMNI}/libcarb.so', mode=ctypes.RTLD_GLOBAL)
print("   OK")

# 2. sys.path 설정
sys.path.insert(0, KERNEL_PY)
sys.path.insert(0, SITE)

# 3. carb 패키지 위치 확인
print("2. carb 패키지 위치 탐색...")
spec = importlib.util.find_spec('carb')
if spec:
    print(f"   carb found: {spec.origin}")
else:
    print("   carb NOT FOUND")

# 4. _carb.so 파일 존재 확인
print(f"3. _carb.so 파일 존재: {os.path.exists(CARB_SO)}")

# 5. _carb 직접 로딩 시도
print("4. carb._carb 직접 로딩 시도...")
spec2 = importlib.util.spec_from_file_location("carb._carb", CARB_SO)
try:
    mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod)
    print("   OK")
except Exception as e:
    print(f"   FAIL: {e}")
