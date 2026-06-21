"""
chest_compression 디버그 스크립트.
physics_callback에서 torso_pos와 _vehicle_int_pos의 실제 값을 추적한다.
"""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
from env.airbag_env import AirbagEnv, PHYSICS_DT, CONTROL_DT

# ── physics_callback 패치: 매 50 스텝마다 내부 값 출력 ─────────────────────
_debug_samples = []  # (step, torso_raw, veh_int_pos, torso_relative)
_cb_step = [0]

_orig_physics_callback = None

def _patched_callback(self_col, step_size):
    """collector.physics_callback을 래핑해 내부 값을 스냅샷한다."""
    # raw torso position (차량 적분 전)
    if self_col.human is not None and self_col.human.articulation is not None:
        try:
            raw_pos, _ = self_col.human.articulation.get_world_pose()
            torso_raw = np.asarray(raw_pos, dtype=np.float32).copy()
        except Exception:
            torso_raw = np.array([float('nan'), float('nan'), float('nan')])
    else:
        torso_raw = np.array([float('nan'), float('nan'), float('nan')])

    veh_int_before = self_col._vehicle_int_pos.copy()

    # 원래 콜백 실행
    _orig_physics_callback(self_col, step_size)

    veh_int_after = self_col._vehicle_int_pos.copy()
    torso_relative = torso_raw - veh_int_after

    _cb_step[0] += 1
    if _cb_step[0] <= 5 or _cb_step[0] % 100 == 0:
        _debug_samples.append({
            "step": _cb_step[0],
            "torso_raw": torso_raw.tolist(),
            "veh_int": veh_int_after.tolist(),
            "torso_rel": torso_relative.tolist(),
        })


# InjuryDataCollector의 physics_callback을 몽키패치
from rl.reward import InjuryDataCollector
_orig_physics_callback = InjuryDataCollector.physics_callback

def patched(self, step_size):
    _patched_callback(self, step_size)

InjuryDataCollector.physics_callback = patched

# ── 1 에피소드 실행 ────────────────────────────────────────────────────────
print("\n=== chest_compression 디버그 ===")
env = AirbagEnv(headless=True, debug=True)
obs, _ = env.reset()

# 고정 시나리오: 정면 충돌, 60 km/h, 벨트 ON
speed_ms = 60.0 / 3.6
init_vel = np.array([speed_ms, 0.0, 0.0])
env.vehicle.body.set_linear_velocity(init_vel)
env.human.set_initial_velocity(init_vel)
env.scenario["angle"] = 0.0
env.scenario["speed"] = 60.0
env.scenario["seatbelt"] = True

print(f"초기 속도: {init_vel}")
print(f"SEAT_LOCAL: {env.human._position}")

action = np.zeros(15, dtype=np.float32)
action[0] = 1.0; action[5] = 5.0/30.0; action[10] = 300.0/600.0

_cb_step[0] = 0
_debug_samples.clear()

done = False
while not done:
    obs, r, done, _, info = env.step(action)

# ── 결과 출력 ─────────────────────────────────────────────────────────────
col = env.collector
print(f"\n총 physics callback 스텝: {_cb_step[0]}")
print(f"torso_pos_history 샘플 수: {len(col.torso_pos_history)}")

print("\n[첫 5회 + 매 100 스텝 스냅샷]")
print(f"  {'step':>5}  {'torso_raw_x':>12}  {'veh_int_x':>12}  {'rel_x(m)':>10}  {'rel_x(mm)':>10}")
for d in _debug_samples:
    print(f"  {d['step']:>5}  {d['torso_raw'][0]:>12.4f}  {d['veh_int'][0]:>12.4f}"
          f"  {d['torso_rel'][0]:>10.4f}  {d['torso_rel'][0]*1000:>10.2f}")

if col.torso_pos_history:
    positions = np.stack(col.torso_pos_history)
    print(f"\n[torso_pos_history 통계]")
    print(f"  첫 번째 위치: {positions[0]}")
    print(f"  마지막 위치: {positions[-1]}")
    print(f"  x 범위: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}] m")
    disp = positions - positions[0]
    print(f"  x 최대 변위: {np.abs(disp[:, 0]).max():.4f} m = {np.abs(disp[:, 0]).max()*1000:.1f} mm")

from rl.reward import compute_chest_compression_mm
comp = compute_chest_compression_mm(col.torso_pos_history)
print(f"\n  compute_chest_compression_mm 결과: {comp:.2f} mm")

env.close()
sim_app.close()
print("\n=== 완료 ===")
