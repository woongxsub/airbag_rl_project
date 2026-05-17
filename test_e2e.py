"""
에어백 RL End-to-End 충돌 시뮬레이션 테스트.

실행 방법:
  # A. 헤드리스 수치 출력만
  python test_e2e.py --headless

  # B. 헤드리스 + 충돌 물리 그래프 PNG 저장
  python test_e2e.py --headless --plot

  # C. WebRTC 3D 스트리밍 (RunPod 포트 49100 TCP / 47998 UDP 필요)
  python test_e2e.py --stream

충돌 검증 항목:
  1. 차량 속도 감소 확인 (충돌 발생 여부)
  2. 인체 DC 센서 데이터 수집 여부 (비어있으면 DC 핸들 실패)
  3. HIC15 / chest_g / Nij / 대퇴부 전체 보상 파이프라인
"""

# ── Isaac Sim 부트스트랩 (SimulationApp 가장 먼저) ───────────────────────
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

import argparse

parser = argparse.ArgumentParser(description="에어백 RL E2E 충돌 테스트")
parser.add_argument("--headless", action="store_true", default=True,
                    help="헤드리스 모드 (default: True)")
parser.add_argument("--stream",   action="store_true", default=False,
                    help="WebRTC 3D 스트리밍 활성화 (headless 무시)")
parser.add_argument("--plot",     action="store_true", default=False,
                    help="충돌 물리 곡선 PNG 저장 (results/plots/)")
parser.add_argument("--angle",    type=float, default=0.0,
                    help="충돌 방향 (도, default: 0=정면)")
parser.add_argument("--speed",    type=float, default=60.0,
                    help="충돌 속도 (km/h, default: 60)")
parser.add_argument("--no-belt",  action="store_true", default=False,
                    help="안전벨트 미착용")
args = parser.parse_args()

# ── SimulationApp 초기화 ─────────────────────────────────────────────────
from isaacsim import SimulationApp

if args.stream:
    STREAMING_KIT = (
        "/workspace/isaacsim_env/lib/python3.12/site-packages/"
        "isaacsim/apps/isaacsim.exp.full.streaming.kit"
    )
    sim_app = SimulationApp({"headless": False}, experience=STREAMING_KIT)
    print("\n" + "="*60)
    print("WebRTC 스트리밍 시작됨")
    print("  신호 포트: TCP 49100  (RunPod Custom Port 로 노출)")
    print("  스트림 포트: UDP 47998")
    print("  Omniverse Streaming Client 또는 브라우저 WebRTC 클라이언트로 접속")
    print("="*60 + "\n")
else:
    sim_app = SimulationApp({"headless": args.headless})

# ── 이후 import (SimulationApp 반드시 먼저) ───────────────────────────────
import numpy as np

from env.airbag_env import AirbagEnv, PHYSICS_DT, CONTROL_DT, COLLISION_STEPS
from rl.reward import (
    compute_hic15, compute_chest_g, compute_chest_3ms_clip,
    compute_chest_compression_mm, compute_femur_force_n, compute_nij,
    HIC_SAFE, CHEST_G_SAFE, CHEST_3MS_SAFE,
    CHEST_COMPRESSION_SAFE, FEMUR_SAFE, NIJ_SAFE,
)

# ── 결과 디렉토리 ────────────────────────────────────────────────────────
if args.plot:
    os.makedirs("results/plots", exist_ok=True)

# ────────────────────────────────────────────────────────────────────────


def run_test_episode():
    """고정 시나리오 1 에피소드 실행, 상세 수치 출력."""
    env = AirbagEnv(headless=(not args.stream))

    # 고정 시나리오 주입 (재현성)
    obs, _ = env.reset()
    env.scenario["angle"]    = args.angle
    env.scenario["speed"]    = args.speed
    env.scenario["stiffness"] = "concrete"
    env.scenario["seatbelt"] = not args.no_belt

    speed_ms  = args.speed / 3.6
    angle_rad = np.deg2rad(args.angle)
    init_vel  = np.array([speed_ms * np.cos(angle_rad),
                           speed_ms * np.sin(angle_rad), 0.0])
    env.vehicle.body.set_linear_velocity(init_vel)
    env.human.set_initial_velocity(init_vel)

    print(f"\n{'='*62}")
    print(f"  에어백 RL 충돌 시뮬레이션 E2E 테스트")
    print(f"{'='*62}")
    print(f"  충돌 방향: {args.angle:.0f}°   속도: {args.speed:.0f} km/h   "
          f"강성: concrete   벨트: {'ON' if not args.no_belt else 'OFF'}")
    print(f"  신장: {env.scenario['height']:.2f}m   체중: {env.scenario['weight']:.1f}kg")
    print(f"  앉은키(실측): {env.scenario.get('sitting_height', 0):.3f}m   "
          f"척추 기울기: {env.scenario.get('spine_tilt_deg', 0):.1f}°")
    print(f"  머리→스티어링: {env.scenario.get('head_to_steering', 0):.3f}m   "
          f"무릎→대시보드: {env.scenario.get('knee_to_dashboard', 0):.3f}m")
    print(f"{'='*62}")
    print(f"  {'스텝':>4} | {'시간(ms)':>8} | {'차량속도(m/s)':>12} | "
          f"{'머리속도(m/s)':>12} | {'수집샘플':>8}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    # 에어백 전체 전개 (front_driver 항상 전개, timing=5ms, pressure=300kPa)
    action = np.zeros(15, dtype=np.float32)
    action[0] = 1.0       # front_driver deploy
    action[5] = 5.0 / 30.0   # timing 5ms
    action[10] = 300.0 / 600.0  # pressure 300kPa

    # 시간 이력 기록 (그래프용)
    history = {
        "time_ms": [], "veh_speed": [], "head_speed": [],
        "head_acc_g": [], "torso_acc_g": [],
    }

    done = False
    step = 0
    while not done:
        obs, reward, done, _, _ = env.step(action)
        step += 1

        t_ms  = step * CONTROL_DT * 1000.0
        vv    = env.vehicle.body.get_linear_velocity()
        veh_speed = float(np.linalg.norm(np.asarray(vv)))
        head_speed = float(np.linalg.norm(env.human.get_head_velocity()))
        n_samples  = len(env.collector.head_acc_g)

        history["time_ms"].append(t_ms)
        history["veh_speed"].append(veh_speed)
        history["head_speed"].append(head_speed)
        if env.collector.head_acc_g:
            history["head_acc_g"].append(env.collector.head_acc_g[-1])
            history["torso_acc_g"].append(env.collector.torso_acc_g[-1] if env.collector.torso_acc_g else 0.0)
        else:
            history["head_acc_g"].append(0.0)
            history["torso_acc_g"].append(0.0)

        # 매 10 스텝 출력
        if step % 10 == 0 or step == 1 or done:
            print(f"  {step:>4} | {t_ms:>8.1f} | {veh_speed:>12.3f} | "
                  f"{head_speed:>12.3f} | {n_samples:>8}")

    # ── 충돌 지표 계산 ──────────────────────────────────────────────────
    col = env.collector
    dt  = PHYSICS_DT

    hic15       = compute_hic15(col.head_acc_g, dt)
    chest_g     = compute_chest_g(col.torso_acc_g)
    chest_3ms   = compute_chest_3ms_clip(col.torso_acc_g, dt)
    compression = compute_chest_compression_mm(col.torso_pos_history)
    femur_n     = compute_femur_force_n(col.thigh_acc_3d)
    nij         = compute_nij(col.head_acc_3d)

    def _fmt(val, safe, unit):
        flag = "✓" if val <= safe else "✗ EXCEEDED"
        return f"{val:8.1f} {unit:<4}  (기준 {safe} {unit})  {flag}"

    print(f"\n{'='*62}")
    print("  충돌 상해 지표 (에피소드 종료)")
    print(f"{'='*62}")
    print(f"  HIC15          : {_fmt(hic15,       HIC_SAFE,           '')}")
    print(f"  흉부 가속도    : {_fmt(chest_g,     CHEST_G_SAFE,       'g')}")
    print(f"  흉부 3ms 클립  : {_fmt(chest_3ms,   CHEST_3MS_SAFE,     'g')}")
    print(f"  흉부 압축량    : {_fmt(compression, CHEST_COMPRESSION_SAFE, 'mm')}")
    print(f"  대퇴부 압축력  : {_fmt(femur_n,     FEMUR_SAFE,         'N')}")
    print(f"  Nij (목 상해)  : {_fmt(nij,         NIJ_SAFE,           '')}")
    print(f"  Reward         : {reward:.4f}")
    print(f"{'='*62}")

    # ── 에어백 전개 상태 진단 ────────────────────────────────────────────────
    from env.airbag import AIRBAG_SPECS
    print(f"\n  [에어백 전개 상태]")
    bag_sys = env.airbag_sys
    for i, spec in AIRBAG_SPECS.items():
        deployed = i in bag_sys._inflate_start or i in bag_sys._fully_inflated
        inflated = i in bag_sys._fully_inflated
        status = "완전팽창" if inflated else ("팽창중" if deployed else "미전개")
        print(f"  [{i}] {spec['name']:<20} : {status}")

    # ── physics_view 상태 진단 ────────────────────────────────────────────────
    print("\n  [Physics View 진단]")
    h = env.human
    pv_ok = h._physics_view is not None
    print(f"  physics_view : {'OK' if pv_ok else 'NONE ← 센서 데이터 없음'}")
    if pv_ok:
        print(f"  link_names   : {h._link_names}")
        print(f"  body indices : torso={h._torso_idx}  head={h._head_idx}  thigh={h._thigh_idx}")
    print(f"  수집된 가속도 샘플 수 : head_acc_g={len(col.head_acc_g)}  "
          f"torso_acc_g={len(col.torso_acc_g)}  thigh_acc_3d={len(col.thigh_acc_3d)}")

    env.close()

    # ── 그래프 생성 ──────────────────────────────────────────────────────
    if args.plot:
        _plot_collision(history, hic15, chest_g, nij)

    return {
        "hic15": hic15, "chest_g": chest_g, "chest_3ms": chest_3ms,
        "compression": compression, "femur_n": femur_n, "nij": nij,
        "reward": reward, "history": history,
    }


def _plot_collision(history: dict, hic15: float, chest_g: float, nij: float):
    """충돌 물리 곡선 4개를 PNG로 저장."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = history["time_ms"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"에어백 RL 충돌 시뮬레이션  |  각도={args.angle:.0f}°  "
            f"속도={args.speed:.0f}km/h",
            fontsize=13,
        )

        # 1. 속도 곡선
        ax = axes[0, 0]
        ax.plot(t, history["veh_speed"],  label="차량 속도 (m/s)", color="steelblue")
        ax.plot(t, history["head_speed"], label="머리 속도 (m/s)", color="tomato")
        ax.set_title("속도 vs 시간")
        ax.set_xlabel("시간 (ms)")
        ax.set_ylabel("속도 (m/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 머리 가속도 (HIC)
        ax = axes[0, 1]
        ax.plot(t, history["head_acc_g"], color="darkorange")
        ax.axhline(80, linestyle="--", color="red", alpha=0.5, label="HIC 참고선")
        ax.set_title(f"머리 가속도 (HIC15={hic15:.0f})")
        ax.set_xlabel("시간 (ms)")
        ax.set_ylabel("가속도 (g)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 흉부 가속도
        ax = axes[1, 0]
        ax.plot(t, history["torso_acc_g"], color="mediumseagreen")
        ax.axhline(60, linestyle="--", color="red", alpha=0.5, label="기준 60g")
        ax.set_title(f"흉부 가속도 (peak={chest_g:.1f}g)")
        ax.set_xlabel("시간 (ms)")
        ax.set_ylabel("가속도 (g)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 차량 속도 감속 (충돌 유무 확인용)
        ax = axes[1, 1]
        decel = -np.gradient(history["veh_speed"], [x / 1000.0 for x in t])
        ax.plot(t, decel, color="purple")
        ax.set_title("차량 감속도 (충돌 발생 확인)")
        ax.set_xlabel("시간 (ms)")
        ax.set_ylabel("감속도 (m/s²)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = f"results/plots/collision_a{args.angle:.0f}_v{args.speed:.0f}.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"\n  그래프 저장 완료: {out_path}")
        print(f"  ↳ scp 또는 RunPod 파일 탐색기로 다운로드하여 확인하세요.\n")

    except ImportError:
        print("  matplotlib 없음 — pip install matplotlib 으로 설치 후 재시도")


# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_test_episode()

    if args.stream:
        print("\n스트리밍 모드: 브라우저 연결 후 Enter 로 종료...")
        input()

    sim_app.close()
    print("테스트 완료.")
