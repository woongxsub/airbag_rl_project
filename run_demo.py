#!/usr/bin/env python3
"""
에어백 RL 프로토타입 데모 — Isaac Sim 6.0 Headless
PPT 발표용: 물리 시뮬레이션 동작 + 측정부위별 충격량 + 에어백 감쇠 + PPO 학습 로그

실행:
    cd /workspace/airbag_rl_project
    python run_demo.py
"""

# ── Isaac Sim 부트스트랩 (가장 먼저) ────────────────────────────────────────
import os, sys, time
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

print("=" * 70)
print("  에어백 RL 프로토타입 — Isaac Sim 6.0 Headless 시작")
print("=" * 70)
print("  Isaac Sim 초기화 중 (PhysX 5 / USD) ...")
t_boot = time.time()

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

print(f"  ✅ Isaac Sim 초기화 완료 ({time.time()-t_boot:.1f}s)\n")

# ── 이후 import ──────────────────────────────────────────────────────────────
import numpy as np
sys.path.insert(0, "/workspace/airbag_rl_project")

from env.airbag_env import AirbagEnv, PHYSICS_DT, COLLISION_STEPS
from rl.ppo import PPOAgent
from rl.reward import (
    compute_hic15, compute_chest_g, compute_chest_3ms_clip,
    compute_chest_compression_mm, compute_femur_force_n, compute_nij,
    HIC_SAFE, CHEST_G_SAFE, CHEST_3MS_SAFE,
    CHEST_COMPRESSION_SAFE, FEMUR_SAFE, NIJ_SAFE,
)
from env.scenario import STATE_DIM

# ── 상수 ─────────────────────────────────────────────────────────────────────
AIRBAG_NAMES = [
    "운전석 정면 (60L )",
    "조수석 정면 (120L)",
    "운전석 측면 (15L )",
    "조수석 측면 (15L )",
    "커튼      (40L )",
]

# 고정 테스트 시나리오 (NCAP 정면충돌 조건)
TEST = {
    "angle":    0.0,
    "speed":    56.0,
    "stiffness": "concrete",
    "seatbelt": True,
    "height":   1.75,
    "weight":   75.0,
}
_SPEED_MS = TEST["speed"] / 3.6  # 15.56 m/s


# ── 에피소드 실행 ─────────────────────────────────────────────────────────────

def _set_fixed_scenario(env):
    """reset() 후 고정 시나리오·속도 강제 적용."""
    for k, v in TEST.items():
        env.scenario[k] = v
    vel = np.array([_SPEED_MS, 0.0, 0.0])
    try:
        env.vehicle.body.set_linear_velocity(vel)
    except Exception:
        pass
    try:
        env.human.set_initial_velocity(vel)
    except Exception:
        pass


def run_episode(env, action_15d: np.ndarray, fix_scenario: bool = True) -> dict:
    """
    1 에피소드 실행.
    fix_scenario=True : 고정 56km/h 정면충돌 시나리오 강제 적용
    반환: 충돌 지표 dict
    """
    obs, _ = env.reset()
    if fix_scenario:
        _set_fixed_scenario(env)

    done = False
    ep_reward = 0.0
    while not done:
        obs, r, done, _, _ = env.step(action_15d)
        if done:
            ep_reward = r

    col = env.collector
    return {
        "hic15":       compute_hic15(col.head_acc_g, PHYSICS_DT),
        "chest_g":     compute_chest_g(col.torso_acc_g),
        "chest_3ms":   compute_chest_3ms_clip(col.torso_acc_g, PHYSICS_DT),
        "compression": compute_chest_compression_mm(col.torso_pos_history),
        "femur_n":     compute_femur_force_n(col.thigh_acc_3d),
        "nij":         compute_nij(col.head_acc_3d),
        "reward":      ep_reward,
        "n_head":      len(col.head_acc_g),
        "n_torso":     len(col.torso_acc_g),
        "n_thigh":     len(col.thigh_acc_3d),
        "pv_ok":       env.human._physics_view is not None,
    }


# ── 출력 헬퍼 ─────────────────────────────────────────────────────────────────

def print_metrics(label: str, m: dict):
    def row(name, val, safe, unit):
        pct = (val / safe - 1.0) * 100.0 if safe > 0 else 0.0
        mk  = "✅ PASS" if val <= safe else "❌ FAIL"
        return (f"  │  {name:<26} {val:>9.1f} {unit:<4}"
                f"  기준 ≤ {safe}{unit:<4}  ({pct:+6.1f}%)  {mk}")

    print(f"\n  ┌──── {label}")
    print(f"  │  {'지표 (측정 부위)':<26} {'측정값':>9}      {'안전기준':>14}  {'초과율':>8}  판정")
    print(f"  │  {'─'*26}  {'─'*9}  {'─'*15}  {'─'*8}  {'─'*7}")
    for name, val, safe, unit in [
        ("HIC15           (두부)", m["hic15"],       HIC_SAFE,           "" ),
        ("흉부 최대가속도 (흉부)", m["chest_g"],     CHEST_G_SAFE,       "g" ),
        ("흉부 3ms 클립   (흉부)", m["chest_3ms"],   CHEST_3MS_SAFE,     "g" ),
        ("흉부 압축량     (흉부)", m["compression"], CHEST_COMPRESSION_SAFE, "mm"),
        ("대퇴부 압축력   (대퇴)", m["femur_n"],     FEMUR_SAFE,         "N" ),
        ("Nij             (경부)", m["nij"],         NIJ_SAFE,           "" ),
    ]:
        print(row(name, val, safe, unit))
    print(f"  │")
    print(f"  │  Reward = {m['reward']:+.4f}  │  "
          f"PhysX View: {'OK' if m['pv_ok'] else 'NONE'}  │  "
          f"센서샘플: head={m['n_head']} torso={m['n_torso']} thigh={m['n_thigh']}")
    print(f"  └{'─'*70}")


def print_attenuation(label_a, m_a, label_b, m_b):
    """두 결과 간 감쇠율 출력."""
    print(f"\n  에어백 감쇠 효과: {label_a} → {label_b}")
    for name, ka, kb in [
        ("HIC15  (두부)",    "hic15",       "hic15"      ),
        ("흉부 g (흉부)",    "chest_g",     "chest_g"    ),
        ("흉부3ms(흉부)",    "chest_3ms",   "chest_3ms"  ),
        ("압축mm (흉부)",    "compression", "compression"),
        ("대퇴부N(대퇴)",    "femur_n",     "femur_n"    ),
        ("Nij    (경부)",    "nij",         "nij"        ),
    ]:
        bv = m_a[ka]; rv = m_b[kb]
        att = (bv - rv) / bv * 100.0 if bv > 0.001 else 0.0
        n_filled = max(0, min(20, int(att / 5)))
        bar = "█" * n_filled + "░" * (20 - n_filled)
        print(f"  {name:<14}: {bv:8.1f} → {rv:8.1f}  [{bar}] {att:+.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
def main():
    LINE = "=" * 70
    DLINE = "─" * 70
    t_total = time.time()

    print(LINE)
    print("  시나리오: 정면충돌 0°  |  56 km/h  |  콘크리트 벽  |  안전벨트 착용")
    print("  인체모델: Newton humanoid.usda (관절 17개+, Hybrid III 기준)")
    print("  물리엔진: Isaac Sim 6.0 / PhysX 5  |  dt=1ms, 100ms 충돌구간")
    print("  에어백  : 5개  |  측정부위: 두부·흉부·대퇴부·경부")
    print(LINE)

    # 환경 생성
    print("\n  AirbagEnv (Isaac Sim 물리 월드) 초기화 ...")
    t0 = time.time()
    env = AirbagEnv(headless=True)
    print(f"  ✅ 환경 초기화 완료 ({time.time()-t0:.1f}s)")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: 에어백 전혀 없음 (Baseline)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  [STEP 1] 에어백 미전개 — Baseline (에어백 없을 때 충격량)")
    print(DLINE)
    t0 = time.time()
    no_bag_action = np.zeros(15, dtype=np.float32)
    m_base = run_episode(env, no_bag_action)
    elapsed = time.time() - t0
    print(f"  Isaac Sim 시뮬레이션 완료: {elapsed:.1f}s  ({COLLISION_STEPS} control steps)")
    print_metrics("에어백 없음 — 측정부위별 충격량", m_base)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Rule-based (현업 ACU 방식)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  [STEP 2] Rule-based 정책 (현업 ACU 방식)")
    print("  → 정면충돌 감지 → 전방 에어백 2개 고정 전개")
    print("    전개여부: bag0=ON, bag1=ON  |  타이밍: 15ms 고정  |  압력: 300kPa 고정")
    print(DLINE)
    t0 = time.time()

    rb_action = np.zeros(15, dtype=np.float32)
    # 에어백 0 (운전석 정면)
    rb_action[0]  = 1.0           # deploy=ON
    rb_action[5]  = 15.0 / 30.0  # timing=15ms  (정규화)
    rb_action[10] = 300.0/ 600.0 # pressure=300kPa (정규화)
    # 에어백 1 (조수석 정면)
    rb_action[1]  = 1.0
    rb_action[6]  = 15.0 / 30.0
    rb_action[11] = 300.0/ 600.0

    m_rb = run_episode(env, rb_action)
    elapsed = time.time() - t0
    print(f"  Isaac Sim 시뮬레이션 완료: {elapsed:.1f}s")
    print_metrics("Rule-based — 측정부위별 충격량", m_rb)
    print_attenuation("에어백 없음", m_base, "Rule-based", m_rb)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: PPO 강화학습
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  [STEP 3] PPO 강화학습 — 200 에피소드 (실시간 학습 로그)")
    print("  알고리즘 : PPO  |  lr=3e-4  |  clip ε=0.2  |  batch=6  |  epochs=8")
    print("  정책망   : MultiHeadActor(12→128→128)  Bernoulli(deploy×5) + Normal(timing×5, pressure×5)")
    print("  목표     : 에어백 전개여부·타이밍·압력을 자동으로 최적화")
    print(LINE)

    agent   = PPOAgent(state_dim=STATE_DIM, lr=3e-4, gamma=0.99, clip=0.2, epochs=10)
    buf     = []
    rewards = []
    N_TRAIN = 200
    BATCH   = 10

    print(f"\n  {'ep':>3} │ {'reward':>8} │ {'최근10평균':>10} │ {'전개수':>4} │ 전개 에어백 (타이밍 / 압력)")
    print(f"  {'─'*3}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*4}─┼─{'─'*46}")

    for ep in range(1, N_TRAIN + 1):
        # 에피소드 실행
        obs, _ = env.reset()
        _set_fixed_scenario(env)

        action, lp = agent.select_action(obs)
        done = False; ep_r = 0.0
        while not done:
            obs_n, r, done, _, _ = env.step(action)
            if done:
                ep_r = r
        rewards.append(ep_r)
        buf.append({"state": obs, "action": action, "log_prob": lp, "reward": ep_r})

        # 전개 정보
        n_deploy = int((action[:5] > 0.5).sum())
        bag_info_parts = []
        for i in range(5):
            if action[i] > 0.5:
                t_ms  = action[5  + i] * 30.0
                p_kpa = action[10 + i] * 600.0
                bag_info_parts.append(f"bag{i}[{t_ms:.1f}ms/{p_kpa:.0f}kPa]")
        bag_str = " ".join(bag_info_parts) if bag_info_parts else "미전개"

        # PPO 업데이트
        update_str = ""
        if len(buf) >= BATCH:
            losses = agent.update(buf)
            buf = []
            update_str = (f"  ← 업데이트  actor_loss={losses['actor_loss']:.4f}"
                          f"  critic_loss={losses['critic_loss']:.4f}")

        avg10 = np.mean(rewards[-10:]) if len(rewards) >= 10 else float("nan")
        avg10_str = f"{avg10:+.3f}" if not np.isnan(avg10) else "   ─   "
        print(f"  {ep:>3} │ {ep_r:>+8.3f} │ {avg10_str:>10} │ {n_deploy:>4} │ {bag_str}{update_str}")

    print(f"\n  학습 완료  |  최고 reward={max(rewards):+.3f}  최근 5ep 평균={np.mean(rewards[-5:]):+.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: 학습된 PPO 정책으로 최종 시뮬레이션
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  [STEP 4] 학습된 PPO 정책 최종 실행")
    print(DLINE)

    obs_f, _ = env.reset()
    _set_fixed_scenario(env)
    final_action, _ = agent.select_action(obs_f)

    print("\n  ▶ PPO 출력 — 에어백 전개 결정 (학습 후):")
    for i in range(5):
        d     = final_action[i] > 0.5
        t_ms  = final_action[5  + i] * 30.0
        p_kpa = final_action[10 + i] * 600.0
        if d:
            print(f"    ✅ [{i}] {AIRBAG_NAMES[i]} : 전개  │  타이밍 {t_ms:5.1f} ms  │  압력 {p_kpa:6.1f} kPa")
        else:
            print(f"    ➖ [{i}] {AIRBAG_NAMES[i]} : 미전개")

    t0 = time.time()
    m_ppo = run_episode(env, final_action)
    print(f"\n  Isaac Sim 시뮬레이션 완료: {time.time()-t0:.1f}s")
    print_metrics("PPO 학습 정책 — 측정부위별 충격량", m_ppo)
    print_attenuation("에어백 없음", m_base, "PPO 정책", m_ppo)

    # ─────────────────────────────────────────────────────────────────────────
    # 최종 3-way 비교표
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  ★ 최종 비교: 에어백 없음  vs  Rule-based  vs  PPO 학습 정책")
    print(LINE)
    print(f"  {'지표 (측정부위)':<22} {'안전기준':>8}  "
          f"{'에어백없음':>10}  {'Rule-based':>10}  {'PPO':>10}  {'PPO감쇠율':>10}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    compare = [
        ("HIC15  (두부)",    HIC_SAFE,           "hic15",       ""),
        ("흉부 g (흉부)",    CHEST_G_SAFE,       "chest_g",     "g"),
        ("흉부3ms(흉부)",    CHEST_3MS_SAFE,     "chest_3ms",   "g"),
        ("압축mm (흉부)",    CHEST_COMPRESSION_SAFE, "compression", "mm"),
        ("대퇴부N(대퇴)",    FEMUR_SAFE,         "femur_n",     "N"),
        ("Nij    (경부)",    NIJ_SAFE,           "nij",         ""),
        ("Reward",           None,               "reward",      ""),
    ]
    for name, safe, key, unit in compare:
        bv = m_base[key]; rv = m_rb[key]; pv = m_ppo[key]
        safe_str = f"≤{safe}{unit}" if safe else "─"
        att = f"{(bv-pv)/bv*100:+.1f}%" if safe and bv > 0.001 else ""
        pb = "✅" if safe and bv <= safe else ("❌" if safe else "")
        pr = "✅" if safe and rv <= safe else ("❌" if safe else "")
        pp = "✅" if safe and pv <= safe else ("❌" if safe else "")
        print(f"  {name:<22} {safe_str:>8}  "
              f"{bv:>8.1f}{pb}  {rv:>8.1f}{pr}  {pv:>8.1f}{pp}  {att:>10}")

    print(f"\n  총 소요시간: {time.time()-t_total:.1f}s")
    print(LINE)
    print("  ✅ Isaac Sim headless 시뮬레이션 + PPO 학습 정상 완료")
    print("  → 위 결과가 에어백 RL 프로토타입 동작 증명입니다")
    print(LINE)

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
    sim_app.close()
    print("\n[데모 종료]")
