"""
PPO 학습 소요시간 측정 스크립트.
10 에피소드 PPO 학습 실행 → 에피소드당/전체 시간 측정.
"""
import os
import sys
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
import yaml
from env.airbag_env import AirbagEnv
from rl.ppo import PPOAgent

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

N_EPISODES = 50
BATCH_SIZE = cfg["ppo"]["batch_size"]  # 256

WINDOW = 10  # 이동 평균 윈도우

print(f"\n{'='*60}")
print(f"PPO 학습 추세 확인 — {N_EPISODES} 에피소드")
print(f"batch_size={BATCH_SIZE}, ppo_epochs={cfg['ppo']['epochs']}")
print(f"{'='*60}\n")

env = AirbagEnv(headless=True)
agent = PPOAgent(
    state_dim=cfg["env"]["state_dim"],
    lr=cfg["ppo"]["lr"],
    gamma=cfg["ppo"]["gamma"],
    clip=cfg["ppo"]["clip"],
    epochs=cfg["ppo"]["epochs"],
    lam=cfg["ppo"].get("lam", 0.95),
    entropy_coeff=cfg["ppo"].get("entropy_coeff", 0.01),
)

buffer = []
ep_times = []
rewards = []
update_count = 0
nan_detected = False
last_losses = {}

wall_start = time.time()

for ep in range(1, N_EPISODES + 1):
    ep_start = time.time()

    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    step_count = 0

    while not done:
        action, log_prob = agent.select_action(obs)
        next_obs, reward, done, _, info = env.step(action)

        if not np.isfinite(reward):
            nan_detected = True
            reward = 0.0

        buffer.append({
            "state":      obs,
            "action":     action,
            "log_prob":   log_prob,
            "reward":     reward,
            "next_state": next_obs,
            "done":       done,
        })
        ep_reward += reward
        step_count += 1
        obs = next_obs

    ep_elapsed = time.time() - ep_start
    ep_times.append(ep_elapsed)
    rewards.append(ep_reward)

    # PPO 업데이트 (buffer >= batch_size)
    update_str = ""
    if len(buffer) >= BATCH_SIZE:
        # 10 에피소드마다 한 번씩만 상세 debug 출력
        do_debug = (ep % WINDOW == 0)
        losses = agent.update(buffer, debug=do_debug)
        buffer = []
        update_count += 1
        last_losses = losses
        al = losses.get("actor_loss", float("nan"))
        cl = losses.get("critic_loss", float("nan"))
        update_str = f"  [UPD#{update_count}] actor={al:.4f}  critic={cl:.4f}"

    ep_info = info if info else {}
    hic15   = ep_info.get("hic15", float("nan"))
    chest_g = ep_info.get("chest_g", float("nan"))

    # 이동 평균 계산
    recent = rewards[-WINDOW:]
    mean_r = float(np.mean(recent)) if len(recent) >= 1 else float("nan")

    print(
        f"EP {ep:>2}/{N_EPISODES}  "
        f"reward={ep_reward:>14.1f}  "
        f"mean{WINDOW}={mean_r:>14.1f}  "
        f"HIC15={hic15:>10.1f}  "
        f"chest_g={chest_g:6.1f}g"
        f"{update_str}"
    )

env.close()
sim_app.close()

# ── 최종 요약 ──────────────────────────────────────────────────────────────
wall_total = time.time() - wall_start
mean_ep    = float(np.mean(ep_times))
std_ep     = float(np.std(ep_times))

print(f"\n{'='*60}")
print("측정 결과 요약")
print(f"{'='*60}")
print(f"에피소드당 평균 소요시간  : {mean_ep:.1f}초  (±{std_ep:.1f}초)")
print(f"에피소드당 최소/최대       : {min(ep_times):.1f}초 / {max(ep_times):.1f}초")
print(f"10 에피소드 총 소요시간   : {sum(ep_times):.1f}초 = {sum(ep_times)/60:.1f}분")
print(f"(Isaac Sim 포함 전체 wall): {wall_total:.1f}초 = {wall_total/60:.1f}분")
print()
print(f"── 외삽 추정 (mean={mean_ep:.1f}s/ep) ──")
for n in [100, 500, 1000, 2000]:
    secs = mean_ep * n
    if secs < 3600:
        print(f"  {n:>5} 에피소드 : {secs/60:>6.1f}분")
    else:
        print(f"  {n:>5} 에피소드 : {secs/3600:>5.1f}시간  ({secs/60:.0f}분)")
print()
print(f"PPO update() 호출 횟수    : {update_count}회")
if last_losses:
    print(f"마지막 업데이트 손실      : actor={last_losses.get('actor_loss',float('nan')):.4f}  critic={last_losses.get('critic_loss',float('nan')):.4f}")
print(f"NaN 발생 여부             : {'있음 ⚠' if nan_detected else '없음 ✓'}")
print(f"평균 보상                 : {np.mean(rewards):.2f}")
print(f"{'='*60}")

print(f"""
┌─────────────────────────────────────────────────────┐
│  에피소드당 평균 소요시간: {mean_ep:.0f}초
│  10 에피소드 총 소요시간: {sum(ep_times):.0f}초 / {sum(ep_times)/60:.1f}분
│  100 에피소드 예상 소요시간: {mean_ep*100/60:.0f}분
│  500 에피소드 예상 소요시간: {mean_ep*500/60:.0f}분 / {mean_ep*500/3600:.1f}시간
│
│  학습 정상 작동 여부: {'Y' if update_count > 0 else 'N (update 없음)'}
│  NaN 발생 여부: {'Y ⚠' if nan_detected else 'N'}
└─────────────────────────────────────────────────────┘
""")
