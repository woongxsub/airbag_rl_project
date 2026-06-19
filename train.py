import os
import sys
import argparse
import time
import yaml
import numpy as np

# ── Isaac Sim 부트스트랩 ─────────────────────────────────────────────────
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--stream",   action="store_true", help="WebRTC 스트리밍 활성화 (port 8211)")
parser.add_argument("--gui",      action="store_true", help="noVNC용 GUI 창 표시 (DISPLAY 환경변수 필요)")
parser.add_argument("--debug",    action="store_true", help="스텝별 샘플 수 출력 (Dense reward 검증용)")
parser.add_argument("--config",   default="config/config.yaml")
parser.add_argument("--mode",     choices=["train", "curriculum", "baseline"], default="train")
parser.add_argument("--episodes", type=int, default=None,
                    help="총 에피소드 수 (없으면 config의 total_episodes 사용)")
args = parser.parse_args()

if args.stream or args.gui:
    os.environ.setdefault(
        "VK_ICD_FILENAMES",
        "/usr/share/vulkan/icd.d/lvp_icd.json",
    )

from isaacsim import SimulationApp

sim_config = {"headless": True}
if args.stream:
    sim_config["headless"]   = False
    sim_config["width"]      = 1280
    sim_config["height"]     = 720
    sim_config["livestream"] = 1
elif args.gui:
    sim_config["headless"] = False
    sim_config["width"]    = 1280
    sim_config["height"]   = 720

sim_app = SimulationApp(sim_config)

import carb
carb.settings.get_settings().set("/physics/cudaDevice", 0)
if args.stream:
    carb.settings.get_settings().set("/app/livestream/websocket/server_port", 8211)
    print("[train] WebRTC streaming 활성화 — port 8211")

from env.airbag_env import AirbagEnv
from rl.ppo import PPOAgent
from baseline.rule_based import rule_based_policy

with open(args.config) as f:
    cfg = yaml.safe_load(f)

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/logs",   exist_ok=True)


# ── 커리큘럼 단계 결정 ──────────────────────────────────────────────────────

def get_stage(ep: int, total: int):
    """
    에피소드 번호 → (커리큘럼 단계, violation_coeff).

    단계 정의 (누적 확장형):
      Stage 1 (1~1/3)   : 정면±45°, 30-60 km/h, rollover=0%   — violation_coeff 2.0
      Stage 2 (1/3~2/3) : 전 각도,   30-90 km/h, rollover=5%   — violation_coeff 5.0
      Stage 3 (2/3~끝)  : 전 각도,  20-120 km/h, rollover=15%  — violation_coeff 8.0
    """
    if ep <= total // 3:
        return 1, 2.0
    elif ep <= 2 * total // 3:
        return 2, 5.0
    else:
        return 3, 8.0


# ── 베이스라인 ──────────────────────────────────────────────────────────────

def run_baseline(episodes: int = 200):
    env = AirbagEnv(headless=True, debug=args.debug)
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        angle             = env.scenario["angle"]
        is_rollover       = bool(env.scenario.get("is_rollover",       False))
        passenger_present = bool(env.scenario.get("passenger_present", True))
        action_matrix = rule_based_policy(angle, is_rollover, passenger_present)
        action = np.concatenate([
            action_matrix[:, 0],
            action_matrix[:, 1] / 30.0,
            action_matrix[:, 2] / 250.0,
        ])
        total_r = 0.0
        done = False
        while not done:
            obs, r, done, _, _ = env.step(action)
            total_r += r
        rewards.append(total_r)
        if (ep + 1) % 50 == 0:
            print(f"[Baseline] ep {ep+1}/{episodes} | "
                  f"mean_reward={np.mean(rewards[-50:]):.1f}", flush=True)
    env.close()
    np.save("results/logs/baseline_rewards.npy", rewards)
    print(f"[Baseline] done. mean_reward={np.mean(rewards):.1f}")


# ── 학습 공통 루프 ──────────────────────────────────────────────────────────

def run_train(curriculum: bool = False):
    label         = "curriculum" if curriculum else "pure"
    N             = args.episodes or cfg["train"]["total_episodes"]
    batch_size    = cfg["ppo"]["batch_size"]
    save_interval = cfg["train"]["save_interval"]
    log_interval  = cfg["train"]["log_interval"]

    print(f"\n{'='*65}")
    print(f"[{label.upper()}] 학습 시작  총={N}ep  batch={batch_size}  "
          f"ppo_epochs={cfg['ppo']['epochs']}")
    print(f"{'='*65}\n", flush=True)

    env = AirbagEnv(headless=True, debug=args.debug)
    agent = PPOAgent(
        state_dim=cfg["env"]["state_dim"],
        lr=cfg["ppo"]["lr"],
        gamma=cfg["ppo"]["gamma"],
        clip=cfg["ppo"]["clip"],
        epochs=cfg["ppo"]["epochs"],
        lam=cfg["ppo"].get("lam", 0.95),
        entropy_coeff=cfg["ppo"].get("entropy_coeff", 0.01),
    )

    all_rewards    = []
    all_hic15      = []
    all_chest_g    = []
    all_crit_loss  = []
    buffer         = []
    current_stage  = -1
    last_cl        = float("nan")
    wall_start     = time.time()

    for ep in range(1, N + 1):

        # ── 커리큘럼: 에피소드마다 단계 설정 ─────────────────────────────
        if curriculum:
            stage, vc = get_stage(ep, N)
            if stage != current_stage:
                current_stage = stage
                print(f"[Curriculum] ===== Stage {stage} 시작 (EP {ep}/{N}) "
                      f"violation_coeff={vc} =====", flush=True)
            env.sampler.stage    = stage
            env.violation_coeff  = vc
        # else: sampler.stage=0(Pure), violation_coeff=5.0 고정

        obs, _ = env.reset()
        done      = False
        ep_reward = 0.0
        ep_info   = {}

        while not done:
            action, log_prob = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            if info:
                ep_info = info
            buffer.append({
                "state":      obs,
                "action":     action,
                "log_prob":   log_prob,
                "reward":     reward,
                "next_state": next_obs,
                "done":       done,
            })
            ep_reward += reward
            obs = next_obs

        all_rewards.append(ep_reward)
        all_hic15.append(ep_info.get("hic15",   float("nan")))
        all_chest_g.append(ep_info.get("chest_g", float("nan")))

        # ── PPO 업데이트 ───────────────────────────────────────────────────
        if len(buffer) >= batch_size:
            losses = agent.update(buffer)
            buffer = []
            last_cl = losses.get("critic_loss", float("nan"))
            all_crit_loss.append(last_cl)

        # ── 주기적 로그 ────────────────────────────────────────────────────
        if ep % log_interval == 0:
            mean_r  = np.nanmean(all_rewards[-log_interval:])
            mean_h  = np.nanmean(all_hic15[-log_interval:])
            mean_c  = np.nanmean(all_chest_g[-log_interval:])
            elapsed = time.time() - wall_start
            stage_str = f" st={current_stage}" if curriculum else ""
            print(
                f"[{label}] ep {ep:>4}/{N}{stage_str} | "
                f"mean_r={mean_r:>10.0f} | "
                f"HIC15={mean_h:>8.0f} | "
                f"chest_g={mean_c:>6.1f}g | "
                f"critic_loss={last_cl:.3f} | "
                f"t={elapsed/60:.1f}min",
                flush=True,
            )

        # ── 주기적 저장 ────────────────────────────────────────────────────
        if ep % save_interval == 0:
            agent.save(f"results/models/{label}_ppo_ep{ep}.pt")
            _save_logs(label, all_rewards, all_hic15, all_chest_g, all_crit_loss)

    # ── 최종 저장 ──────────────────────────────────────────────────────────
    agent.save(f"results/models/{label}_ppo_final.pt")
    _save_logs(label, all_rewards, all_hic15, all_chest_g, all_crit_loss)

    elapsed = time.time() - wall_start
    print(f"\n[{label.upper()}] 학습 완료  {N}ep  소요={elapsed/60:.1f}분", flush=True)
    print(f"  mean_reward (전체)  : {np.nanmean(all_rewards):.1f}")
    print(f"  mean_HIC15  (전체)  : {np.nanmean(all_hic15):.0f}")
    print(f"  mean_chest_g (전체) : {np.nanmean(all_chest_g):.1f}g")
    print(f"  마지막 critic_loss  : {last_cl:.4f}", flush=True)

    env.close()
    sim_app.close()


def _save_logs(label, rewards, hic15, chest_g, crit_loss):
    np.save(f"results/logs/{label}_ppo_rewards.npy",   rewards)
    np.save(f"results/logs/{label}_ppo_hic15.npy",     hic15)
    np.save(f"results/logs/{label}_ppo_chest_g.npy",   chest_g)
    np.save(f"results/logs/{label}_ppo_critic_loss.npy", crit_loss)


# ── 진입점 ──────────────────────────────────────────────────────────────────

if args.mode == "baseline":
    run_baseline(args.episodes or 200)
elif args.mode == "curriculum":
    run_train(curriculum=True)
else:
    run_train(curriculum=False)
