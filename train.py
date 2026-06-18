import os
import sys
import argparse
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
parser.add_argument("--config", default="config/config.yaml")
parser.add_argument("--mode", choices=["train", "baseline"], default="train")
args = parser.parse_args()

# lavapipe Vulkan (GUI/stream 모드에서 GPU 없을 때)
if args.stream or args.gui:
    os.environ.setdefault(
        "VK_ICD_FILENAMES",
        "/usr/share/vulkan/icd.d/lvp_icd.json",
    )

from isaacsim import SimulationApp

sim_config = {"headless": True}
if args.stream:
    sim_config["headless"]    = False
    sim_config["width"]       = 1280
    sim_config["height"]      = 720
    sim_config["livestream"]  = 1
elif args.gui:
    # noVNC 모드: GUI 창만 띄움 (WebRTC 없음, X11 디스플레이에 렌더링)
    sim_config["headless"]    = False
    sim_config["width"]       = 1280
    sim_config["height"]      = 720

sim_app = SimulationApp(sim_config)

import carb
carb.settings.get_settings().set("/physics/cudaDevice", 0)
if args.stream:
    carb.settings.get_settings().set("/app/livestream/websocket/server_port", 8211)
    print("[train] WebRTC streaming 활성화 — port 8211")

# SimulationApp 이후에 import
from env.airbag_env import AirbagEnv
from rl.ppo import PPOAgent
from baseline.rule_based import rule_based_policy

with open(args.config) as f:
    cfg = yaml.safe_load(f)

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)


def run_baseline(episodes=200):
    env = AirbagEnv(headless=True, debug=args.debug)
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        angle             = env.scenario["angle"]
        is_rollover       = bool(env.scenario.get("is_rollover",       False))
        passenger_present = bool(env.scenario.get("passenger_present", True))
        action_matrix = rule_based_policy(angle, is_rollover, passenger_present)
        # airbag_env action 형식으로 변환 [deploy*5, timing*5, pressure*5]
        action = np.concatenate([
            action_matrix[:, 0],
            action_matrix[:, 1] / 30.0,
            action_matrix[:, 2] / 600.0,
        ])
        total_r = 0.0
        done = False
        while not done:
            obs, r, done, _, _ = env.step(action)
            total_r += r
        rewards.append(total_r)
        if (ep + 1) % 50 == 0:
            print(f"[Baseline] ep {ep+1} | mean_reward: {np.mean(rewards[-50:]):.4f}")
    env.close()
    np.save("results/logs/baseline_rewards.npy", rewards)
    print(f"Baseline mean reward: {np.mean(rewards):.4f}")


def run_train():
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

    total_episodes = cfg["train"]["total_episodes"]
    batch_size = cfg["ppo"]["batch_size"]
    save_interval = cfg["train"]["save_interval"]
    log_interval = cfg["train"]["log_interval"]

    all_rewards = []
    buffer = []

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        log_prob_sum = 0.0
        step_count = 0

        while not done:
            action, log_prob = agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            buffer.append({
                "state":      obs,
                "action":     action,
                "log_prob":   log_prob,
                "reward":     reward,
                "next_state": next_obs,
                "done":       done,
            })
            ep_reward += reward
            log_prob_sum += log_prob
            step_count += 1
            obs = next_obs

        all_rewards.append(ep_reward)

        if len(buffer) >= batch_size:
            losses = agent.update(buffer)
            buffer = []

        if ep % log_interval == 0:
            mean_r = np.mean(all_rewards[-log_interval:])
            print(f"ep {ep}/{total_episodes} | mean_reward: {mean_r:.4f}")

        if ep % save_interval == 0:
            agent.save(f"results/models/ppo_ep{ep}.pt")
            np.save("results/logs/train_rewards.npy", all_rewards)

    env.close()
    sim_app.close()
    print("Training done.")


if args.mode == "baseline":
    run_baseline()
else:
    run_train()
