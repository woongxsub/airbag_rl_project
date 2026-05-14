import argparse
import os
import yaml
import numpy as np

# Isaac Sim은 반드시 SimulationApp을 가장 먼저 초기화해야 함
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--config", default="config/config.yaml")
parser.add_argument("--mode", choices=["train", "baseline"], default="train")
args = parser.parse_args()

sim_app = SimulationApp({"headless": args.headless})

# SimulationApp 이후에 import
from env.airbag_env import AirbagEnv
from rl.ppo import PPOAgent
from baseline.rule_based import rule_based_policy

with open(args.config) as f:
    cfg = yaml.safe_load(f)

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)


def run_baseline(episodes=200):
    env = AirbagEnv(headless=args.headless)
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        angle = env.scenario["angle"]
        action_matrix = rule_based_policy(angle)
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
    env = AirbagEnv(headless=args.headless)
    agent = PPOAgent(
        state_dim=7,
        lr=cfg["ppo"]["lr"],
        gamma=cfg["ppo"]["gamma"],
        clip=cfg["ppo"]["clip"],
        epochs=cfg["ppo"]["epochs"],
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
                "state": obs,
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
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
