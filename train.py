import os
import sys
import csv
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
parser.add_argument("--exclude-explosions", action="store_true",
                     help="HIC15 폭발 에피소드(HIC15>1M)를 PPO buffer에서 제외 (실험 B)")
parser.add_argument("--episodes", type=int, default=None,
                     help="config의 train.total_episodes를 덮어씀 (짧은 비교 실험용)")
parser.add_argument("--tag", default="",
                     help="결과 파일명에 붙일 접미사 (예: A, B) — A/B 비교 실험 시 결과 덮어쓰기 방지")
args = parser.parse_args()

_suffix = f"_{args.tag}" if args.tag else ""

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

    total_episodes = args.episodes if args.episodes is not None else cfg["train"]["total_episodes"]
    batch_size = cfg["ppo"]["batch_size"]
    save_interval = cfg["train"]["save_interval"]
    log_interval = cfg["train"]["log_interval"]

    all_rewards = []
    buffer = []
    n_explosions = 0

    hic_log_path    = f"results/logs/train_hic_log{_suffix}.csv"
    update_log_path = f"results/logs/train_update_log{_suffix}.csv"
    hic_log_f    = open(hic_log_path, "w", newline="")
    update_log_f = open(update_log_path, "w", newline="")
    hic_writer    = csv.writer(hic_log_f)
    update_writer = csv.writer(update_log_f)
    hic_writer.writerow(["episode", "hic15", "is_explosion", "reward"])
    update_writer.writerow(["episode", "actor_loss", "critic_loss", "buffer_size"])

    print(f"[train] exclude_explosions={args.exclude_explosions} tag='{args.tag}' "
          f"episodes={total_episodes}")

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        log_prob_sum = 0.0
        step_count = 0
        ep_transitions = []
        info = {}

        while not done:
            action, log_prob = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            ep_transitions.append({
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

        hic15        = info.get("hic15", float("nan"))
        is_explosion = bool(info.get("is_explosion", False))
        all_rewards.append(ep_reward)
        hic_writer.writerow([ep, hic15, is_explosion, ep_reward])

        if is_explosion:
            n_explosions += 1

        # 실험 B(--exclude-explosions): 폭발 에피소드의 transition은 buffer에서
        # 완전히 제외한다 (reward만 클리핑하지 않는 이유는 아래 run_train 밖
        # 보고 참고 — 상태/행동 시퀀스 자체가 수치 불안정 구간이라 gradient에
        # 유효한 신호가 없고, 거대한 |advantage| 하나가 배치 정규화를 왜곡함).
        if args.exclude_explosions and is_explosion:
            print(f"[train] ep {ep}: HIC15={hic15:.1f} > threshold → buffer 제외")
        else:
            buffer.extend(ep_transitions)

        if len(buffer) >= batch_size:
            losses = agent.update(buffer)
            update_writer.writerow([
                ep,
                losses.get("actor_loss"),
                losses.get("critic_loss"),
                len(buffer),
            ])
            buffer = []

        if ep % log_interval == 0:
            mean_r = np.mean(all_rewards[-log_interval:])
            explosion_rate = n_explosions / ep * 100.0
            print(f"ep {ep}/{total_episodes} | mean_reward: {mean_r:.4f} | "
                  f"explosion_rate: {explosion_rate:.1f}%")

        if ep % save_interval == 0:
            agent.save(f"results/models/ppo_ep{ep}{_suffix}.pt")
            np.save(f"results/logs/train_rewards{_suffix}.npy", all_rewards)

    hic_log_f.close()
    update_log_f.close()
    agent.save(f"results/models/ppo_final{_suffix}.pt")
    np.save(f"results/logs/train_rewards{_suffix}.npy", all_rewards)

    explosion_rate = n_explosions / total_episodes * 100.0
    print(f"[train] done. total explosions: {n_explosions}/{total_episodes} "
          f"({explosion_rate:.1f}%)")

    env.close()
    sim_app.close()
    print("Training done.")


if args.mode == "baseline":
    run_baseline()
else:
    run_train()
