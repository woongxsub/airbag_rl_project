#!/usr/bin/env python3
"""
에어백 RL 프로토타입 — 발표용 빠른 실행 스크립트

총 100 에피소드:
  Phase 1 : Rule-Based 베이스라인  10 ep
  Phase 2 : PPO 학습               80 ep
  Phase 3 : RL 결정론적 평가       10 ep

결과 파일:
  results/prototype_summary.json
  results/prototype_log.csv
  results/models/prototype_ppo.pt
"""

import os
import sys
import json
import csv
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

# SimulationApp 이후 import
from env.airbag_env import AirbagEnv, TIMING_MAX_MS
from rl.ppo import PPOAgent
from rl.reward import (
    HIC_SAFE, CHEST_G_SAFE, CHEST_3MS_SAFE,
    CHEST_COMPRESSION_SAFE, FEMUR_SAFE, NIJ_SAFE,
)
from baseline.rule_based import rule_based_policy

# ── 하이퍼파라미터 ─────────────────────────────────────────────────
N_BASELINE  = 10
N_TRAIN     = 80
N_EVAL      = 10
BATCH_SIZE  = 60   # 매 에피소드(60 스텝)마다 업데이트
LR          = 5e-4
LOG_EVERY   = 10   # N 에피소드마다 진행 로그

AIRBAG_NAMES = ["운전석 전면", "조수석 전면", "운전석 측면", "조수석 측면", "커튼"]

SAFE_LIMITS = {
    "hic15":                HIC_SAFE,
    "chest_g":              CHEST_G_SAFE,
    "chest_3ms":            CHEST_3MS_SAFE,
    "chest_compression_mm": CHEST_COMPRESSION_SAFE,
    "femur_n":              FEMUR_SAFE,
    "nij":                  NIJ_SAFE,
}
METRIC_LABEL = {
    "hic15":                "HIC15 (두부 상해 지수)",
    "chest_g":              "흉부 최대 가속도 (g)",
    "chest_3ms":            "흉부 3ms 클립 (g)",
    "chest_compression_mm": "흉부 압축량 (mm)",
    "femur_n":              "대퇴부 압축력 (N)",
    "nij":                  "목 상해 지수 (Nij)",
}
METRIC_UNIT = {
    "hic15": "", "chest_g": " g", "chest_3ms": " g",
    "chest_compression_mm": " mm", "femur_n": " N", "nij": "",
}
METRIC_KEYS = list(METRIC_LABEL.keys())

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/logs",   exist_ok=True)

# ── 헬퍼 ──────────────────────────────────────────────────────────

def avg_metrics(metrics_list: list) -> dict:
    if not metrics_list:
        return {k: 0.0 for k in METRIC_KEYS}
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in METRIC_KEYS}


def run_episode_baseline(env):
    obs, _ = env.reset()
    angle             = env.scenario["angle"]
    is_rollover       = bool(env.scenario.get("is_rollover",       False))
    passenger_present = bool(env.scenario.get("passenger_present", True))
    am     = rule_based_policy(angle, is_rollover, passenger_present)
    action = np.concatenate([
        am[:, 0],
        am[:, 1] / TIMING_MAX_MS,
        am[:, 2] / 600.0,
    ])
    done = False; total_r = 0.0; info = {}
    while not done:
        obs, r, done, _, info = env.step(action)
        total_r += r
    raw = env.last_raw_actions.copy() if env.last_raw_actions is not None else np.zeros((5, 3))
    return total_r, info, raw


def run_episode_train(env, agent, buffer):
    obs, _ = env.reset()
    done = False; ep_r = 0.0; info = {}
    while not done:
        action, log_prob = agent.select_action(obs)
        next_obs, r, done, _, info = env.step(action)
        buffer.append({"state": obs, "action": action, "log_prob": log_prob, "reward": r})
        ep_r += r
        obs = next_obs
    raw = env.last_raw_actions.copy() if env.last_raw_actions is not None else np.zeros((5, 3))
    return ep_r, info, raw


def run_episode_eval(env, agent):
    obs, _ = env.reset()
    done = False; total_r = 0.0; info = {}
    while not done:
        action = agent.get_deterministic_action(obs)
        next_obs, r, done, _, info = env.step(action)
        total_r += r
        obs = next_obs
    raw = env.last_raw_actions.copy() if env.last_raw_actions is not None else np.zeros((5, 3))
    return total_r, info, raw


# ── 메인 ──────────────────────────────────────────────────────────

env   = AirbagEnv(headless=True)
agent = PPOAgent(state_dim=12, lr=LR, gamma=0.99, clip=0.2, epochs=10)

print("\n" + "=" * 68)
print("  에어백 RL 프로토타입  |  Isaac Sim 6.0  |  PPO Headless")
print(f"  총 {N_BASELINE + N_TRAIN + N_EVAL} 에피소드 "
      f"(Baseline {N_BASELINE} + Train {N_TRAIN} + Eval {N_EVAL})")
print("=" * 68)

# ── Phase 1: Rule-Based Baseline ──────────────────────────────────
print(f"\n[Phase 1] Rule-Based 베이스라인  ({N_BASELINE} 에피소드)")
bl_rewards = []; bl_metrics_list = []
for ep in range(1, N_BASELINE + 1):
    r, info, _ = run_episode_baseline(env)
    bl_rewards.append(r)
    if info:
        bl_metrics_list.append({k: info[k] for k in METRIC_KEYS})
    print(f"  ep {ep:2d} | reward: {r:+.3f}")

bl_mean_r = float(np.mean(bl_rewards))
bl_metrics = avg_metrics(bl_metrics_list)
print(f"  ▶ 평균 보상: {bl_mean_r:+.3f}")

# ── Phase 2: PPO Training ─────────────────────────────────────────
print(f"\n[Phase 2] PPO 학습  ({N_TRAIN} 에피소드, lr={LR}, batch={BATCH_SIZE})")
train_rewards = []; buffer = []
for ep in range(1, N_TRAIN + 1):
    r, _, _ = run_episode_train(env, agent, buffer)
    train_rewards.append(r)

    if len(buffer) >= BATCH_SIZE:
        agent.update(buffer)
        buffer = []

    if ep % LOG_EVERY == 0:
        window = train_rewards[-LOG_EVERY:]
        mean_r = float(np.mean(window))
        prev_window = train_rewards[-2*LOG_EVERY:-LOG_EVERY]
        trend = "↑" if prev_window and mean_r > np.mean(prev_window) else "→"
        print(f"  ep {ep:3d}/{N_TRAIN} | mean_reward(last {LOG_EVERY}): {mean_r:+.3f}  {trend}")

if buffer:
    agent.update(buffer)

agent.save("results/models/prototype_ppo.pt")
np.save("results/logs/prototype_train_rewards.npy", np.array(train_rewards))
print(f"  ▶ 학습 완료  ep1={train_rewards[0]:+.3f} → ep{N_TRAIN}={train_rewards[-1]:+.3f}")

# ── Phase 3: RL Evaluation (Deterministic) ───────────────────────
print(f"\n[Phase 3] RL 결정론적 평가  ({N_EVAL} 에피소드)")
ev_rewards = []; ev_metrics_list = []
best_r = -1e9; best_raw = None
for ep in range(1, N_EVAL + 1):
    r, info, raw = run_episode_eval(env, agent)
    ev_rewards.append(r)
    if info:
        ev_metrics_list.append({k: info[k] for k in METRIC_KEYS})
    if r > best_r:
        best_r = r; best_raw = raw
    print(f"  ep {ep:2d} | reward: {r:+.3f}")

ev_mean_r = float(np.mean(ev_rewards))
ev_metrics = avg_metrics(ev_metrics_list)
print(f"  ▶ 평균 보상: {ev_mean_r:+.3f}")

# ── 결과 출력 ─────────────────────────────────────────────────────
improvement = ((ev_mean_r - bl_mean_r) / abs(bl_mean_r) * 100) if bl_mean_r != 0 else 0.0

print("\n")
print("=" * 68)
print("  신체 부위별 충격량 비교  (NHTSA FMVSS 208 기준)")
print("=" * 68)
print(f"  {'지표':<26}  {'Rule-Based':>11}  {'RL 최적화':>11}  {'기준선':>9}  판정")
print("  " + "─" * 64)
for key in METRIC_KEYS:
    bl_v  = bl_metrics[key]
    ev_v  = ev_metrics[key]
    safe  = SAFE_LIMITS[key]
    label = METRIC_LABEL[key]
    unit  = METRIC_UNIT[key]
    judge = "RL ✓" if ev_v < bl_v else ("동등" if abs(ev_v - bl_v) < 1e-3 else "   -")
    print(f"  {label:<26}  {bl_v:>9.2f}{unit:2s}  {ev_v:>9.2f}{unit:2s}  {safe:>7.0f}{unit:2s}  {judge}")

print()
print("=" * 68)
print("  최적화된 에어백 파라미터  (평가 최고 에피소드 기준)")
print("=" * 68)
print(f"  {'에어백':<18}  {'전개':^6}  {'타이밍':>10}  {'압력':>12}")
print("  " + "─" * 52)
if best_raw is not None:
    for i, name in enumerate(AIRBAG_NAMES):
        dep = best_raw[i, 0] > 0.5
        t   = best_raw[i, 1] if dep else 0.0
        p   = best_raw[i, 2] if dep else 0.0
        d_s = "  ✓  " if dep else "  -  "
        t_s = f"{t:5.1f} ms" if dep else "     -  "
        p_s = f"{p:6.1f} kPa" if dep else "        -"
        print(f"  {name:<18}  {d_s}  {t_s:>10}  {p_s:>12}")

print()
print("=" * 68)
print(f"  보상 추이 : ep1 {train_rewards[0]:+.3f}  →  ep{N_TRAIN} {train_rewards[-1]:+.3f}")
print(f"  Rule-Based 평균 보상 : {bl_mean_r:+.3f}")
print(f"  RL 최적화  평균 보상 : {ev_mean_r:+.3f}")
print(f"  보상 개선율          : {improvement:+.1f}%")
print("=" * 68)

# ── 파일 저장 ─────────────────────────────────────────────────────
summary = {
    "config": {
        "n_baseline": N_BASELINE,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "lr": LR,
        "batch_size": BATCH_SIZE,
    },
    "baseline": {
        "mean_reward": bl_mean_r,
        "metrics": bl_metrics,
    },
    "rl": {
        "mean_reward": ev_mean_r,
        "metrics": ev_metrics,
        "reward_improvement_pct": improvement,
    },
    "safe_limits": SAFE_LIMITS,
    "optimized_airbag_params": [
        {
            "id": i,
            "name": AIRBAG_NAMES[i],
            "deploy": bool(best_raw[i, 0] > 0.5) if best_raw is not None else False,
            "timing_ms": float(best_raw[i, 1]) if best_raw is not None and best_raw[i, 0] > 0.5 else 0.0,
            "pressure_kpa": float(best_raw[i, 2]) if best_raw is not None and best_raw[i, 0] > 0.5 else 0.0,
        }
        for i in range(5)
    ],
    "train_reward_curve": [float(r) for r in train_rewards],
}

with open("results/prototype_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

csv_path = "results/prototype_log.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "phase", "reward"])
    for i, r in enumerate(bl_rewards):
        writer.writerow([i + 1, "baseline", f"{r:.4f}"])
    for i, r in enumerate(train_rewards):
        writer.writerow([N_BASELINE + i + 1, "train", f"{r:.4f}"])
    for i, r in enumerate(ev_rewards):
        writer.writerow([N_BASELINE + N_TRAIN + i + 1, "eval", f"{r:.4f}"])

print(f"\n  결과 저장 완료:")
print(f"    results/prototype_summary.json")
print(f"    results/prototype_log.csv")
print(f"    results/models/prototype_ppo.pt")
print()

env.close()
sim_app.close()
