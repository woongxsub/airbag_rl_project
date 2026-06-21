"""
Rule-Based / Pure PPO / Curriculum PPO — 500 에피소드 공정 비교 평가.

동일한 ScenarioSampler(seed=2025) 시드로 세 정책을 순차 실행:
  - 각도 0~360°, 속도 20~120km/h, Pure 분포 (stage=0)
  - PPO는 get_deterministic_action() (확률적 샘플 아님)

결과 저장:
  results/comparison/{rule_based|pure_ppo|curriculum_ppo}_{hic15|chest_g}.npy
  results/comparison/baseline_comparison_boxplot.png
"""

import os
import sys
import time
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from env.airbag_env import AirbagEnv
from env.scenario import ScenarioSampler
from rl.ppo import PPOAgent
from baseline.rule_based import rule_based_policy
import yaml

os.makedirs("results/comparison", exist_ok=True)

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

N_EVAL  = 500
SEED    = 2025     # 학습 시드와 다른 테스트 전용 시드
TIMING_MAX_MS = 30.0

# ── PPO 에이전트 로드 헬퍼 ──────────────────────────────────────────────
def load_agent(path: str) -> PPOAgent:
    agent = PPOAgent(state_dim=cfg["env"]["state_dim"])
    agent.load(path)
    agent.actor.eval()
    agent.critic.eval()
    return agent


# ══════════════════════════════════════════════════════════════════════════
# 공통 평가 루프
# ══════════════════════════════════════════════════════════════════════════
def run_eval(policy_name: str, action_fn, env: AirbagEnv) -> dict:
    """
    action_fn(obs, scenario) → action ndarray (15,)
    """
    hic15_list   = []
    chest_g_list = []

    # 동일한 시드로 샘플러 재설정
    env.sampler = ScenarioSampler(seed=SEED)
    env.sampler.stage = 0  # Pure 분포

    t_start = time.time()
    for ep in range(1, N_EVAL + 1):
        obs, _ = env.reset()
        done      = False
        ep_info   = {}
        while not done:
            action = action_fn(obs, env.scenario)
            obs, _, done, _, info = env.step(action)
            if info:
                ep_info = info

        hic15_list.append(ep_info.get("hic15",   float("nan")))
        chest_g_list.append(ep_info.get("chest_g", float("nan")))

        if ep % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  [{policy_name}] ep {ep}/{N_EVAL}  "
                  f"median_HIC15={np.nanmedian(hic15_list):.0f}  "
                  f"t={elapsed/60:.1f}min", flush=True)

    hic15  = np.array(hic15_list,   dtype=np.float64)
    chest_g = np.array(chest_g_list, dtype=np.float64)
    np.save(f"results/comparison/{policy_name}_hic15.npy",   hic15)
    np.save(f"results/comparison/{policy_name}_chest_g.npy", chest_g)

    elapsed = time.time() - t_start
    print(f"  [{policy_name}] 완료  median_HIC15={np.nanmedian(hic15):.0f}  "
          f"mean_HIC15={np.nanmean(hic15):.0f}  t={elapsed/60:.1f}min\n", flush=True)
    return {"hic15": hic15, "chest_g": chest_g}


# ── 정책 래퍼 함수 ─────────────────────────────────────────────────────
def make_rule_based_fn():
    def fn(obs, scenario):
        angle             = scenario["angle"]
        is_rollover       = bool(scenario.get("is_rollover",       False))
        passenger_present = bool(scenario.get("passenger_present", True))
        mat = rule_based_policy(angle, is_rollover, passenger_present)
        return np.concatenate([
            mat[:, 0],
            mat[:, 1] / TIMING_MAX_MS,
            mat[:, 2] / 250.0,
        ])
    return fn


def make_ppo_fn(agent: PPOAgent):
    def fn(obs, scenario):
        return agent.get_deterministic_action(obs)
    return fn


# ══════════════════════════════════════════════════════════════════════════
# 메인 평가
# ══════════════════════════════════════════════════════════════════════════
env = AirbagEnv(headless=True)

print("=" * 60)
print(f"공정 비교 평가  N={N_EVAL}  seed={SEED}")
print("=" * 60 + "\n")

results = {}

print("[1/3] Rule-Based 평가...")
results["rule_based"] = run_eval("rule_based", make_rule_based_fn(), env)

print("[2/3] Pure PPO 평가...")
pure_agent = load_agent("results/models/pure_ppo_final.pt")
results["pure_ppo"] = run_eval("pure_ppo", make_ppo_fn(pure_agent), env)
del pure_agent

print("[3/3] Curriculum PPO 평가...")
cur_agent = load_agent("results/models/curriculum_ppo_final.pt")
results["curriculum_ppo"] = run_eval("curriculum_ppo", make_ppo_fn(cur_agent), env)
del cur_agent

env.close()

# ══════════════════════════════════════════════════════════════════════════
# 비교표 출력
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("    500 에피소드 공정 비교 (seed=2025, Pure 분포 기준)")
print("=" * 72)

RB  = results["rule_based"]
PP  = results["pure_ppo"]
CP  = results["curriculum_ppo"]

def p(arr): return np.nanmedian(arr)
def m(arr): return np.nanmean(arr)
def pct_better(base, comp, lower_is_better=True):
    if lower_is_better:
        return (base - comp) / base * 100
    return (comp - base) / base * 100

print(f"\n{'지표':<36}  {'Rule-Based':>12}  {'Pure PPO':>12}  {'Curriculum PPO':>14}")
print("-" * 78)
for label, fn in [("HIC15  median", p), ("HIC15  mean", m),
                  ("chest_g median (g)", p), ("chest_g mean (g)", m)]:
    rv, pv, cv = fn(RB["hic15"] if "HIC15" in label else RB["chest_g"]), \
                 fn(PP["hic15"] if "HIC15" in label else PP["chest_g"]), \
                 fn(CP["hic15"] if "HIC15" in label else CP["chest_g"])
    print(f"  {label:<34}  {rv:>12.0f}  {pv:>12.0f}  {cv:>14.0f}")

# 개선율: Rule-Based 대비 PPO
rb_h_med = p(RB["hic15"])
rb_c_med = p(RB["chest_g"])
print("-" * 78)
for label, pv, cv in [
    ("HIC15  median 개선율 vs Rule-Based",
     pct_better(rb_h_med, p(PP["hic15"])),
     pct_better(rb_h_med, p(CP["hic15"]))),
    ("chest_g median 개선율 vs Rule-Based",
     pct_better(rb_c_med, p(PP["chest_g"])),
     pct_better(rb_c_med, p(CP["chest_g"]))),
]:
    sign = lambda v: f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"
    print(f"  {label:<34}  {'—':>12}  {sign(pv):>12}  {sign(cv):>14}")

print("\n* 양수(+) = Rule-Based보다 낮은(좋은) HIC15/chest_g")

# ══════════════════════════════════════════════════════════════════════════
# Boxplot 생성
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

LABELS = ["Rule-Based", "Pure PPO", "Curriculum\nPPO"]
C_RB   = "#4CAF50"   # Green

def clip_finite(arr, pct_hi=99.9):
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        return v
    hi = np.percentile(v, pct_hi)
    return v[v <= hi]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, key, ylabel, title in [
    (axes[0], "hic15",   "HIC15 (log scale)",      "HIC15 Distribution"),
    (axes[1], "chest_g", "Peak Chest Acceleration (g)", "Chest-g Distribution"),
]:
    data = [
        clip_finite(RB[key]),
        clip_finite(PP[key]),
        clip_finite(CP[key]),
    ]
    bp = ax.boxplot(data, labels=LABELS, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.4),
                    medianprops=dict(color="black", linewidth=2))
    colors = [C_RB, C_PURE, C_CUR]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    if key == "hic15":
        ax.set_yscale("log")
        ax.axhline(700, color="#D32F2F", lw=1.5, ls=":", label="Safety limit (700)")
        ax.legend(fontsize=9)

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.35)

# 색 범례
handles = [
    mpatches.Patch(color=C_RB,   alpha=0.7, label="Rule-Based"),
    mpatches.Patch(color=C_PURE, alpha=0.7, label="Pure PPO"),
    mpatches.Patch(color=C_CUR,  alpha=0.7, label="Curriculum PPO"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 1.01), framealpha=0.9)

fig.suptitle(f"Airbag RL — Policy Comparison ({N_EVAL} eval episodes, seed={SEED})",
             fontsize=12, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig("results/comparison/baseline_comparison_boxplot.png", bbox_inches="tight")
plt.close()
print("\n✓ baseline_comparison_boxplot.png 저장 완료")

sim_app.close()
print("\n평가 완료.")
