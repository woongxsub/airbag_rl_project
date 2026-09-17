"""
Pure PPO(1000ep) vs Hybrid PPO(900ep) 비교 그래프.

입력:
  results/logs/pure_ppo_checkpoints.csv   (100ep 단위, ep100~1000)
  results/logs/hybrid_ppo_checkpoints.csv (100ep 단위, ep100~900)
출력:
  results/comparison/pure_vs_hybrid_final.png
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_checkpoints(path):
    episodes, values = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            values.append(float(row["hic15_median_excl_explosion"]))
    return episodes, values


pure_ep, pure_val = load_checkpoints("results/logs/pure_ppo_checkpoints.csv")
hybrid_ep, hybrid_val = load_checkpoints("results/logs/hybrid_ppo_checkpoints.csv")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(pure_ep, pure_val, marker="o", label="Pure PPO (1000ep)", color="#1f77b4")
ax.plot(hybrid_ep, hybrid_val, marker="s", label="Hybrid PPO (900ep)", color="#d62728")

ax.set_xlabel("Episode")
ax.set_ylabel("HIC15 median (excl. explosion)")
ax.set_title("Pure PPO vs Hybrid PPO — HIC15 median (excl. explosion)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = "results/comparison/pure_vs_hybrid_final.png"
fig.savefig(out_path, dpi=150)
print(f"saved: {out_path}")
