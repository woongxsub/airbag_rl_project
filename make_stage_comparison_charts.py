"""
Pure PPO vs Hybrid PPO 학습 곡선 - Stage별 비교.

입력 CSV (100ep 단위 체크포인트):
  results/logs/pure_ppo_checkpoints.csv   (1000ep)
  results/logs/hybrid_ppo_checkpoints.csv (900ep)

지표: hic15_median_excl_explosion

출력:
  results/comparison/learning_curve_by_stage.png
  results/comparison/stage_comparison_bar.png
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "results/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

METRIC = "hic15_median_excl_explosion"

# ── 색상 ─────────────────────────────────────────────────────────────────
C_PURE = "#1565C0"    # 진한 파랑 - Pure PPO
C_HYBRID = "#E65100"  # 진한 주황 - Hybrid PPO

STAGE_BG = ["#BBDEFB", "#C8E6C9", "#FFF9C4", "#FFCDD2"]  # 연한 파랑/초록/노랑/빨강
STAGE_BOUNDS = [0, 250, 500, 750, 900]
STAGE_LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]


def load_csv(path):
    episodes, values = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            values.append(float(row[METRIC]))
    return np.array(episodes), np.array(values)


pure_ep, pure_val = load_csv("results/logs/pure_ppo_checkpoints.csv")
hyb_ep, hyb_val = load_csv("results/logs/hybrid_ppo_checkpoints.csv")


def stage_mean(episodes, values, lo, hi):
    mask = (episodes > lo) & (episodes <= hi)
    if not mask.any():
        return np.nan
    return values[mask].mean()


pure_stage_means = [
    stage_mean(pure_ep, pure_val, STAGE_BOUNDS[i], STAGE_BOUNDS[i + 1])
    for i in range(4)
]
hyb_stage_means = [
    stage_mean(hyb_ep, hyb_val, STAGE_BOUNDS[i], STAGE_BOUNDS[i + 1])
    for i in range(4)
]

# ── 1) 전체 학습 곡선 (Stage 배경 구분) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))

for i in range(4):
    ax.axvspan(STAGE_BOUNDS[i], STAGE_BOUNDS[i + 1], color=STAGE_BG[i], alpha=0.5, zorder=0)
    mid = (STAGE_BOUNDS[i] + STAGE_BOUNDS[i + 1]) / 2
    ax.text(mid, 1.015, STAGE_LABELS[i], transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, color="#555555")

for b in STAGE_BOUNDS[1:-1]:
    ax.axvline(b, color="#9e9e9e", linewidth=0.8, linestyle="--", zorder=1)

ax.plot(pure_ep, pure_val, color=C_PURE, linewidth=2, marker="o", markersize=5,
        label="Pure PPO", zorder=3)
ax.plot(hyb_ep, hyb_val, color=C_HYBRID, linewidth=2, marker="o", markersize=5,
        label="Hybrid PPO", zorder=3)

ax.set_xlabel("Episode")
ax.set_ylabel("HIC15 median (excl. explosion)")
ax.set_title("Pure PPO vs Hybrid PPO - Learning Curve by Stage", pad=28)
ax.set_xlim(0, 1000)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

stage_patches = [mpatches.Patch(color=STAGE_BG[i], alpha=0.5, label=STAGE_LABELS[i]) for i in range(4)]
line_handles, line_labels = ax.get_legend_handles_labels()
ax.legend(handles=line_handles + stage_patches, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=6, fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "learning_curve_by_stage.png"), dpi=150)
plt.close(fig)

# ── 2) Stage별 평균 HIC15 비교 막대그래프 ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(4)
width = 0.35

bars_pure = ax.bar(x - width / 2, pure_stage_means, width, color=C_PURE, label="Pure PPO")
bars_hyb = ax.bar(x + width / 2, hyb_stage_means, width, color=C_HYBRID, label="Hybrid PPO")

for bars in (bars_pure, bars_hyb):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,.0f}", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(STAGE_LABELS)
ax.set_xlabel("Stage")
ax.set_ylabel("HIC15 median (excl. explosion) - stage average")
ax.set_title("Pure PPO vs Hybrid PPO - Stage Average HIC15 Comparison")
ax.grid(axis="y", color="#e0e0e0", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "stage_comparison_bar.png"), dpi=150)
plt.close(fig)

print("Saved:")
print(" ", os.path.join(OUT_DIR, "learning_curve_by_stage.png"))
print(" ", os.path.join(OUT_DIR, "stage_comparison_bar.png"))
print("Stage means (Pure):", pure_stage_means)
print("Stage means (Hybrid):", hyb_stage_means)
