"""
500-에피소드 평가 결과 boxplot 생성.
저장: results/comparison/baseline_comparison_boxplot.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/comparison", exist_ok=True)

RB_h  = np.load("results/comparison/rule_based_hic15.npy")
PP_h  = np.load("results/comparison/pure_ppo_hic15.npy")
CP_h  = np.load("results/comparison/curriculum_ppo_hic15.npy")
RB_c  = np.load("results/comparison/rule_based_chest_g.npy")
PP_c  = np.load("results/comparison/pure_ppo_chest_g.npy")
CP_c  = np.load("results/comparison/curriculum_ppo_chest_g.npy")

C_RB   = "#4CAF50"
C_PURE = "#2196F3"
C_CUR  = "#FF6F00"
LABELS = ["Rule-Based", "Pure PPO", "Curriculum\nPPO"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 150})

def clip_finite(arr, pct_hi=99.5):
    v = arr[np.isfinite(arr) & (arr > 0)]
    if len(v) == 0:
        return v
    hi = np.percentile(v, pct_hi)
    return v[v <= hi]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, hdata, cdata, ylabel, title, do_log in [
    (axes[0],
     [clip_finite(RB_h), clip_finite(PP_h), clip_finite(CP_h)],
     [clip_finite(RB_h), clip_finite(PP_h), clip_finite(CP_h)],
     "HIC15 (log scale)", "HIC15 Distribution\n(500 eval episodes, seed=2025)", True),
    (axes[1],
     [clip_finite(RB_c), clip_finite(PP_c), clip_finite(CP_c)],
     [clip_finite(RB_c), clip_finite(PP_c), clip_finite(CP_c)],
     "Peak Chest Acceleration (g)", "Chest-g Distribution\n(500 eval episodes, seed=2025)", False),
]:
    bp = ax.boxplot(hdata, tick_labels=LABELS, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.35),
                    medianprops=dict(color="black", linewidth=2.2))
    for patch, color in zip(bp["boxes"], [C_RB, C_PURE, C_CUR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    if do_log:
        ax.set_yscale("log")
        ax.axhline(700, color="#D32F2F", lw=1.5, ls=":", label="Safety limit (700)")
        ax.legend(fontsize=9)

    # median 수치 라벨
    for i, d in enumerate(hdata):
        med = np.median(d[np.isfinite(d)])
        ax.text(i + 1, med * (1.35 if do_log else 1.02),
                f"med={med/1e3:.1f}K" if med >= 1000 else f"med={med:.0f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=[C_RB, C_PURE, C_CUR][i])

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

# 색 범례
handles = [
    mpatches.Patch(color=C_RB,   alpha=0.7, label="Rule-Based"),
    mpatches.Patch(color=C_PURE, alpha=0.7, label="Pure PPO"),
    mpatches.Patch(color=C_CUR,  alpha=0.7, label="Curriculum PPO"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 1.01), framealpha=0.9)
plt.tight_layout()
plt.savefig("results/comparison/baseline_comparison_boxplot.png", bbox_inches="tight")
plt.close()
print("✓ baseline_comparison_boxplot.png")

# 상세 통계
print("\n=== 500 에피소드 최종 비교 (seed=2025) ===")
for name, h, c in [("Rule-Based",     RB_h, RB_c),
                   ("Pure PPO",        PP_h, PP_c),
                   ("Curriculum PPO",  CP_h, CP_c)]:
    hf = h[np.isfinite(h) & (h > 0)]
    cf = c[np.isfinite(c) & (c > 0)]
    print(f"\n[{name}]")
    print(f"  HIC15  median={np.median(hf):.0f}  mean={np.mean(hf):.0f}  "
          f"P25={np.percentile(hf,25):.0f}  P75={np.percentile(hf,75):.0f}")
    print(f"  chest_g median={np.median(cf):.0f}  mean={np.mean(cf):.0f}  "
          f"P25={np.percentile(cf,25):.0f}  P75={np.percentile(cf,75):.0f}")
    print(f"  HIC15 < 1만 비율: {(hf < 10000).mean()*100:.1f}%  "
          f"< 10만: {(hf < 100000).mean()*100:.1f}%  "
          f"> 100만: {(hf > 1e6).mean()*100:.1f}%")
