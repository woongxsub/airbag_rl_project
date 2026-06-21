"""
학습 곡선 시각화 (Isaac Sim 불필요 — npy 직접 로드).
생성 파일:
  results/comparison/hic15_curve.png
  results/comparison/reward_curve.png
  results/comparison/critic_loss_curve.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/comparison", exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────────────────
pure_r  = np.load("results/logs/pure_ppo_rewards.npy")
cur_r   = np.load("results/logs/curriculum_ppo_rewards.npy")
pure_h  = np.load("results/logs/pure_ppo_hic15.npy")
cur_h   = np.load("results/logs/curriculum_ppo_hic15.npy")
pure_cl = np.load("results/logs/pure_ppo_critic_loss.npy")
cur_cl  = np.load("results/logs/curriculum_ppo_critic_loss.npy")

N       = 1000
WINDOW  = 100
eps     = np.arange(1, N + 1)

# ── 색상 / 스타일 ──────────────────────────────────────────────────────────
C_PURE = "#2196F3"       # Blue
C_CUR  = "#FF6F00"       # Deep Orange
C_S1   = "#E8F5E9"       # Stage 1 bg  (Light Green)
C_S2   = "#FFF9C4"       # Stage 2 bg  (Light Yellow)
C_S3   = "#FCE4EC"       # Stage 3 bg  (Light Pink)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   11,
    "axes.grid":   True,
    "grid.alpha":  0.35,
    "figure.dpi":  150,
})

# ── 유틸 ──────────────────────────────────────────────────────────────────
def rolling_median(arr, w=WINDOW):
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        a = max(0, i - w // 2)
        b = min(len(arr), i + w // 2 + 1)
        v = arr[a:b]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) >= 10:
            out[i] = np.median(v)
    return out


def add_stage_bg(ax):
    ax.axvspan(1,   333, alpha=0.18, color=C_S1, label="Stage 1 (easy)")
    ax.axvspan(334, 666, alpha=0.18, color=C_S2, label="Stage 2 (mid)")
    ax.axvspan(667, 1000, alpha=0.18, color=C_S3, label="Stage 3 (hard)")
    ax.axvline(334, color="#9E9E9E", lw=0.8, ls="--")
    ax.axvline(667, color="#9E9E9E", lw=0.8, ls="--")


# ══════════════════════════════════════════════════════════════════════════
# 1-1. HIC15 감소 추이 (log scale)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

add_stage_bg(ax)

ph = rolling_median(pure_h)
ch = rolling_median(cur_h)

ax.plot(eps, ph, color=C_PURE, lw=2.0, label="Pure PPO (rolling median)")
ax.plot(eps, ch, color=C_CUR,  lw=2.0, label="Curriculum PPO (rolling median)")

# HIC15 안전 기준선
ax.axhline(700, color="#D32F2F", lw=1.5, ls=":", label="HIC15 safety limit (700)")

ax.set_yscale("log")
ax.set_xlim(1, N)
ax.set_ylim(bottom=1e3)
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("HIC15 (log scale, rolling median w=100)", fontsize=12)
ax.set_title("HIC15 Reduction Curve — Pure PPO vs Curriculum PPO", fontsize=13, fontweight="bold")

# 범례 정리
stage_patches = [
    mpatches.Patch(color=C_S1, alpha=0.5, label="Curriculum Stage 1: frontal ±45°, 30–60 km/h, vc=2.0"),
    mpatches.Patch(color=C_S2, alpha=0.5, label="Curriculum Stage 2: all angles, 30–90 km/h, vc=5.0"),
    mpatches.Patch(color=C_S3, alpha=0.5, label="Curriculum Stage 3: all angles, 20–120 km/h, vc=8.0"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + stage_patches, loc="upper right", fontsize=8.5, framealpha=0.9)

plt.tight_layout()
plt.savefig("results/comparison/hic15_curve.png")
plt.close()
print("✓ hic15_curve.png")


# ══════════════════════════════════════════════════════════════════════════
# 1-2. 보상 추이 (median)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

add_stage_bg(ax)

def rolling_median_reward(arr, w=WINDOW):
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        a = max(0, i - w // 2)
        b = min(len(arr), i + w // 2 + 1)
        v = arr[a:b]
        v = v[np.isfinite(v)]
        if len(v) >= 10:
            out[i] = np.median(v)
    return out

pr = rolling_median_reward(pure_r)
cr = rolling_median_reward(cur_r)

ax.plot(eps, pr, color=C_PURE, lw=2.0, label="Pure PPO (rolling median)")
ax.plot(eps, cr, color=C_CUR,  lw=2.0, label="Curriculum PPO (rolling median)")

ax.set_xlim(1, N)
ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Episode Reward (rolling median w=100)", fontsize=12)
ax.set_title("Episode Reward Curve — Pure PPO vs Curriculum PPO", fontsize=13, fontweight="bold")

stage_patches = [
    mpatches.Patch(color=C_S1, alpha=0.5, label="Stage 1: easy"),
    mpatches.Patch(color=C_S2, alpha=0.5, label="Stage 2: mid"),
    mpatches.Patch(color=C_S3, alpha=0.5, label="Stage 3: hard"),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + stage_patches, fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("results/comparison/reward_curve.png")
plt.close()
print("✓ reward_curve.png")


# ══════════════════════════════════════════════════════════════════════════
# 1-3. critic_loss 수렴 추이
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.5))

upd = np.arange(1, len(pure_cl) + 1)

# 10-업데이트 rolling median
def rolling_med(arr, w=10):
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        a = max(0, i - w // 2)
        b = min(len(arr), i + w // 2 + 1)
        out[i] = np.median(arr[a:b])
    return out

ax.plot(upd, pure_cl, color=C_PURE,  alpha=0.25, lw=1.0)
ax.plot(upd, cur_cl,  color=C_CUR,   alpha=0.25, lw=1.0)
ax.plot(upd, rolling_med(pure_cl), color=C_PURE, lw=2.2,
        label=f"Pure PPO  (final median={np.median(pure_cl[-20:]):.2f})")
ax.plot(upd, rolling_med(cur_cl),  color=C_CUR,  lw=2.2,
        label=f"Curriculum PPO (final median={np.median(cur_cl[-20:]):.2f})")

# 커리큘럼 단계 전환 수직선 (업데이트 단위 변환: ~5ep/update)
for stage_ep, stage_lbl in [(334, "Stage 2"), (667, "Stage 3")]:
    upd_idx = stage_ep // 5
    ax.axvline(upd_idx, color="#9E9E9E", lw=1.0, ls="--")
    ax.text(upd_idx + 1, ax.get_ylim()[1] if not np.isnan(ax.get_ylim()[1]) else 12,
            stage_lbl, fontsize=8, color="#757575", va="top")

ax.set_xlim(1, len(pure_cl))
ax.set_xlabel("PPO Update #", fontsize=12)
ax.set_ylabel("Critic Loss", fontsize=12)
ax.set_title("Critic Loss Convergence — Pure PPO vs Curriculum PPO", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig("results/comparison/critic_loss_curve.png")
plt.close()
print("✓ critic_loss_curve.png")

print("\n모든 차트 생성 완료 → results/comparison/")
