"""
발표용 HIC15 Trend 최종본.
기반: make_charts3.py (1000ep, 50-ep chunk, Stage 1/2/3)

수정:
  - y축 눈금 숫자 제거 (log scale 격자선만 유지)
  - x축 "끝 episode 숫자"로 단순화 (50, 100, ..., 1000)
  - 범례 그래프 아래 배치 (데이터 선과 비겹침)

저장: results/comparison/hic_trend_final_for_presentation.png  (200dpi)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/comparison", exist_ok=True)

pure_h = np.load("results/logs/pure_ppo_hic15.npy")
cur_h  = np.load("results/logs/curriculum_ppo_hic15.npy")

C_PURE = "#2196F3"
C_CUR  = "#FF6F00"
C_S1   = "#E8F5E9"
C_S2   = "#FFF9C4"
C_S3   = "#FCE4EC"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 200})

# ── 50ep 구간 median ──────────────────────────────────────────────────────
CHUNK    = 50
N_CHUNKS = 1000 // CHUNK   # 20

def chunk_median(arr, k):
    a, b = k * CHUNK, (k + 1) * CHUNK
    v = arr[a:b]
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) >= 3 else float("nan")

p_vals = np.array([chunk_median(pure_h, k) for k in range(N_CHUNKS)])
c_vals = np.array([chunk_median(cur_h,  k) for k in range(N_CHUNKS)])

def seg_median(arr, a, b):
    v = arr[a:b]; v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else float("nan")

stage_meds = {
    "pure": [seg_median(pure_h,  0, 333), seg_median(pure_h, 333, 666), seg_median(pure_h, 666, 1000)],
    "cur":  [seg_median(cur_h,   0, 333), seg_median(cur_h,  333, 666), seg_median(cur_h,  666, 1000)],
}

x_idx = np.arange(N_CHUNKS)
s2_x  = 333 / CHUNK    # 6.66
s3_x  = 666 / CHUNK    # 13.32
stage_x = [(-0.5, s2_x), (s2_x, s3_x), (s3_x, N_CHUNKS - 0.5)]
Y_MAX_DISP = 5e7

# ── 그래프 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.8))

# Stage 배경 음영
stage_info = [
    ("Stage 1", "frontal ±45°\n30–60 km/h", C_S1),
    ("Stage 2", "all angles\n30–90 km/h",   C_S2),
    ("Stage 3", "all angles\n20–120 km/h",  C_S3),
]
for (xa, xb), (stlbl, sublbl, color) in zip(stage_x, stage_info):
    ax.axvspan(xa, xb, alpha=0.20, color=color)
    # get_xaxis_transform: x=data좌표, y=axes비율(0=하단, 1=상단) → 플롯 박스 안에 고정
    ax.text((xa + xb) / 2, 0.97, stlbl,
            ha="center", va="top", fontsize=10, color="#444", style="italic",
            transform=ax.get_xaxis_transform())
    ax.text((xa + xb) / 2, 0.82, sublbl,
            ha="center", va="top", fontsize=7.5, color="#999",
            transform=ax.get_xaxis_transform())

# Stage 경계 수직 점선
ax.axvline(s2_x, color="#9E9E9E", lw=0.9, ls="--")
ax.axvline(s3_x, color="#9E9E9E", lw=0.9, ls="--")

# Stage 전체 median 가로 점선
for si, (xa, xb) in enumerate(stage_x):
    kw = dict(lw=1.8, alpha=0.75, linestyle=(0, (4, 3)))
    ax.hlines(stage_meds["pure"][si], xa, xb, colors=C_PURE, **kw)
    ax.hlines(stage_meds["cur"][si],  xa, xb, colors=C_CUR,  **kw)

# 50ep 구간 median 꺾은선
ax.plot(x_idx, p_vals, color=C_PURE, lw=2.2, marker="o", markersize=4,
        label="Pure PPO  (50-ep median)")
ax.plot(x_idx, c_vals, color=C_CUR,  lw=2.2, marker="s", markersize=4,
        label="Curriculum PPO  (50-ep median)")

# 클리핑 주석
for arr, color, yoff in [(p_vals, C_PURE, 1.25), (c_vals, C_CUR, 0.65)]:
    for i, v in enumerate(arr):
        if np.isfinite(v) and v > Y_MAX_DISP:
            ax.annotate(f"{v/1e6:.1f}M↑",
                        xy=(i, Y_MAX_DISP * 0.80),
                        xytext=(i + 0.15, Y_MAX_DISP * yoff),
                        fontsize=7.5, color=color, ha="left",
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

ax.set_yscale("log")
ax.set_ylim(8e3, Y_MAX_DISP)
ax.set_xlim(-0.5, N_CHUNKS - 0.5)

# ── y축: 눈금 숫자 제거, 격자선·log간격 유지 ──────────────────────────────
ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.tick_params(axis="y", which="both", length=0)   # 눈금 막대도 제거
ax.set_ylabel("Median HIC15  (relative comparison, log scale)", fontsize=11)

# ── x축: 끝 episode 숫자로 단순화 ────────────────────────────────────────
end_eps = [(k + 1) * CHUNK for k in range(N_CHUNKS)]   # 50, 100, ..., 1000
# 50ep마다 표기하면 빽빽하므로 100ep 간격으로 표기 (인덱스 홀수 제거)
tick_idx    = [k for k in range(N_CHUNKS) if (k + 1) % 2 == 0]   # 1,3,5,...,19 → ep100,200,...,1000
tick_labels = [str(end_eps[k]) for k in tick_idx]
ax.set_xticks([x_idx[k] for k in tick_idx])
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_xlabel("Episode", fontsize=12)

ax.set_title("HIC15 Absolute Trend — Pure PPO vs Curriculum PPO\n"
             "(solid: 50-ep chunk median  |  dashed: stage-wide median)",
             fontsize=12, fontweight="bold")

# ── 범례: 그래프 아래 배치 ────────────────────────────────────────────────
line_handles, _ = ax.get_legend_handles_labels()
dash_p = mpatches.Patch(color=C_PURE, alpha=0.8, label="Pure PPO  (stage median)")
dash_c = mpatches.Patch(color=C_CUR,  alpha=0.8, label="Curriculum PPO  (stage median)")
stage_patches = [
    mpatches.Patch(color=C_S1, alpha=0.6, label="Stage 1: frontal ±45°, 30–60 km/h"),
    mpatches.Patch(color=C_S2, alpha=0.6, label="Stage 2: all angles, 30–90 km/h"),
    mpatches.Patch(color=C_S3, alpha=0.6, label="Stage 3: all angles, 20–120 km/h"),
]
ax.legend(
    handles    = line_handles + [dash_p, dash_c] + stage_patches,
    fontsize   = 8.5,
    loc        = "upper center",
    bbox_to_anchor = (0.5, -0.18),
    ncol       = 3,
    framealpha = 0.93,
    edgecolor  = "#ccc",
)

ax.grid(axis="y", which="major", alpha=0.20)
ax.grid(axis="y", which="minor", alpha=0.08)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig("results/comparison/hic_trend_final_for_presentation.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("✓ results/comparison/hic_trend_final_for_presentation.png  (200dpi)")
