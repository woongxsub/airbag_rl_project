"""
HIC15 절댓값 흐름 + IQR 밴드.
저장: results/comparison/hic15_trend_with_band.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

os.makedirs("results/comparison", exist_ok=True)

pure_h = np.load("results/logs/pure_ppo_hic15.npy")
cur_h  = np.load("results/logs/curriculum_ppo_hic15.npy")

C_PURE = "#2196F3"
C_CUR  = "#FF6F00"
C_S1   = "#E8F5E9"
C_S2   = "#FFF9C4"
C_S3   = "#FCE4EC"

CHUNK    = 50
N_CHUNKS = 1000 // CHUNK   # 20
Y_MIN    = 8e3
Y_MAX    = 5e7

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 150})

# ── 구간별 통계 계산 ────────────────────────────────────────────────────
def chunk_stats(arr, k):
    a, b = k * CHUNK, (k + 1) * CHUNK
    v = arr[a:b]
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    q1  = float(np.percentile(v, 25))
    med = float(np.median(v))
    q3  = float(np.percentile(v, 75))
    std = float(np.std(v))
    return q1, med, q3, std, len(v)

p_stats = [chunk_stats(pure_h, k) for k in range(N_CHUNKS)]
c_stats = [chunk_stats(cur_h,  k) for k in range(N_CHUNKS)]

p_q1  = np.array([s[0] for s in p_stats])
p_med = np.array([s[1] for s in p_stats])
p_q3  = np.array([s[2] for s in p_stats])
p_std = np.array([s[3] for s in p_stats])

c_q1  = np.array([s[0] for s in c_stats])
c_med = np.array([s[1] for s in c_stats])
c_q3  = np.array([s[2] for s in c_stats])
c_std = np.array([s[3] for s in c_stats])

# log scale에서 fill_between 쓸 때 Q1/Q3를 y축 범위로 클리핑
p_q1_clipped = np.clip(p_q1, Y_MIN, Y_MAX)
p_q3_clipped = np.clip(p_q3, Y_MIN, Y_MAX)
c_q1_clipped = np.clip(c_q1, Y_MIN, Y_MAX)
c_q3_clipped = np.clip(c_q3, Y_MIN, Y_MAX)

# Stage 전체 median (막대그래프와 동일)
def seg_median(arr, a, b):
    v = arr[a:b]; v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else float("nan")

stage_meds = {
    "pure": [seg_median(pure_h, 0, 333),  seg_median(pure_h, 333, 666),  seg_median(pure_h, 666, 1000)],
    "cur":  [seg_median(cur_h,  0, 333),  seg_median(cur_h,  333, 666),  seg_median(cur_h,  666, 1000)],
}

x_idx = np.arange(N_CHUNKS)
s2_x  = 333 / CHUNK   # 6.66
s3_x  = 666 / CHUNK   # 13.32
stage_x = [(-0.5, s2_x), (s2_x, s3_x), (s3_x, N_CHUNKS - 0.5)]

# ── 그래프 ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Stage 배경 음영
for (xa, xb), color, stlbl, sublbl in zip(
    stage_x,
    [C_S1, C_S2, C_S3],
    ["Stage 1", "Stage 2", "Stage 3"],
    ["frontal±45°\n30–60 km/h", "all angles\n30–90 km/h", "all angles\n20–120 km/h"],
):
    ax.axvspan(xa, xb, alpha=0.20, color=color)
    ax.text((xa + xb) / 2, Y_MAX * 1.5, stlbl,
            ha="center", fontsize=10, color="#555", style="italic", va="top")
    ax.text((xa + xb) / 2, Y_MAX * 0.65, sublbl,
            ha="center", fontsize=7, color="#999", va="top")

# Stage 경계 수직 점선
ax.axvline(s2_x, color="#9E9E9E", lw=0.9, ls="--")
ax.axvline(s3_x, color="#9E9E9E", lw=0.9, ls="--")

# Stage 전체 median 가로 점선
for si, (xa, xb) in enumerate(stage_x):
    ax.hlines(stage_meds["pure"][si], xa, xb, colors=C_PURE, lw=1.6,
              alpha=0.7, linestyle=(0, (4, 3)))
    ax.hlines(stage_meds["cur"][si],  xa, xb, colors=C_CUR,  lw=1.6,
              alpha=0.7, linestyle=(0, (4, 3)))

# ── IQR 밴드 ────────────────────────────────────────────────────────
ax.fill_between(x_idx, p_q1_clipped, p_q3_clipped,
                color=C_PURE, alpha=0.18, label="_nolegend_")
ax.fill_between(x_idx, c_q1_clipped, c_q3_clipped,
                color=C_CUR,  alpha=0.18, label="_nolegend_")

# Q3 클리핑 표시 (실제값이 Y_MAX 초과하는 구간에 ↑ 표기)
for arr_q3, arr_med, color in [(p_q3, p_med, C_PURE), (c_q3, c_med, C_CUR)]:
    for i, (q3v, mv) in enumerate(zip(arr_q3, arr_med)):
        if np.isfinite(q3v) and q3v > Y_MAX:
            ax.text(i, Y_MAX * 0.80, f"Q3↑\n{q3v/1e6:.0f}M",
                    ha="center", va="bottom", fontsize=6.5,
                    color=color, alpha=0.8)

# ── Median 꺾은선 ────────────────────────────────────────────────────
ax.plot(x_idx, np.clip(p_med, Y_MIN, Y_MAX),
        color=C_PURE, lw=2.2, marker="o", markersize=4,
        zorder=5, label="Pure PPO  (median)")
ax.plot(x_idx, np.clip(c_med, Y_MIN, Y_MAX),
        color=C_CUR,  lw=2.2, marker="s", markersize=4,
        zorder=5, label="Curriculum PPO  (median)")

ax.set_yscale("log")
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlim(-0.5, N_CHUNKS - 0.5)

chunk_labels = [f"{k*CHUNK+1}–{(k+1)*CHUNK}" for k in range(N_CHUNKS)]
ax.set_xticks(x_idx)
ax.set_xticklabels(chunk_labels, rotation=45, ha="right", fontsize=7.5)
ax.set_xlabel("Episode Window (50-ep chunks)", fontsize=12)
ax.set_ylabel("HIC15 (log scale)", fontsize=12)
ax.set_title(
    "HIC15 Trend with Stability Band — Pure PPO vs Curriculum PPO\n"
    "solid line = median  |  shaded band = IQR (Q1–Q3)  |  dashed = stage-wide median",
    fontsize=11.5, fontweight="bold"
)

# ── 범례 ────────────────────────────────────────────────────────────
h_p_line  = mlines.Line2D([], [], color=C_PURE, lw=2.2, marker="o",
                           markersize=5, label="Pure PPO  (median line)")
h_c_line  = mlines.Line2D([], [], color=C_CUR,  lw=2.2, marker="s",
                           markersize=5, label="Curriculum PPO  (median line)")
h_p_band  = mpatches.Patch(color=C_PURE, alpha=0.35, label="Pure PPO  (IQR band, Q1–Q3)")
h_c_band  = mpatches.Patch(color=C_CUR,  alpha=0.35, label="Curriculum PPO  (IQR band, Q1–Q3)")
h_p_dash  = mlines.Line2D([], [], color=C_PURE, lw=1.6, ls="--",
                           alpha=0.8, label="Pure PPO  (stage median)")
h_c_dash  = mlines.Line2D([], [], color=C_CUR,  lw=1.6, ls="--",
                           alpha=0.8, label="Curriculum PPO  (stage median)")
h_s1 = mpatches.Patch(color=C_S1, alpha=0.5, label="Stage 1: easy")
h_s2 = mpatches.Patch(color=C_S2, alpha=0.5, label="Stage 2: mid")
h_s3 = mpatches.Patch(color=C_S3, alpha=0.5, label="Stage 3: hard")

ax.legend(handles=[h_p_line, h_c_line, h_p_band, h_c_band, h_p_dash, h_c_dash,
                   h_s1, h_s2, h_s3],
          fontsize=8, loc="upper left", framealpha=0.92, ncol=2)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("results/comparison/hic15_trend_with_band.png")
plt.close()
print("✓ hic15_trend_with_band.png")

# ── IQR 분석 출력 ─────────────────────────────────────────────────────
def fmt(v):
    if np.isnan(v): return "  NaN  "
    if v >= 1e6:  return f"{v/1e6:.2f}M"
    if v >= 1e3:  return f"{v/1e3:.1f}K"
    return f"{v:.0f}"

print("\n--- 50ep 구간별 IQR 분석 (Pure / Curriculum) ---")
print(f"{'구간':<12}  {'P Q1':>8} {'P Med':>8} {'P Q3':>8}"
      f"  │  {'C Q1':>8} {'C Med':>8} {'C Q3':>8}  {'C 안정성':>10}")
print("-" * 85)
for k in range(N_CHUNKS):
    stage_tag = ""
    if k == int(s2_x): stage_tag = " ←S2"
    if k == int(s3_x): stage_tag = " ←S3"
    # IQR 비율: Q3/Q1 (클수록 불안정)
    p_iqr_ratio = p_q3[k] / p_q1[k] if (np.isfinite(p_q1[k]) and p_q1[k] > 0) else float("nan")
    c_iqr_ratio = c_q3[k] / c_q1[k] if (np.isfinite(c_q1[k]) and c_q1[k] > 0) else float("nan")
    stability = "stable" if (np.isfinite(c_iqr_ratio) and c_iqr_ratio < 5) else "UNSTBL"
    print(f"  ep{k*CHUNK+1:>4}–{(k+1)*CHUNK:<4}  "
          f"{fmt(p_q1[k]):>8} {fmt(p_med[k]):>8} {fmt(p_q3[k]):>8}"
          f"  │  {fmt(c_q1[k]):>8} {fmt(c_med[k]):>8} {fmt(c_q3[k]):>8}"
          f"  {stability}{stage_tag}")

# Stage 3 후반부 집중 분석 (ep 801-1000)
print("\n--- Stage 3 후반 (ep 801-1000) IQR 집중 분석 ---")
late_chunks = range(16, 20)   # ep 801-1000
for k in late_chunks:
    p_spread = p_q3[k] / p_q1[k] if (np.isfinite(p_q1[k]) and p_q1[k] > 0) else float("nan")
    c_spread = c_q3[k] / c_q1[k] if (np.isfinite(c_q1[k]) and c_q1[k] > 0) else float("nan")
    print(f"  ep{k*CHUNK+1}–{(k+1)*CHUNK}  "
          f"Pure IQR ratio={p_spread:.1f}x   Curriculum IQR ratio={c_spread:.1f}x")

p_late_med = np.nanmedian([p_std[k] for k in range(16, 20)])
c_late_med = np.nanmedian([c_std[k] for k in range(16, 20)])
print(f"\n  ep801-1000 std median — Pure={fmt(p_late_med)}  Curriculum={fmt(c_late_med)}")
print(f"  Pure std / Curriculum std = {p_late_med / c_late_med:.1f}x (Pure가 더 불안정)")
