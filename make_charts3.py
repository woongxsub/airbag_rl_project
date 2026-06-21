"""
HIC15 절댓값 기반 꺾은선 흐름 그래프.
저장: results/comparison/hic15_trend_absolute.png
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

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 150})

# ── 50ep 구간 median 계산 ──────────────────────────────────────────────
CHUNK    = 50
N_CHUNKS = 1000 // CHUNK   # 20

def chunk_median(arr, k):
    a, b = k * CHUNK, (k + 1) * CHUNK
    v = arr[a:b]
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) >= 3 else float("nan")

p_vals = np.array([chunk_median(pure_h, k) for k in range(N_CHUNKS)])
c_vals = np.array([chunk_median(cur_h,  k) for k in range(N_CHUNKS)])

# Stage 전체 median (막대그래프와 동일)
def seg_median(arr, a, b):
    v = arr[a:b]; v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else float("nan")

stage_meds = {
    "pure":  [seg_median(pure_h, 0, 333),   seg_median(pure_h, 333, 666),   seg_median(pure_h, 666, 1000)],
    "cur":   [seg_median(cur_h,  0, 333),   seg_median(cur_h,  333, 666),   seg_median(cur_h,  666, 1000)],
}

# x 좌표: 구간 인덱스 0~19
x_idx = np.arange(N_CHUNKS)
# Stage 경계 (chunk 단위)
s2_x = 333 / CHUNK   # 6.66
s3_x = 666 / CHUNK   # 13.32

# Stage 구간 x 범위 (chunk index)
stage_x = [(-0.5, s2_x), (s2_x, s3_x), (s3_x, N_CHUNKS - 0.5)]

# ── 그래프 ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))

# Stage 배경 음영
for (xa, xb), color, stlbl in zip(stage_x, [C_S1, C_S2, C_S3],
                                   ["Stage 1", "Stage 2", "Stage 3"]):
    ax.axvspan(xa, xb, alpha=0.20, color=color)
    ax.text((xa + xb) / 2, 3.5e7, stlbl,
            ha="center", fontsize=10, color="#555", style="italic", va="top")

# Stage 경계 수직 점선
ax.axvline(s2_x, color="#9E9E9E", lw=0.9, ls="--")
ax.axvline(s3_x, color="#9E9E9E", lw=0.9, ls="--")

# Stage 전체 median 가로 점선 오버레이
for si, (xa, xb) in enumerate(stage_x):
    pm = stage_meds["pure"][si]
    cm = stage_meds["cur"][si]
    kw = dict(lw=1.8, alpha=0.75, linestyle=(0, (4, 3)))
    ax.hlines(pm, xa, xb, colors=C_PURE, **kw)
    ax.hlines(cm, xa, xb, colors=C_CUR,  **kw)

# 50ep 구간 median 꺾은선 (메인)
ax.plot(x_idx, p_vals, color=C_PURE, lw=2.2, marker="o", markersize=4,
        label="Pure PPO  (50-ep median)")
ax.plot(x_idx, c_vals, color=C_CUR,  lw=2.2, marker="s", markersize=4,
        label="Curriculum PPO  (50-ep median)")

# 클리핑 주석: 상한을 넘는 값
Y_MAX_DISP = 5e7
for arr, color, offset in [(p_vals, C_PURE, 1.3), (c_vals, C_CUR, 0.7)]:
    for i, v in enumerate(arr):
        if np.isfinite(v) and v > Y_MAX_DISP:
            ax.annotate(f"{v/1e6:.1f}M↑",
                        xy=(i, Y_MAX_DISP * 0.75),
                        xytext=(i + 0.15, Y_MAX_DISP * offset),
                        fontsize=7.5, color=color, ha="left",
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

ax.set_yscale("log")
ax.set_ylim(8e3, Y_MAX_DISP)
ax.set_xlim(-0.5, N_CHUNKS - 0.5)

# x축 라벨
chunk_labels = [f"{k*CHUNK+1}–{(k+1)*CHUNK}" for k in range(N_CHUNKS)]
ax.set_xticks(x_idx)
ax.set_xticklabels(chunk_labels, rotation=45, ha="right", fontsize=7.5)

ax.set_xlabel("Episode Window (50-ep chunks)", fontsize=12)
ax.set_ylabel("Median HIC15 (log scale)", fontsize=12)
ax.set_title("HIC15 Absolute Trend — Pure PPO vs Curriculum PPO\n"
             "(solid: 50-ep chunk median  |  dashed: stage-wide median)",
             fontsize=12, fontweight="bold")

# 범례
line_handles, _ = ax.get_legend_handles_labels()
dash_p = mpatches.Patch(color=C_PURE, alpha=0.75, label="Pure PPO  (stage median, dashed)")
dash_c = mpatches.Patch(color=C_CUR,  alpha=0.75, label="Curriculum PPO  (stage median, dashed)")
stage_patches = [
    mpatches.Patch(color=C_S1, alpha=0.5, label="Stage 1: frontal ±45°, 30–60 km/h"),
    mpatches.Patch(color=C_S2, alpha=0.5, label="Stage 2: all angles, 30–90 km/h"),
    mpatches.Patch(color=C_S3, alpha=0.5, label="Stage 3: all angles, 20–120 km/h"),
]
ax.legend(handles=line_handles + [dash_p, dash_c] + stage_patches,
          fontsize=8.5, loc="upper left", framealpha=0.90, ncol=1)

ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("results/comparison/hic15_trend_absolute.png")
plt.close()

# ── 보조 통계 출력 ─────────────────────────────────────────────────────
def fmt(v):
    if np.isnan(v): return "  NaN "
    if v >= 1e6: return f"{v/1e6:.2f}M"
    if v >= 1e3: return f"{v/1e3:.1f}K"
    return f"{v:.0f}"

print("✓ hic15_trend_absolute.png")
print("\n--- 50ep 구간별 median HIC15 절댓값 ---")
print(f"{'구간':<12}  {'Pure':>9}  {'Curriculum':>12}  {'Cur/Pure':>10}")
for k in range(N_CHUNKS):
    pv, cv = p_vals[k], c_vals[k]
    ratio  = cv / pv if (np.isfinite(pv) and np.isfinite(cv) and pv > 0) else float("nan")
    tag = ""
    if k == int(s2_x): tag = " ← Stage2 시작"
    if k == int(s3_x): tag = " ← Stage3 시작"
    print(f"  ep{k*CHUNK+1:>4}–{(k+1)*CHUNK:<4}  {fmt(pv):>9}  {fmt(cv):>12}  {ratio:>9.2f}x{tag}")

print(f"\nStage 3 (ep 667-1000) chunk ratio median: "
      f"{np.nanmedian(c_vals[13:] / p_vals[13:]):.3f}x  (Curriculum/Pure)")
print(f"Stage 3 chunks where Curriculum < Pure: "
      f"{sum(c_vals[13+i] < p_vals[13+i] for i in range(7))}/7")

print("\n[정규화 그래프 관련] hic15_normalized_trend.png 는 폐기 — "
      "각 모델의 baseline(ep1-50)이 42K vs 2.1M으로 50배 차이나서\n"
      "정규화 후 비교 시 절댓값 차이가 왜곡됨. "
      "절댓값 그래프(hic15_trend_absolute.png)를 발표 자료로 사용할 것.")
