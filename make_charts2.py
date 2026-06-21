"""
보조 시각화 2종:
  results/comparison/hic15_stage_barchart.png
  results/comparison/hic15_normalized_trend.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

os.makedirs("results/comparison", exist_ok=True)

pure_h = np.load("results/logs/pure_ppo_hic15.npy")
cur_h  = np.load("results/logs/curriculum_ppo_hic15.npy")

C_PURE = "#2196F3"
C_CUR  = "#FF6F00"
C_S1   = "#E8F5E9"
C_S2   = "#FFF9C4"
C_S3   = "#FCE4EC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 150,
})

def seg_median(arr, a, b):
    v = arr[a:b]
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else float("nan")


# ══════════════════════════════════════════════════════════════════════════
# 그래프 1 — Stage별 median HIC15 grouped bar chart
# ══════════════════════════════════════════════════════════════════════════
segs   = [(0, 333), (333, 666), (666, 1000)]
labels = ["Stage 1\n(ep 1–333)", "Stage 2\n(ep 334–666)", "Stage 3\n(ep 667–1000)"]

p_vals = [seg_median(pure_h, a, b) for a, b in segs]
c_vals = [seg_median(cur_h,  a, b) for a, b in segs]

fig, ax = plt.subplots(figsize=(9, 5.5))

x      = np.arange(len(labels))
width  = 0.36

bars_p = ax.bar(x - width/2, p_vals, width, label="Pure PPO",
                color=C_PURE, alpha=0.82, edgecolor="white", linewidth=0.8)
bars_c = ax.bar(x + width/2, c_vals, width, label="Curriculum PPO",
                color=C_CUR,  alpha=0.82, edgecolor="white", linewidth=0.8)

ax.set_yscale("log")
ax.set_ylim(bottom=1e4, top=5e7)

# 막대 위 숫자 라벨
def fmt(v):
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    elif v >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:.0f}"

for bar in bars_p:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h * 1.08,
            fmt(h), ha="center", va="bottom", fontsize=9, color=C_PURE, fontweight="bold")
for bar in bars_c:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h * 1.08,
            fmt(h), ha="center", va="bottom", fontsize=9, color=C_CUR, fontweight="bold")

# Stage3 차이 강조 화살표
s3_p, s3_c = p_vals[2], c_vals[2]
ratio = s3_p / s3_c
ax.annotate(
    f"Curriculum\n{ratio:.0f}× lower",
    xy=(x[2] - width/2, s3_c * 3.5),
    xytext=(x[2] + 0.55, s3_c * 50),
    fontsize=9, color="#D32F2F", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Median HIC15 (log scale)", fontsize=12)
ax.set_title("Stage-wise Median HIC15 — Pure PPO vs Curriculum PPO",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", alpha=0.3)

# Stage 설명 주석
ax.text(x[0], 1.2e4, "frontal±45°\n30–60 km/h\nvc=2.0",
        ha="center", fontsize=7.5, color="#555", style="italic")
ax.text(x[1], 1.2e4, "all angles\n30–90 km/h\nvc=5.0",
        ha="center", fontsize=7.5, color="#555", style="italic")
ax.text(x[2], 1.2e4, "all angles\n20–120 km/h\nvc=8.0",
        ha="center", fontsize=7.5, color="#555", style="italic")

plt.tight_layout()
plt.savefig("results/comparison/hic15_stage_barchart.png")
plt.close()
print("✓ hic15_stage_barchart.png")
for i, (lbl, pv, cv) in enumerate(zip(labels, p_vals, c_vals)):
    print(f"  {lbl.replace(chr(10),' '):<22}  Pure={fmt(pv):>8}  Curriculum={fmt(cv):>8}"
          f"{'  ← Curriculum '+fmt(cv/pv*100)+'% of Pure' if i==2 else ''}")


# ══════════════════════════════════════════════════════════════════════════
# 그래프 2 — 50ep 구간별 정규화 감소율(%) 꺾은선
# ══════════════════════════════════════════════════════════════════════════
CHUNK = 50
n_chunks = 1000 // CHUNK   # 20

# 각 구간 median 계산
p_chunk = []
c_chunk = []
for k in range(n_chunks):
    a, b = k * CHUNK, (k + 1) * CHUNK
    p_chunk.append(seg_median(pure_h, a, b))
    c_chunk.append(seg_median(cur_h,  a, b))

p_chunk = np.array(p_chunk)
c_chunk = np.array(c_chunk)

# 기준점: 첫 구간(ep 1-50) median
p_base = p_chunk[0]
c_base = c_chunk[0]

p_norm = p_chunk / p_base * 100.0
c_norm = c_chunk / c_base * 100.0

# x축: 구간 중앙 에피소드
chunk_centers = np.arange(1, n_chunks + 1) * CHUNK - CHUNK // 2  # 25, 75, 125, ...
chunk_labels  = [f"{k*CHUNK+1}–{(k+1)*CHUNK}" for k in range(n_chunks)]

# Stage 경계 구간 인덱스 (ep → chunk)
stage2_x = 333 / CHUNK   # ≈6.66
stage3_x = 666 / CHUNK   # ≈13.32

fig, ax = plt.subplots(figsize=(12, 5))

# Stage 배경 음영 (x축을 구간 인덱스 0~19로)
x_idx = np.arange(n_chunks)
ax.axvspan(-0.5,     stage2_x, alpha=0.18, color=C_S1)
ax.axvspan(stage2_x, stage3_x, alpha=0.18, color=C_S2)
ax.axvspan(stage3_x, n_chunks - 0.5, alpha=0.18, color=C_S3)

ax.axvline(stage2_x, color="#9E9E9E", lw=0.9, ls="--")
ax.axvline(stage3_x, color="#9E9E9E", lw=0.9, ls="--")

# Stage 라벨
for xpos, stlbl in [(stage2_x / 2,               "Stage 1"),
                    ((stage2_x + stage3_x) / 2,   "Stage 2"),
                    ((stage3_x + n_chunks) / 2,    "Stage 3")]:
    ax.text(xpos, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 200,
            stlbl, ha="center", fontsize=9, color="#757575", style="italic")

# 기준선 y=100%
ax.axhline(100, color="#757575", lw=1.2, ls=":", label="Baseline (ep 1–50 = 100%)")

# 꺾은선
ax.plot(x_idx, p_norm, color=C_PURE, lw=2.2, marker="o", markersize=4,
        label=f"Pure PPO  (baseline={fmt(p_base)})")
ax.plot(x_idx, c_norm, color=C_CUR,  lw=2.2, marker="s", markersize=4,
        label=f"Curriculum PPO  (baseline={fmt(c_base)})")

# 값 클리핑 표시: 일부 구간이 y=500% 초과 시 화살표
Y_MAX = 500
for arr, color in [(p_norm, C_PURE), (c_norm, C_CUR)]:
    for i, v in enumerate(arr):
        if v > Y_MAX:
            ax.annotate(f"{v/100:.0f}×",
                        xy=(i, Y_MAX - 20),
                        xytext=(i, Y_MAX - 20),
                        fontsize=7, color=color, ha="center",
                        arrowprops=None)
            ax.annotate("", xy=(i, Y_MAX), xytext=(i, Y_MAX - 30),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

ax.set_xlim(-0.5, n_chunks - 0.5)
ax.set_ylim(0, Y_MAX)
ax.set_xticks(x_idx)
ax.set_xticklabels(chunk_labels, rotation=45, ha="right", fontsize=7.5)
ax.set_xlabel("Episode Window (50-ep chunks)", fontsize=12)
ax.set_ylabel("Normalized Median HIC15 (%)\n(lower = better than baseline)", fontsize=11)
ax.set_title("Normalized HIC15 Trend — Relative to Each Model's ep 1–50 Baseline",
             fontsize=12, fontweight="bold")

ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

# Stage 라벨 (좌상단 텍스트로 재배치)
for xpos, stlbl in [(stage2_x / 2, "Stage 1"),
                    ((stage2_x + stage3_x) / 2, "Stage 2"),
                    ((stage3_x + n_chunks) / 2, "Stage 3")]:
    ax.text(xpos, Y_MAX * 0.95, stlbl, ha="center",
            fontsize=9, color="#555", style="italic", va="top")

plt.tight_layout()
plt.savefig("results/comparison/hic15_normalized_trend.png")
plt.close()
print("✓ hic15_normalized_trend.png")

print("\n--- 50ep 구간별 정규화값 (%) ---")
print(f"{'구간':<12}  {'Pure(%)':>9}  {'Curriculum(%)':>14}")
for k in range(n_chunks):
    a, b = k*CHUNK, (k+1)*CHUNK
    marker = " ← Stage2" if k == int(stage2_x) else (" ← Stage3" if k == int(stage3_x) else "")
    print(f"  ep{a+1:>4}–{b:<4}  {p_norm[k]:>9.1f}  {c_norm[k]:>14.1f}{marker}")

print(f"\nStage3 (ep667-1000) Curriculum/Pure ratio: {np.mean(c_norm[13:]) / np.mean(p_norm[13:]):.2f}x")
