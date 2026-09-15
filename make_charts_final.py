"""
3종 비교 정규화 학습 곡선 (하이브리드 커리큘럼 실험 결과).

입력 CSV (100ep 단위 체크포인트):
  results/logs/rulebased_3k_checkpoints.csv
  results/logs/pure_3k_ppo_checkpoints.csv
  results/logs/hybrid_3k_ppo_checkpoints.csv

출력:
  results/comparison/hic_normalized_trend_final.png    (HIC15_median / 700)
  results/comparison/chestg_normalized_trend_final.png (chest_g_median / 60)
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("results/comparison", exist_ok=True)

# ── 상수 ────────────────────────────────────────────────────────────────────
HIC_SAFE   = 700.0
CHEST_SAFE = 60.0

C_RB   = "#4CAF50"   # 초록
C_PURE = "#2196F3"   # 파랑
C_CUR  = "#FF6F00"   # 주황

# 4단계 Stage 배경 색
STAGE_COLORS = ["#E8F5E9", "#FFF9C4", "#FCE4EC", "#EDE7F6"]
STAGE_LABELS = ["Stage 1\nfrontal±45°\n30–60 km/h",
                "Stage 2\nall angles\n30–90 km/h",
                "Stage 3\nall angles\n20–120 km/h",
                "Stage 4\nall angles\n20–120 km/h"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})

# ── CSV 로더 ─────────────────────────────────────────────────────────────────
def load_csv(path: str, col: str):
    """CSV → (episodes[], values[])  — 파일 없으면 빈 배열 반환."""
    episodes, values = [], []
    if not os.path.exists(path):
        print(f"  [WARN] 파일 없음: {path}")
        return np.array([]), np.array([])
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                episodes.append(int(row["episode"]))
                values.append(float(row[col]))
            except (KeyError, ValueError):
                pass
    return np.array(episodes, dtype=float), np.array(values, dtype=float)


# ── 차트 생성 ────────────────────────────────────────────────────────────────
def make_chart(metric_col: str, safe_limit: float,
               ylabel: str, outfile: str, title: str):

    ep_rb,   v_rb   = load_csv("results/logs/rulebased_3k_checkpoints.csv",  metric_col)
    ep_pure, v_pure = load_csv("results/logs/pure_3k_ppo_checkpoints.csv",   metric_col)
    ep_cur,  v_cur  = load_csv("results/logs/hybrid_3k_ppo_checkpoints.csv", metric_col)

    if len(ep_rb) == 0 and len(ep_pure) == 0 and len(ep_cur) == 0:
        print(f"  [SKIP] 데이터 없음 → {outfile}")
        return

    norm = lambda v: v / safe_limit

    fig, ax = plt.subplots(figsize=(15, 6))

    # Stage 배경 음영 (4단계 × 25%)
    total_ep = max(
        ep_rb[-1]   if len(ep_rb)   else 0,
        ep_pure[-1] if len(ep_pure) else 0,
        ep_cur[-1]  if len(ep_cur)  else 0,
    )
    stage_bounds = [0,
                    total_ep * 0.25,
                    total_ep * 0.50,
                    total_ep * 0.75,
                    total_ep]
    for i in range(4):
        ax.axvspan(stage_bounds[i], stage_bounds[i + 1],
                   alpha=0.13, color=STAGE_COLORS[i])
        mid = (stage_bounds[i] + stage_bounds[i + 1]) / 2
        ax.text(mid, 1, STAGE_LABELS[i],
                ha="center", va="bottom", fontsize=7.5,
                color="#666", style="italic", transform=ax.get_xaxis_transform())
    for b in stage_bounds[1:-1]:
        ax.axvline(b, color="#9E9E9E", lw=0.8, ls="--")

    # 안전 한계선 y = 1.0
    ax.axhline(1.0, color="#D32F2F", lw=1.5, ls=":",
               label=f"Safety limit  (={safe_limit:.0f})", zorder=3)

    # ── 3개 정책 실선 ──────────────────────────────────────────────────────
    if len(ep_rb):
        ax.plot(ep_rb, norm(v_rb),
                color=C_RB, lw=2.0, marker="^", markersize=4, zorder=5,
                label="Rule-Based  (100-ep median)")

    if len(ep_pure):
        ax.plot(ep_pure, norm(v_pure),
                color=C_PURE, lw=2.2, marker="o", markersize=4, zorder=5,
                label="Pure PPO  (100-ep median)")

    if len(ep_cur):
        ax.plot(ep_cur, norm(v_cur),
                color=C_CUR, lw=2.2, marker="s", markersize=4, zorder=5,
                label="Curriculum PPO Hybrid  (100-ep median)")

    # log scale: 값 범위 10배 이상이면 자동 적용
    all_vals = np.concatenate([
        norm(v_rb)   if len(v_rb)   else np.array([]),
        norm(v_pure) if len(v_pure) else np.array([]),
        norm(v_cur)  if len(v_cur)  else np.array([]),
    ])
    all_valid = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if len(all_valid) and np.max(all_valid) / np.min(all_valid) > 10:
        ax.set_yscale("log")
        print(f"  → log scale 적용 (범위 {np.min(all_valid):.3f} ~ {np.max(all_valid):.2f})")

    ax.set_xlim(left=0)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2,
              framealpha=0.95, edgecolor="#ccc")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"✓ {outfile}")


# ── 실행 ─────────────────────────────────────────────────────────────────────
print("=== make_charts_final.py ===")

make_chart(
    metric_col = "hic15_median",
    safe_limit = HIC_SAFE,
    ylabel     = "HIC15_median / 700  (safety limit = 1.0)",
    outfile    = "results/comparison/hic_normalized_trend_final.png",
    title      = ("HIC15 Normalized Learning Curve  (3000 episodes)\n"
                  "Rule-Based vs Pure PPO vs Curriculum PPO Hybrid  |  seed=42"),
)

make_chart(
    metric_col = "chest_g_median",
    safe_limit = CHEST_SAFE,
    ylabel     = "chest_g_median / 60  (safety limit = 1.0)",
    outfile    = "results/comparison/chestg_normalized_trend_final.png",
    title      = ("Chest-g Normalized Learning Curve  (3000 episodes)\n"
                  "Rule-Based vs Pure PPO vs Curriculum PPO Hybrid  |  seed=42"),
)

print("완료.")
