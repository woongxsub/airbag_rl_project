"""
4k 학습 완료 후 자동 실행되는 차트 생성기.
입력 CSV: results/logs/pure_4k_ppo_checkpoints.csv
          results/logs/curriculum_4k_ppo_checkpoints.csv

출력:
  results/comparison/hic_trend_4k_full.png          (전체 데이터, 폭발 포함)
  results/comparison/hic_trend_4k_excl_explosion.png (폭발 제외 median)

스타일: make_presentation_chart.py 동일
  - y축 눈금 숫자 제거 (log scale 격자선 유지)
  - x축 끝 episode 숫자 (500, 1000, ..., 4000)
  - 범례 그래프 아래 배치
  - Stage 라벨 플롯 박스 안 상단
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker

os.makedirs("results/comparison", exist_ok=True)

# ── CSV 로더 ─────────────────────────────────────────────────────────────────
def load_csv_col(path, col):
    eps, vals = [], []
    if not os.path.exists(path):
        print(f"  [WARN] 파일 없음: {path}")
        return np.array([]), np.array([])
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                eps.append(int(row["episode"]))
                vals.append(float(row[col]))
            except (KeyError, ValueError):
                pass
    return np.array(eps, dtype=float), np.array(vals, dtype=float)

PURE_CSV = "results/logs/pure_4k_ppo_checkpoints.csv"
CUR_CSV  = "results/logs/curriculum_4k_ppo_checkpoints.csv"

C_PURE = "#2196F3"
C_CUR  = "#FF6F00"
C_S1   = "#E8F5E9"
C_S2   = "#FFF9C4"
C_S3   = "#FCE4EC"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.dpi": 200})

# ── 공통 플롯 함수 ────────────────────────────────────────────────────────────
def make_chart(pure_vals, cur_vals, eps_pure, eps_cur, col_label, outfile, title_suffix):
    N_EP   = 4000
    CHUNK  = 100   # CSV가 100ep 단위

    # Stage 경계 (episode 기준)
    s2_ep = N_EP * 0.333
    s3_ep = N_EP * 0.666

    stage_info = [
        ("Stage 1", "frontal ±45°\n30-60 km/h", C_S1, 0,     s2_ep),
        ("Stage 2", "all angles\n30-90 km/h",   C_S2, s2_ep, s3_ep),
        ("Stage 3", "all angles\n20-120 km/h",  C_S3, s3_ep, N_EP),
    ]

    fig, ax = plt.subplots(figsize=(14, 5.8))

    # Stage 배경
    for stlbl, sublbl, color, xa, xb in stage_info:
        ax.axvspan(xa, xb, alpha=0.20, color=color)
        ax.text((xa + xb) / 2, 0.97, stlbl,
                ha="center", va="top", fontsize=10, color="#444", style="italic",
                transform=ax.get_xaxis_transform())
        ax.text((xa + xb) / 2, 0.82, sublbl,
                ha="center", va="top", fontsize=7.5, color="#999",
                transform=ax.get_xaxis_transform())
    ax.axvline(s2_ep, color="#9E9E9E", lw=0.9, ls="--")
    ax.axvline(s3_ep, color="#9E9E9E", lw=0.9, ls="--")

    # Stage-wide median 가로 점선
    for xa, xb, arr in [(0, s2_ep, pure_vals), (s2_ep, s3_ep, pure_vals), (s3_ep, N_EP, pure_vals)]:
        mask = (eps_pure >= xa) & (eps_pure <= xb)
        v = pure_vals[mask]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            ax.hlines(np.median(v), xa, xb, colors=C_PURE, lw=1.8, alpha=0.75,
                      linestyle=(0, (4, 3)))

    for xa, xb in [(0, s2_ep), (s2_ep, s3_ep), (s3_ep, N_EP)]:
        mask = (eps_cur >= xa) & (eps_cur <= xb)
        v = cur_vals[mask]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            ax.hlines(np.median(v), xa, xb, colors=C_CUR, lw=1.8, alpha=0.75,
                      linestyle=(0, (4, 3)))

    # 실선
    if len(eps_pure):
        ax.plot(eps_pure, pure_vals, color=C_PURE, lw=2.2, marker="o", markersize=3.5,
                label="Pure PPO  (100-ep median)")
    if len(eps_cur):
        ax.plot(eps_cur, cur_vals, color=C_CUR, lw=2.2, marker="s", markersize=3.5,
                label="Curriculum PPO  (100-ep median)")

    ax.set_yscale("log")
    ax.set_xlim(0, N_EP + 50)

    # y축: 숫자 제거, 격자선 유지
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_ylabel("Median HIC15  (relative comparison, log scale)", fontsize=11)

    # x축: 500ep 간격
    xticks = list(range(500, N_EP + 1, 500))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=9)
    ax.set_xlabel("Episode", fontsize=12)

    ax.set_title(f"HIC15 Trend — Pure PPO vs Curriculum PPO  (4000 episodes, {title_suffix})\n"
                 "(solid: 100-ep median  |  dashed: stage-wide median)",
                 fontsize=12, fontweight="bold")

    # 범례 아래 배치
    line_handles, _ = ax.get_legend_handles_labels()
    dash_p = mpatches.Patch(color=C_PURE, alpha=0.8, label="Pure PPO  (stage median)")
    dash_c = mpatches.Patch(color=C_CUR,  alpha=0.8, label="Curriculum PPO  (stage median)")
    stage_patches = [
        mpatches.Patch(color=C_S1, alpha=0.6, label="Stage 1: frontal ±45°, 30-60 km/h"),
        mpatches.Patch(color=C_S2, alpha=0.6, label="Stage 2: all angles, 30-90 km/h"),
        mpatches.Patch(color=C_S3, alpha=0.6, label="Stage 3: all angles, 20-120 km/h"),
    ]
    ax.legend(handles=line_handles + [dash_p, dash_c] + stage_patches,
              fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3,
              framealpha=0.93, edgecolor="#ccc")

    ax.grid(axis="y", which="major", alpha=0.20)
    ax.grid(axis="y", which="minor", alpha=0.08)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ {outfile}")


# ── 실행 ─────────────────────────────────────────────────────────────────────
print("=== make_charts_6k.py (4k run) ===")

try:
    # 버전 A: 전체 (폭발 포함)
    ep_p, v_p = load_csv_col(PURE_CSV, "hic15_median")
    ep_c, v_c = load_csv_col(CUR_CSV,  "hic15_median")
    if len(ep_p) or len(ep_c):
        make_chart(v_p, v_c, ep_p, ep_c,
                   col_label="hic15_median",
                   outfile="results/comparison/hic_trend_4k_full.png",
                   title_suffix="all episodes incl. explosion")
    else:
        print("  [SKIP] hic15_median 데이터 없음")
except Exception as e:
    print(f"  [ERROR] 전체 차트 실패: {e}")

try:
    # 버전 B: 폭발 제외
    ep_p2, v_p2 = load_csv_col(PURE_CSV, "hic15_median_excl_explosion")
    ep_c2, v_c2 = load_csv_col(CUR_CSV,  "hic15_median_excl_explosion")
    if len(ep_p2) or len(ep_c2):
        make_chart(v_p2, v_c2, ep_p2, ep_c2,
                   col_label="hic15_median_excl_explosion",
                   outfile="results/comparison/hic_trend_4k_excl_explosion.png",
                   title_suffix="explosion excluded (HIC15 < 1M)")
    else:
        print("  [SKIP] hic15_median_excl_explosion 데이터 없음")
except Exception as e:
    print(f"  [ERROR] 폭발 제외 차트 실패: {e}")

print("완료.")
