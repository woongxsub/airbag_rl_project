"""교수님 미팅용 결과 시각화 — 2026-06-08"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogFormatterSciNotation
import json, os

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
OUT = os.path.dirname(__file__)

# ── 데이터 로드 ────────────────────────────────────────────────────────────
train_r = np.load(os.path.join(OUT, "logs/train_rewards.npy"))          # (500,)
proto_r = np.load(os.path.join(OUT, "logs/prototype_train_rewards.npy"))# (80,)
with open(os.path.join(OUT, "prototype_summary.json")) as f:
    summary = json.load(f)

safe_limits = summary["safe_limits"]
baseline_metrics = summary["baseline"]["metrics"]
rl_metrics       = summary["rl"]["metrics"]

# ── 헬퍼: 이동평균 ────────────────────────────────────────────────────────
def rolling(arr, w=20):
    out = np.full(len(arr), np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.mean(arr[i - w + 1:i + 1])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — 학습 곡선 (500 에피소드)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Airbag RL — Training Progress (500 Episodes)", fontsize=14, fontweight="bold")

# (a) 전체 학습 곡선 (log 스케일 음수 처리)
ax = axes[0]
r_abs = np.abs(train_r)
ep = np.arange(1, len(train_r) + 1)
roll = rolling(r_abs, 30)

ax.semilogy(ep, r_abs, color="#aec6e8", alpha=0.4, linewidth=0.8, label="Episode reward")
ax.semilogy(ep, roll,  color="#1f77b4", linewidth=2.0, label="Rolling mean (30 ep)")

# 베스트 구간 표시
best_idx = np.argmin(r_abs)
ax.axvline(best_idx + 1, color="green", linestyle="--", alpha=0.7)
ax.text(best_idx + 5, r_abs[best_idx] * 3, f"Best ep {best_idx+1}\n|r|={r_abs[best_idx]:.0f}",
        fontsize=8, color="green")

ax.set_xlabel("Episode")
ax.set_ylabel("|Reward| (log scale)")
ax.set_title("(a) Full 500-Episode Training Curve")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

# (b) 후반부 수렴 추이 (마지막 200 에피소드)
ax2 = axes[1]
last200 = train_r[-200:]
ep2 = np.arange(301, 501)
clip = np.clip(last200, -1e8, 0)   # 극단값 클리핑으로 추세 가시화
roll2 = rolling(clip, 20)

ax2.plot(ep2, clip,  color="#aec6e8", alpha=0.4, linewidth=0.8, label="Episode reward (clipped)")
ax2.plot(ep2, roll2, color="#1f77b4", linewidth=2.0, label="Rolling mean (20 ep)")
ax2.axhline(np.mean(last200[-50:]), color="orange", linestyle="--", linewidth=1.5,
            label=f"Last-50 mean: {np.mean(last200[-50:]):.0f}")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Reward (clipped at -1e8)")
ax2.set_title("(b) Last 200 Episodes — Convergence View")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "report_fig1_training_curve.png"), bbox_inches="tight")
plt.close()
print("Fig1 saved")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — 안전 지표 비교 (Baseline vs RL, 프로토타입 기준)
# ═══════════════════════════════════════════════════════════════════════════
metrics_labels = {
    "hic15":                 "HIC15",
    "chest_g":               "Chest-G",
    "chest_3ms":             "Chest 3ms\nClip (g)",
    "chest_compression_mm":  "Chest\nCompression (mm)",
    "femur_n":               "Femur\nForce (N)",
    "nij":                   "Nij",
}
keys = list(metrics_labels.keys())

# 안전 한계 대비 비율 (1.0 = 한계값)
def ratio(metrics):
    return [metrics[k] / safe_limits[k] for k in keys]

base_ratio = ratio(baseline_metrics)
rl_ratio   = ratio(rl_metrics)
limit_ratio = [1.0] * len(keys)

x = np.arange(len(keys))
w = 0.32

fig, ax = plt.subplots(figsize=(13, 6))
bars_base = ax.bar(x - w/2, base_ratio, w, label="Rule-based Baseline",
                   color="#e07070", alpha=0.85, edgecolor="white")
bars_rl   = ax.bar(x + w/2, rl_ratio,   w, label="PPO RL (prototype)",
                   color="#5b9bd5", alpha=0.85, edgecolor="white")
ax.axhline(1.0, color="red", linestyle="--", linewidth=1.8, label="Safety Limit (×1.0)")

# 개선율 표시
for i, (b, r) in enumerate(zip(base_ratio, rl_ratio)):
    if b > 0 and r > 0:
        pct = (b - r) / b * 100
        color = "green" if pct > 0 else "red"
        ax.text(x[i], max(b, r) * 1.5, f"{pct:+.0f}%", ha="center",
                fontsize=8, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([metrics_labels[k] for k in keys], fontsize=10)
ax.set_ylabel("Ratio to Safety Limit  (log scale,  < 1.0 = SAFE)", fontsize=11)
ax.set_title("Safety Metric Comparison: Baseline vs PPO RL\n"
             "(Prototype Run, 80 Training Episodes)", fontsize=13, fontweight="bold")
ax.set_yscale("log")
ax.legend(fontsize=10)
ax.grid(axis="y", which="both", alpha=0.3)
ax.axhspan(0.001, 1.0, color="green", alpha=0.06)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "report_fig2_safety_metrics.png"), bbox_inches="tight")
plt.close()
print("Fig2 saved")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — 최적화된 에어백 파라미터
# ═══════════════════════════════════════════════════════════════════════════
airbag_params = summary["optimized_airbag_params"]
names    = [p["name"] for p in airbag_params]
timings  = [p["timing_ms"] if p["deploy"] else 0 for p in airbag_params]
pressures= [p["pressure_kpa"] if p["deploy"] else 0 for p in airbag_params]
deployed = [p["deploy"] for p in airbag_params]

x = np.arange(len(names))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("Optimized Airbag Deployment Parameters (PPO Policy)", fontsize=13, fontweight="bold")

colors = ["#5b9bd5" if d else "#cccccc" for d in deployed]
ax1.bar(x, timings, color=colors, edgecolor="white", alpha=0.9)
ax1.set_ylabel("Deploy Timing (ms)", fontsize=11)
ax1.axhline(30.0, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="Max limit (30ms)")
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)
for i, (t, d) in enumerate(zip(timings, deployed)):
    if d:
        ax1.text(i, t + 0.3, f"{t:.1f}ms", ha="center", fontsize=9, fontweight="bold")
    else:
        ax1.text(i, 0.5, "OFF", ha="center", fontsize=9, color="gray")

ax2.bar(x, pressures, color=colors, edgecolor="white", alpha=0.9)
ax2.set_ylabel("Pressure (kPa)", fontsize=11)
ax2.axhline(600.0, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="Max limit (600kPa)")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels(names, fontsize=10)
for i, (p, d) in enumerate(zip(pressures, deployed)):
    if d:
        ax2.text(i, p + 5, f"{p:.0f}", ha="center", fontsize=9, fontweight="bold")

blue_patch  = mpatches.Patch(color="#5b9bd5", alpha=0.9, label="Deployed")
gray_patch  = mpatches.Patch(color="#cccccc", alpha=0.9, label="Not Deployed")
fig.legend(handles=[blue_patch, gray_patch], loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "report_fig3_airbag_params.png"), bbox_inches="tight")
plt.close()
print("Fig3 saved")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — 종합 요약 대시보드
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("#f8f9fa")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

fig.text(0.5, 0.97, "Airbag RL Optimization — Progress Summary for Meeting",
         ha="center", va="top", fontsize=15, fontweight="bold", color="#1a1a2e")
fig.text(0.5, 0.94, "2026-06-08  |  PPO + Isaac Sim 6.0  |  CPU PhysX + lavapipe Vulkan",
         ha="center", va="top", fontsize=10, color="#555555")

# ─ 왼쪽 위: 학습 곡선 요약 ─
ax_lc = fig.add_subplot(gs[0, :2])
r_abs_all = np.abs(train_r)
roll_all  = rolling(r_abs_all, 30)
ep_all    = np.arange(1, 501)
ax_lc.semilogy(ep_all, r_abs_all, color="#aec6e8", alpha=0.35, linewidth=0.7)
ax_lc.semilogy(ep_all, roll_all,  color="#1f77b4", linewidth=2.2, label="Rolling mean (30ep)")
ax_lc.set_xlabel("Episode", fontsize=10)
ax_lc.set_ylabel("|Reward|", fontsize=10)
ax_lc.set_title("Training Curve (500 ep)", fontsize=11, fontweight="bold")
ax_lc.grid(True, which="both", alpha=0.25)
ax_lc.legend(fontsize=9)
ax_lc.set_facecolor("#ffffff")

# ─ 오른쪽 위: 핵심 수치 ─
ax_num = fig.add_subplot(gs[0, 2])
ax_num.axis("off")
ax_num.set_facecolor("#ffffff")

# 핵심 수치 계산
best_reward  = train_r.max()
last50_mean  = np.mean(train_r[-50:])
proto_best   = proto_r.max()
reward_impv  = summary["rl"]["reward_improvement_pct"]

lines = [
    ("Training Episodes",     "500",                          "#1f77b4"),
    ("Best Reward (ep500)",   f"{best_reward:.1f}",           "#2ca02c"),
    ("Last-50 Mean Reward",   f"{last50_mean:.2e}",           "#ff7f0e"),
    ("Prototype Best Reward", f"{proto_best:.1f}",            "#9467bd"),
    ("Reward Improvement",    f"+{reward_impv:.1f}% vs base", "#2ca02c"),
    ("Physics Substeps",      "17  (≈16.7ms/step)",          "#8c564b"),
    ("NaN Explosions",        "FIXED (compliant contact)",    "#2ca02c"),
]
ax_num.set_xlim(0, 1); ax_num.set_ylim(0, 1)
for i, (label, val, color) in enumerate(lines):
    y = 0.93 - i * 0.135
    ax_num.text(0.0, y, label + ":", fontsize=8.5, color="#333333", va="top")
    ax_num.text(1.0, y, val, fontsize=8.5, color=color, va="top", ha="right", fontweight="bold")
ax_num.set_title("Key Numbers", fontsize=11, fontweight="bold")
ax_num.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
    facecolor="white", edgecolor="#dddddd", transform=ax_num.transAxes, clip_on=False))

# ─ 하단 왼쪽: 안전지표 비교 ─
ax_m = fig.add_subplot(gs[1, :2])
n_metrics = len(keys)
x_m = np.arange(n_metrics)
br = ax_m.bar(x_m - 0.2, base_ratio, 0.38, label="Baseline", color="#e07070", alpha=0.85, edgecolor="white")
rr = ax_m.bar(x_m + 0.2, rl_ratio,   0.38, label="PPO RL",   color="#5b9bd5", alpha=0.85, edgecolor="white")
ax_m.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Safety limit")
ax_m.set_xticks(x_m)
ax_m.set_xticklabels([metrics_labels[k] for k in keys], fontsize=8.5)
ax_m.set_ylabel("× Safety Limit", fontsize=10)
ax_m.set_title("Safety Metrics vs Limit  (prototype, 80ep)", fontsize=11, fontweight="bold")
ax_m.legend(fontsize=9)
ax_m.grid(axis="y", alpha=0.25)
ax_m.set_facecolor("#ffffff")
ax_m.fill_between([-0.5, n_metrics - 0.5], 0, 1.0, color="green", alpha=0.07)

# ─ 하단 오른쪽: 현황 체크리스트 ─
ax_chk = fig.add_subplot(gs[1, 2])
ax_chk.axis("off")
ax_chk.set_facecolor("#ffffff")
items = [
    ("✓", "1ms physics timestep",          "#2ca02c"),
    ("✓", "Compliant contact (NaN fix)",   "#2ca02c"),
    ("✓", "Dense reward function",         "#2ca02c"),
    ("✓", "GAE(λ=0.95) + entropy bonus",  "#2ca02c"),
    ("✓", "State dim 12→11 (real sensors)","#2ca02c"),
    ("◑", "500/10000 episodes done",       "#ff7f0e"),
    ("◑", "Reward still unstable",         "#ff7f0e"),
    ("✗", "Metrics below safety limit",    "#e07070"),
    ("✗", "Visual streaming (TCP issue)",  "#e07070"),
]
ax_chk.set_xlim(0, 1); ax_chk.set_ylim(0, 1)
for i, (sym, txt, color) in enumerate(items):
    y = 0.95 - i * 0.105
    ax_chk.text(0.0, y, sym, fontsize=11, color=color, va="top", fontweight="bold")
    ax_chk.text(0.12, y, txt, fontsize=8.2, color="#333333", va="top")
ax_chk.set_title("Status Checklist", fontsize=11, fontweight="bold")

plt.savefig(os.path.join(OUT, "report_fig4_dashboard.png"), bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Fig4 saved")


# ═══════════════════════════════════════════════════════════════════════════
# 수치 요약 출력
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("  AIRBAG RL — NUMERICAL RESULTS SUMMARY  (2026-06-08)")
print("="*62)
print(f"\n[Training]")
print(f"  Episodes completed   : {len(train_r)} / 10,000")
print(f"  Best single reward   : {train_r.max():.2f}")
print(f"  Last-50 mean reward  : {np.mean(train_r[-50:]):.4e}")
print(f"  Last-50 std          : {np.std(train_r[-50:]):.4e}")
print(f"  Prototype best       : {proto_r.max():.2f}  (80 ep)")

print(f"\n[Safety Metrics — Prototype Eval]")
print(f"  {'Metric':<26} {'Baseline':>14} {'PPO RL':>14} {'Safe Limit':>12} {'Improvement':>12}")
print(f"  {'-'*26} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
for k in keys:
    b = baseline_metrics[k]
    r = rl_metrics[k]
    lim = safe_limits[k]
    pct = (b - r) / abs(b) * 100 if b != 0 else 0
    flag = "✓" if r < lim else "✗"
    print(f"  {metrics_labels[k]:<26} {b:>14.1f} {r:>14.1f} {lim:>12.1f}  {pct:>+8.1f}%  {flag}")

print(f"\n[Physics Setup]")
print(f"  Timestep (physics)   : 1ms  (PHYSICS_SUBSTEPS=17)")
print(f"  Timestep (control)   : 16.7ms  (60 Hz)")
print(f"  Contact model        : compliant (stiffness=2e5, damping=1e5)")
print(f"  Rendering            : CPU lavapipe Vulkan (no GPU)")
print(f"  NaN fix              : compliant contact → 0 NaN explosions")

print(f"\n[Optimized Airbag Parameters (best policy)]")
for p in airbag_params:
    status = f"{p['timing_ms']:.1f}ms  {p['pressure_kpa']:.0f}kPa" if p["deploy"] else "NOT DEPLOYED"
    print(f"  {p['name']:<16} : {status}")

print("\n[Saved figures]")
for fn in ["report_fig1_training_curve.png", "report_fig2_safety_metrics.png",
           "report_fig3_airbag_params.png", "report_fig4_dashboard.png"]:
    print(f"  results/{fn}")
print("="*62 + "\n")
