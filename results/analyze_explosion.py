"""
폭발 에피소드 제외 전/후(A/B) 비교 분석.

전제: train.py를 RunPod GPU 인스턴스에서 아래처럼 두 번 실행해 로그를 만들어 둔 상태.

    python3 train.py --mode train --episodes 150 --tag A
    python3 train.py --mode train --episodes 150 --exclude-explosions --tag B

각 실행은 results/logs/train_hic_log_{tag}.csv, train_update_log_{tag}.csv를 생성한다.
이 스크립트는 그 두 CSV 쌍을 읽어 비교 통계를 출력하고
results/comparison/explosion_before_after.png 그래프를 저장한다.

사용법:
    python3 results/analyze_explosion.py --a A --b B
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
OUT_DIR = os.path.join(HERE, "comparison")


def load_hic_log(tag: str):
    path = os.path.join(LOGS, f"train_hic_log_{tag}.csv")
    episodes, hic15, is_explosion, reward = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            hic15.append(float(row["hic15"]))
            is_explosion.append(row["is_explosion"].strip().lower() == "true")
            reward.append(float(row["reward"]))
    return {
        "episode": np.array(episodes),
        "hic15": np.array(hic15),
        "is_explosion": np.array(is_explosion, dtype=bool),
        "reward": np.array(reward),
    }


def load_update_log(tag: str):
    path = os.path.join(LOGS, f"train_update_log_{tag}.csv")
    if not os.path.exists(path):
        return None
    episodes, actor_loss, critic_loss = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            actor_loss.append(float(row["actor_loss"]))
            critic_loss.append(float(row["critic_loss"]))
    return {
        "episode": np.array(episodes),
        "actor_loss": np.array(actor_loss),
        "critic_loss": np.array(critic_loss),
    }


def rolling(arr, w=20):
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        out[i] = np.mean(arr[lo:i + 1])
    return out


def summarize(tag: str, hic: dict, upd: dict):
    n = len(hic["episode"])
    n_exp = int(hic["is_explosion"].sum())
    rate = 100.0 * n_exp / n if n else float("nan")
    r_mean = float(np.mean(hic["reward"]))
    r_std = float(np.std(hic["reward"]))
    r_mean_clean = float(np.mean(hic["reward"][~hic["is_explosion"]])) if n_exp < n else float("nan")
    r_std_clean = float(np.std(hic["reward"][~hic["is_explosion"]])) if n_exp < n else float("nan")
    hic_median = float(np.median(hic["hic15"][~hic["is_explosion"]])) if n_exp < n else float("nan")

    critic_std = float(np.std(upd["critic_loss"])) if upd is not None and len(upd["critic_loss"]) else float("nan")
    critic_last_std = (
        float(np.std(upd["critic_loss"][-max(1, len(upd["critic_loss"]) // 2):]))
        if upd is not None and len(upd["critic_loss"])
        else float("nan")
    )

    print(f"\n[{tag}] n_episodes={n}")
    print(f"  explosion rate         : {rate:.1f}%  ({n_exp}/{n})")
    print(f"  reward mean / std (all): {r_mean:.4f} / {r_std:.4f}")
    print(f"  reward mean / std (폭발 제외): {r_mean_clean:.4f} / {r_std_clean:.4f}")
    print(f"  HIC15 median (폭발 제외)     : {hic_median:.1f}")
    print(f"  critic_loss std (전체)       : {critic_std:.4f}")
    print(f"  critic_loss std (후반 절반)  : {critic_last_std:.4f}")

    return {
        "n": n, "n_explosions": n_exp, "rate": rate,
        "reward_mean": r_mean, "reward_std": r_std,
        "reward_mean_clean": r_mean_clean, "reward_std_clean": r_std_clean,
        "hic_median_clean": hic_median,
        "critic_loss_std": critic_std, "critic_loss_std_2nd_half": critic_last_std,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="A", help="실험 A(기존, 폭발 미제외) tag")
    ap.add_argument("--b", default="B", help="실험 B(폭발 제외) tag")
    args = ap.parse_args()

    hic_a, hic_b = load_hic_log(args.a), load_hic_log(args.b)
    upd_a, upd_b = load_update_log(args.a), load_update_log(args.b)

    stats_a = summarize(args.a, hic_a, upd_a)
    stats_b = summarize(args.b, hic_b, upd_b)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(hic_a["episode"], rolling(hic_a["reward"]), label=f"{args.a} (no exclusion)", color="#d62728")
    ax.plot(hic_b["episode"], rolling(hic_b["reward"]), label=f"{args.b} (explosion excluded)", color="#1f77b4")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (20-ep rolling mean)")
    ax.set_title("Training Curve: Before vs After Explosion Exclusion")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if upd_a is not None:
        ax.plot(upd_a["episode"], upd_a["critic_loss"], label=f"{args.a} critic_loss", color="#d62728", alpha=0.8)
    if upd_b is not None:
        ax.plot(upd_b["episode"], upd_b["critic_loss"], label=f"{args.b} critic_loss", color="#1f77b4", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Critic Loss")
    ax.set_title("Critic Loss Stability")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "explosion_before_after.png")
    fig.savefig(out_path)
    print(f"\n저장 완료: {out_path}")

    print("\n[결론 판단 힌트]")
    print(f"  폭발 비율: {args.a}={stats_a['rate']:.1f}% / {args.b}={stats_b['rate']:.1f}%"
          f" (B가 0%가 아니면 threshold나 exclude 로직을 다시 확인할 것)")
    print(f"  reward std: {args.a}={stats_a['reward_std']:.4f} / {args.b}={stats_b['reward_std']:.4f}"
          f" (B가 유의미하게 작으면 폭발이 불안정성의 주 원인일 가능성 높음)")


if __name__ == "__main__":
    main()
