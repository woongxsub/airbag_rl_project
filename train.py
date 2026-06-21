import os
import sys
import csv
import argparse
import time
import yaml
import numpy as np

# ── Isaac Sim 부트스트랩 ─────────────────────────────────────────────────
os.environ["OMNI_KIT_ACCEPT_EULA"] = "yes"
sys.path.insert(0, "/workspace/isaacsim_env/lib/python3.12/site-packages")

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--stream",   action="store_true", help="WebRTC 스트리밍 활성화 (port 8211)")
parser.add_argument("--gui",      action="store_true", help="noVNC용 GUI 창 표시 (DISPLAY 환경변수 필요)")
parser.add_argument("--debug",    action="store_true", help="스텝별 샘플 수 출력 (Dense reward 검증용)")
parser.add_argument("--config",   default="config/config.yaml")
parser.add_argument("--mode",     choices=["train", "curriculum", "baseline",
                                           "hybrid_curriculum", "rulebased"],
                    default="train")
parser.add_argument("--episodes", type=int, default=None,
                    help="총 에피소드 수 (없으면 config의 total_episodes 사용)")
parser.add_argument("--seed",     type=int, default=None,
                    help="ScenarioSampler seed (재현성 보장)")
parser.add_argument("--label",    type=str, default=None,
                    help="출력 파일 접두사 (기본: pure/curriculum/rulebased/hybrid_3k)")
args = parser.parse_args()

if args.stream or args.gui:
    os.environ.setdefault(
        "VK_ICD_FILENAMES",
        "/usr/share/vulkan/icd.d/lvp_icd.json",
    )

from isaacsim import SimulationApp

sim_config = {"headless": True}
if args.stream:
    sim_config["headless"]   = False
    sim_config["width"]      = 1280
    sim_config["height"]     = 720
    sim_config["livestream"] = 1
elif args.gui:
    sim_config["headless"] = False
    sim_config["width"]    = 1280
    sim_config["height"]   = 720

sim_app = SimulationApp(sim_config)

import carb
carb.settings.get_settings().set("/physics/cudaDevice", 0)
if args.stream:
    carb.settings.get_settings().set("/app/livestream/websocket/server_port", 8211)
    print("[train] WebRTC streaming 활성화 — port 8211")

from env.airbag_env import AirbagEnv
from env.scenario import ScenarioSampler
from rl.ppo import PPOAgent
from baseline.rule_based import rule_based_policy

with open(args.config) as f:
    cfg = yaml.safe_load(f)

os.makedirs("results/models", exist_ok=True)
os.makedirs("results/logs",   exist_ok=True)


# ── 커리큘럼 단계 결정 ──────────────────────────────────────────────────────

def get_stage(ep: int, total: int):
    """기존 3단계 커리큘럼 (--mode curriculum 호환 유지)."""
    if ep <= total // 3:
        return 1, 2.0
    elif ep <= 2 * total // 3:
        return 2, 5.0
    else:
        return 3, 8.0


def get_hybrid_stage(ep: int, total: int) -> dict:
    """
    4단계 하이브리드 커리큘럼 스케줄 (--mode hybrid_curriculum).

    핵심 원칙: 가이드레일 항(correct/wrong/over/late)은 후기 단계에서
    강화하지 않고 반드시 0으로 수렴. 후기 단계는 violation_coeff와
    peak_weight만으로 미세 조정.

    Stage | ep 범위       | 분포                     | vc   | correct | wrong | over | late | peak
    ------+---------------+--------------------------+------+---------+-------+------+------+-----
      1   | 1 ~ 25%      | 정면±45°, 30-60 km/h     | 2.0  | 0.30    | 0.20  | 0.10 | 0.20 | 0.0
      2   | 25% ~ 50%    | 전 각도,   30-90 km/h    | 5.0  | 0.15    | 0.10  | 0.05 | 0.10 | 1.0
      3   | 50% ~ 75%    | 전 각도,  20-120 km/h    | 8.0  | 0.00    | 0.00  | 0.00 | 0.00 | 2.0
      4   | 75% ~ 100%   | 전 각도,  20-120 km/h    | 10.0 | 0.00    | 0.00  | 0.00 | 0.00 | 3.0
    """
    pct = ep / total
    if pct <= 0.25:
        return dict(stage=1, violation_coeff=2.0,
                    correct_weight=0.30, wrong_weight=0.20,
                    over_weight=0.10,   late_weight=0.20, peak_weight=0.0)
    elif pct <= 0.50:
        return dict(stage=2, violation_coeff=5.0,
                    correct_weight=0.15, wrong_weight=0.10,
                    over_weight=0.05,   late_weight=0.10, peak_weight=1.0)
    elif pct <= 0.75:
        return dict(stage=3, violation_coeff=8.0,
                    correct_weight=0.0,  wrong_weight=0.0,
                    over_weight=0.0,    late_weight=0.0,  peak_weight=2.0)
    else:
        return dict(stage=4, violation_coeff=10.0,
                    correct_weight=0.0,  wrong_weight=0.0,
                    over_weight=0.0,    late_weight=0.0,  peak_weight=3.0)


# ── 체크포인트 CSV ────────────────────────────────────────────────────────────

def _write_checkpoint_csv(path: str, episode: int, hic15_buf: list, chest_g_buf: list):
    """100ep 윈도우 통계를 CSV에 한 줄 추가."""
    hic_arr   = np.array(hic15_buf,   dtype=float)
    chest_arr = np.array(chest_g_buf, dtype=float)
    hic_f     = hic_arr[np.isfinite(hic_arr)   & (hic_arr   > 0)]
    chest_f   = chest_arr[np.isfinite(chest_arr) & (chest_arr > 0)]
    hic_safe  = hic_f[hic_f < 1_000_000]   # 물리 폭발 제외 (HIC15 < 1M)

    row = {
        "episode":                     episode,
        "hic15_median":                float(np.median(hic_f))     if len(hic_f)    else float("nan"),
        "chest_g_median":              float(np.median(chest_f))   if len(chest_f)  else float("nan"),
        "hic15_mean":                  float(np.mean(hic_f))       if len(hic_f)    else float("nan"),
        "chest_g_mean":                float(np.mean(chest_f))     if len(chest_f)  else float("nan"),
        "hic15_median_excl_explosion": float(np.median(hic_safe))  if len(hic_safe) else float("nan"),
    }
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Rule-Based 측정 ──────────────────────────────────────────────────────────

def run_rulebased():
    """
    Rule-Based 정책 N ep 측정 (그래디언트 없음).
    Pure PPO / Curriculum PPO와 동일 seed=args.seed 사용 → 공정 비교.
    100ep마다 체크포인트 CSV 기록.
    """
    N     = args.episodes or 3000
    seed  = args.seed
    label = args.label or "rulebased"
    csv_path = f"results/logs/{label}_checkpoints.csv"

    print(f"\n{'='*65}")
    print(f"[RULEBASED] 측정 시작  총={N}ep  seed={seed}  label={label}")
    print(f"{'='*65}\n", flush=True)

    env = AirbagEnv(headless=True, debug=args.debug)
    if seed is not None:
        env.sampler = ScenarioSampler(seed=seed)
    env.sampler.stage = 0   # Pure distribution (full range)

    all_hic15    = []
    all_chest_g  = []
    ckpt_hic15   = []
    ckpt_chest_g = []
    wall_start   = time.time()

    for ep in range(1, N + 1):
        obs, _ = env.reset()
        angle             = env.scenario["angle"]
        is_rollover       = bool(env.scenario.get("is_rollover",       False))
        passenger_present = bool(env.scenario.get("passenger_present", True))
        action_matrix = rule_based_policy(angle, is_rollover, passenger_present)
        action = np.concatenate([
            action_matrix[:, 0],
            action_matrix[:, 1] / 30.0,
            action_matrix[:, 2] / 250.0,
        ])

        done = False
        ep_info = {}
        while not done:
            _, _, done, _, info = env.step(action)
            if info:
                ep_info = info

        h = ep_info.get("hic15",   float("nan"))
        c = ep_info.get("chest_g", float("nan"))
        all_hic15.append(h)
        all_chest_g.append(c)
        ckpt_hic15.append(h)
        ckpt_chest_g.append(c)

        if ep % 100 == 0:
            _write_checkpoint_csv(csv_path, ep, ckpt_hic15, ckpt_chest_g)
            ckpt_hic15.clear()
            ckpt_chest_g.clear()
            h_arr = np.array(all_hic15)
            h_f   = h_arr[np.isfinite(h_arr) & (h_arr > 0)]
            elapsed = time.time() - wall_start
            print(
                f"[RuleBased] ep {ep:>4}/{N} | "
                f"HIC15_med={np.median(h_f) if len(h_f) else float('nan'):.0f} | "
                f"t={elapsed/60:.1f}min",
                flush=True,
            )

    env.close()
    np.save(f"results/logs/{label}_hic15.npy",   all_hic15)
    np.save(f"results/logs/{label}_chest_g.npy", all_chest_g)

    elapsed = time.time() - wall_start
    h_arr = np.array(all_hic15)
    h_f   = h_arr[np.isfinite(h_arr) & (h_arr > 0)]
    print(f"\n[RULEBASED] 완료  {N}ep  label={label}  소요={elapsed/60:.1f}분")
    print(f"  HIC15 median={np.median(h_f) if len(h_f) else float('nan'):.0f}")
    sim_app.close()


# ── 구 Baseline (하위 호환) ──────────────────────────────────────────────────

def run_baseline(episodes: int = 200):
    env = AirbagEnv(headless=True, debug=args.debug)
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        angle             = env.scenario["angle"]
        is_rollover       = bool(env.scenario.get("is_rollover",       False))
        passenger_present = bool(env.scenario.get("passenger_present", True))
        action_matrix = rule_based_policy(angle, is_rollover, passenger_present)
        action = np.concatenate([
            action_matrix[:, 0],
            action_matrix[:, 1] / 30.0,
            action_matrix[:, 2] / 250.0,
        ])
        total_r = 0.0
        done = False
        while not done:
            obs, r, done, _, _ = env.step(action)
            total_r += r
        rewards.append(total_r)
        if (ep + 1) % 50 == 0:
            print(f"[Baseline] ep {ep+1}/{episodes} | "
                  f"mean_reward={np.mean(rewards[-50:]):.1f}", flush=True)
    env.close()
    np.save("results/logs/baseline_rewards.npy", rewards)
    print(f"[Baseline] done. mean_reward={np.mean(rewards):.1f}")
    sim_app.close()


# ── 학습 공통 루프 ──────────────────────────────────────────────────────────

def run_train(curriculum: bool = False):
    """Pure PPO (curriculum=False) 또는 기존 3단계 커리큘럼 (curriculum=True)."""
    default_label = "curriculum" if curriculum else "pure"
    label         = args.label or default_label
    N             = args.episodes or cfg["train"]["total_episodes"]
    seed          = args.seed
    batch_size    = cfg["ppo"]["batch_size"]
    save_interval = cfg["train"]["save_interval"]
    log_interval  = cfg["train"]["log_interval"]
    csv_path      = f"results/logs/{label}_ppo_checkpoints.csv"

    import datetime
    EP_RATE_EST = 11.0   # ep/min (2프로세스 병렬 기준, 직전 3000ep 실측 보수적 추정)
    est_min = N / EP_RATE_EST
    eta = datetime.datetime.now() + datetime.timedelta(minutes=est_min)

    print(f"\n{'='*65}")
    print(f"[{label.upper()}] 학습 시작  총={N}ep  batch={batch_size}  "
          f"seed={seed}  ppo_epochs={cfg['ppo']['epochs']}")
    print(f"  [시간 예측] {EP_RATE_EST:.0f} ep/min 가정 → {N}ep ≈ {est_min/60:.1f}시간 소요")
    print(f"  [예상 완료] {eta.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}\n", flush=True)

    env = AirbagEnv(headless=True, debug=args.debug)
    if seed is not None:
        env.sampler = ScenarioSampler(seed=seed)

    agent = PPOAgent(
        state_dim=cfg["env"]["state_dim"],
        lr=cfg["ppo"]["lr"],
        gamma=cfg["ppo"]["gamma"],
        clip=cfg["ppo"]["clip"],
        epochs=cfg["ppo"]["epochs"],
        lam=cfg["ppo"].get("lam", 0.95),
        entropy_coeff=cfg["ppo"].get("entropy_coeff", 0.01),
    )

    all_rewards    = []
    all_hic15      = []
    all_chest_g    = []
    all_crit_loss  = []
    buffer         = []
    current_stage  = -1
    last_cl        = float("nan")
    wall_start     = time.time()
    ckpt_hic15     = []
    ckpt_chest_g   = []

    for ep in range(1, N + 1):

        # ── 커리큘럼: 에피소드마다 단계 설정 ─────────────────────────────
        if curriculum:
            stage, vc = get_stage(ep, N)
            if stage != current_stage:
                current_stage = stage
                print(f"[Curriculum] ===== Stage {stage} 시작 (EP {ep}/{N}) "
                      f"violation_coeff={vc} =====", flush=True)
            env.sampler.stage   = stage
            env.violation_coeff = vc
        # else: sampler.stage=0(Pure), violation_coeff=5.0 고정

        obs, _ = env.reset()
        done      = False
        ep_reward = 0.0
        ep_info   = {}

        while not done:
            action, log_prob = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            if info:
                ep_info = info
            buffer.append({
                "state":      obs,
                "action":     action,
                "log_prob":   log_prob,
                "reward":     reward,
                "next_state": next_obs,
                "done":       done,
            })
            ep_reward += reward
            obs = next_obs

        h = ep_info.get("hic15",   float("nan"))
        c = ep_info.get("chest_g", float("nan"))
        all_rewards.append(ep_reward)
        all_hic15.append(h)
        all_chest_g.append(c)
        ckpt_hic15.append(h)
        ckpt_chest_g.append(c)

        # ── 100ep 체크포인트 CSV ───────────────────────────────────────────
        if ep % 100 == 0:
            _write_checkpoint_csv(csv_path, ep, ckpt_hic15, ckpt_chest_g)
            ckpt_hic15.clear()
            ckpt_chest_g.clear()

        # ── PPO 업데이트 ───────────────────────────────────────────────────
        if len(buffer) >= batch_size:
            losses = agent.update(buffer)
            buffer = []
            last_cl = losses.get("critic_loss", float("nan"))
            all_crit_loss.append(last_cl)

        # ── 주기적 로그 ────────────────────────────────────────────────────
        if ep % log_interval == 0:
            mean_r  = np.nanmean(all_rewards[-log_interval:])
            mean_h  = np.nanmean(all_hic15[-log_interval:])
            mean_c  = np.nanmean(all_chest_g[-log_interval:])
            elapsed = time.time() - wall_start
            stage_str = f" st={current_stage}" if curriculum else ""
            print(
                f"[{label}] ep {ep:>4}/{N}{stage_str} | "
                f"mean_r={mean_r:>10.0f} | "
                f"HIC15={mean_h:>8.0f} | "
                f"chest_g={mean_c:>6.1f}g | "
                f"critic_loss={last_cl:.3f} | "
                f"t={elapsed/60:.1f}min",
                flush=True,
            )

        # ── 주기적 저장 ────────────────────────────────────────────────────
        if ep % save_interval == 0:
            agent.save(f"results/models/{label}_ppo_ep{ep}.pt")
            _save_logs(label, all_rewards, all_hic15, all_chest_g, all_crit_loss)

    # ── 최종 저장 ──────────────────────────────────────────────────────────
    agent.save(f"results/models/{label}_ppo_final.pt")
    _save_logs(label, all_rewards, all_hic15, all_chest_g, all_crit_loss)

    elapsed = time.time() - wall_start
    print(f"\n[{label.upper()}] 학습 완료  {N}ep  소요={elapsed/60:.1f}분", flush=True)
    print(f"  mean_reward (전체)  : {np.nanmean(all_rewards):.1f}")
    print(f"  mean_HIC15  (전체)  : {np.nanmean(all_hic15):.0f}")
    print(f"  mean_chest_g (전체) : {np.nanmean(all_chest_g):.1f}g")
    print(f"  마지막 critic_loss  : {last_cl:.4f}", flush=True)

    env.close()
    sim_app.close()


# ── 하이브리드 커리큘럼 PPO ──────────────────────────────────────────────────

def run_hybrid_curriculum():
    """
    4단계 하이브리드 커리큘럼 PPO (get_hybrid_stage 스케줄 적용).
    - Stage 1/2: 가이드레일 항(correct/wrong/over/late) 점진적 감쇠
    - Stage 3/4: 가이드레일 0, violation_coeff + peak_weight로만 미세조정
    """
    N             = args.episodes or 3000
    seed          = args.seed
    label         = args.label or "hybrid_3k"
    batch_size    = cfg["ppo"]["batch_size"]
    save_interval = cfg["train"]["save_interval"]
    log_interval  = cfg["train"]["log_interval"]
    csv_path      = f"results/logs/{label}_ppo_checkpoints.csv"

    print(f"\n{'='*65}")
    print(f"[HYBRID_CURRICULUM] 학습 시작  총={N}ep  batch={batch_size}  "
          f"seed={seed}  label={label}  ppo_epochs={cfg['ppo']['epochs']}")
    print(f"{'='*65}\n", flush=True)

    env = AirbagEnv(headless=True, debug=args.debug)
    if seed is not None:
        env.sampler = ScenarioSampler(seed=seed)

    agent = PPOAgent(
        state_dim=cfg["env"]["state_dim"],
        lr=cfg["ppo"]["lr"],
        gamma=cfg["ppo"]["gamma"],
        clip=cfg["ppo"]["clip"],
        epochs=cfg["ppo"]["epochs"],
        lam=cfg["ppo"].get("lam", 0.95),
        entropy_coeff=cfg["ppo"].get("entropy_coeff", 0.01),
    )

    all_rewards    = []
    all_hic15      = []
    all_chest_g    = []
    all_crit_loss  = []
    buffer         = []
    current_stage  = -1
    last_cl        = float("nan")
    wall_start     = time.time()
    ckpt_hic15     = []
    ckpt_chest_g   = []

    for ep in range(1, N + 1):

        # ── 하이브리드 커리큘럼 단계 설정 ────────────────────────────────
        sched = get_hybrid_stage(ep, N)
        stage = sched["stage"]
        if stage != current_stage:
            current_stage = stage
            print(
                f"[HybridCur] ===== Stage {stage} 시작 (EP {ep}/{N}) "
                f"vc={sched['violation_coeff']}  "
                f"correct={sched['correct_weight']}  wrong={sched['wrong_weight']}  "
                f"over={sched['over_weight']}  late={sched['late_weight']}  "
                f"peak={sched['peak_weight']} =====",
                flush=True,
            )
        env.sampler.stage    = stage
        env.violation_coeff  = sched["violation_coeff"]
        env.correct_weight   = sched["correct_weight"]
        env.wrong_weight     = sched["wrong_weight"]
        env.over_weight      = sched["over_weight"]
        env.late_weight      = sched["late_weight"]
        env.peak_weight      = sched["peak_weight"]

        obs, _ = env.reset()
        done      = False
        ep_reward = 0.0
        ep_info   = {}

        while not done:
            action, log_prob = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            if info:
                ep_info = info
            buffer.append({
                "state":      obs,
                "action":     action,
                "log_prob":   log_prob,
                "reward":     reward,
                "next_state": next_obs,
                "done":       done,
            })
            ep_reward += reward
            obs = next_obs

        h = ep_info.get("hic15",   float("nan"))
        c = ep_info.get("chest_g", float("nan"))
        all_rewards.append(ep_reward)
        all_hic15.append(h)
        all_chest_g.append(c)
        ckpt_hic15.append(h)
        ckpt_chest_g.append(c)

        # ── 100ep 체크포인트 CSV ───────────────────────────────────────────
        if ep % 100 == 0:
            _write_checkpoint_csv(csv_path, ep, ckpt_hic15, ckpt_chest_g)
            ckpt_hic15.clear()
            ckpt_chest_g.clear()

        # ── PPO 업데이트 ───────────────────────────────────────────────────
        if len(buffer) >= batch_size:
            losses = agent.update(buffer)
            buffer = []
            last_cl = losses.get("critic_loss", float("nan"))
            all_crit_loss.append(last_cl)

        # ── 주기적 로그 ────────────────────────────────────────────────────
        if ep % log_interval == 0:
            mean_r  = np.nanmean(all_rewards[-log_interval:])
            mean_h  = np.nanmean(all_hic15[-log_interval:])
            mean_c  = np.nanmean(all_chest_g[-log_interval:])
            elapsed = time.time() - wall_start
            print(
                f"[hybrid] ep {ep:>4}/{N} st={current_stage} | "
                f"mean_r={mean_r:>10.0f} | "
                f"HIC15={mean_h:>8.0f} | "
                f"chest_g={mean_c:>6.1f}g | "
                f"critic_loss={last_cl:.3f} | "
                f"t={elapsed/60:.1f}min",
                flush=True,
            )

        # ── 주기적 저장 ────────────────────────────────────────────────────
        if ep % save_interval == 0:
            agent.save(f"results/models/{label}_ppo_ep{ep}.pt")
            _save_logs(label, all_rewards, all_hic15, all_chest_g, all_crit_loss)

    # ── 최종 저장 ──────────────────────────────────────────────────────────
    agent.save(f"results/models/{label}_ppo_final.pt")
    _save_logs(label, all_rewards, all_hic15, all_chest_g, all_crit_loss)

    elapsed = time.time() - wall_start
    h_arr = np.array(all_hic15)
    h_f   = h_arr[np.isfinite(h_arr) & (h_arr > 0)]
    print(f"\n[HYBRID_CURRICULUM] 학습 완료  {N}ep  label={label}  소요={elapsed/60:.1f}분", flush=True)
    print(f"  mean_reward (전체)  : {np.nanmean(all_rewards):.1f}")
    print(f"  HIC15 median (전체) : {np.median(h_f) if len(h_f) else float('nan'):.0f}")
    print(f"  mean_chest_g (전체) : {np.nanmean(all_chest_g):.1f}g")
    print(f"  마지막 critic_loss  : {last_cl:.4f}", flush=True)

    env.close()
    sim_app.close()


def _save_logs(label, rewards, hic15, chest_g, crit_loss):
    np.save(f"results/logs/{label}_ppo_rewards.npy",   rewards)
    np.save(f"results/logs/{label}_ppo_hic15.npy",     hic15)
    np.save(f"results/logs/{label}_ppo_chest_g.npy",   chest_g)
    np.save(f"results/logs/{label}_ppo_critic_loss.npy", crit_loss)


# ── 진입점 ──────────────────────────────────────────────────────────────────

if args.mode == "rulebased":
    run_rulebased()
elif args.mode == "baseline":
    run_baseline(args.episodes or 200)
elif args.mode == "hybrid_curriculum":
    run_hybrid_curriculum()
elif args.mode == "curriculum":
    run_train(curriculum=True)
else:
    run_train(curriculum=False)
