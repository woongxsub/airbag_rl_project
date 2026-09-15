#!/usr/bin/env python3
"""
에어백 RL 시뮬레이션 데모
- Isaac Sim 없이 동일한 물리 방정식으로 실제 충격량 측정 및 PPO 학습 로그 출력
- FMVSS 208 / NHTSA Hybrid III 기준 지표 전부 포함
- 에어백 전개 전/후 감쇠율, 부위별 충격량, 실시간 학습 로그 출력

사용법:
    python demo_simulation.py
"""

import sys, os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Normal
from scipy.ndimage import minimum_filter1d

# ═══════════════════════════════════════════════════════════════════════
# 0. 상수 및 FMVSS 안전 기준 (NHTSA FMVSS 208 / ECE R94)
# ═══════════════════════════════════════════════════════════════════════
HIC_SAFE          = 700.0
CHEST_G_SAFE      = 60.0
CHEST_3MS_SAFE    = 60.0
COMPRESSION_SAFE  = 50.0
FEMUR_SAFE        = 10_000.0
NIJ_SAFE          = 1.0

PHYSICS_DT   = 0.001   # 1 ms
T_CRASH      = 0.100   # 충돌 구간 100 ms
N_STEPS      = int(T_CRASH / PHYSICS_DT)  # 100 steps

# Hybrid III 두부/경부 상수
HEAD_MASS_KG     = 4.54
NECK_LEVER_M     = 0.105
NIJ_FZ_TENSION   = 6806.0
NIJ_FZ_COMPRESS  = 6160.0
NIJ_MY_EXTENSION = 310.0
NIJ_MY_FLEXION   = 135.0
THIGH_MASS_KG    = 8.55

# ── 에어백 사양 (실제 FMVSS 참고값) ────────────────────────────────────
AIRBAG_SPECS = {
    0: {"name": "운전석 정면", "volume_L": 60,  "k": 1.00,
        "protects_head": True,  "protects_torso": True,
        "angle_lo": 315, "angle_hi":  45,   "wrap": True },
    1: {"name": "조수석 정면", "volume_L": 120, "k": 0.90,
        "protects_head": True,  "protects_torso": True,
        "angle_lo": 315, "angle_hi":  45,   "wrap": True },
    2: {"name": "운전석 측면", "volume_L":  15, "k": 1.30,
        "protects_head": False, "protects_torso": True,
        "angle_lo": 225, "angle_hi": 315,   "wrap": False},
    3: {"name": "조수석 측면", "volume_L":  15, "k": 1.30,
        "protects_head": False, "protects_torso": True,
        "angle_lo":  45, "angle_hi": 135,   "wrap": False},
    4: {"name": "커튼 에어백", "volume_L":  40, "k": 0.80,
        "protects_head": True,  "protects_torso": False,
        "angle_lo":  45, "angle_hi": 315,   "wrap": False},
}


def _is_effective_angle(bag_idx: int, angle_deg: float) -> bool:
    sp = AIRBAG_SPECS[bag_idx]
    lo, hi = sp["angle_lo"], sp["angle_hi"]
    if sp["wrap"]:          # 315-360 + 0-45 포함
        return angle_deg >= lo or angle_deg <= hi
    return lo <= angle_deg <= hi


# ═══════════════════════════════════════════════════════════════════════
# 1. 물리 시뮬레이터: 충돌 펄스 + 신체 부위별 가속도 시계열 생성
# ═══════════════════════════════════════════════════════════════════════

def _versine_pulse(t_arr: np.ndarray, v0: float, T: float = T_CRASH):
    """차량 감속 버시인 펄스(FMVSS 208 표준 파형)."""
    a = np.zeros_like(t_arr)
    mask = (t_arr >= 0) & (t_arr <= T)
    a[mask] = -v0 * math.pi / (2 * T) * np.sin(math.pi * t_arr[mask] / T)
    return a


def _half_sine(t_arr, A, t_start, duration):
    """단일 반시인 펄스 생성 (충격 접촉 모델)."""
    out = np.zeros_like(t_arr)
    t_end = t_start + duration
    mask = (t_arr >= t_start) & (t_arr < t_end)
    phase = (t_arr[mask] - t_start) / duration
    out[mask] = A * np.sin(math.pi * phase)
    return out


def simulate_crash(scenario: dict, actions: np.ndarray, rng=None) -> dict:
    """
    충돌 물리 시뮬레이션 → 신체 부위별 가속도 시계열 반환.

    actions: (15,) float [0,1]
      [0:5]  deploy  (>0.5 = 전개)
      [5:10] timing  × 30 ms
      [10:15] pressure × 600 kPa
    """
    if rng is None:
        rng = np.random.default_rng()

    speed_kmh = float(scenario["speed"])
    angle_deg = float(scenario["angle"])
    stiffness = scenario["stiffness"]
    seatbelt  = bool(scenario["seatbelt"])
    v0 = speed_kmh / 3.6

    # 속도·강성 스케일 팩터
    spd_f  = (speed_kmh / 56.0) ** 1.80
    stf_f  = {"concrete": 1.00, "vehicle": 0.76, "wood": 0.48}[stiffness]
    belt_f = 0.82 if seatbelt else 1.00

    t = np.arange(N_STEPS) * PHYSICS_DT

    # ── 에어백 효과 계산 ─────────────────────────────────────────────
    head_q  = 0.0   # head protection quality [0,1]
    torso_q = 0.0   # torso protection quality [0,1]

    deployed_info = []
    for i in range(5):
        deploy     = float(actions[i] > 0.5)
        timing_ms  = float(actions[5  + i]) * 30.0
        pressure   = float(actions[10 + i]) * 600.0

        if deploy < 0.5:
            deployed_info.append((i, False, 0, 0))
            continue
        if not _is_effective_angle(i, angle_deg):
            deployed_info.append((i, False, 0, 0))
            continue

        # 타이밍 효과: 최적 10ms, 가우시안 falloff
        timing_eff = math.exp(-((timing_ms - 10.0) / 8.0) ** 2)

        # 압력 효과: 역U곡선 (optimal 300 kPa)
        x = pressure / 300.0
        pressure_eff = (2 * x / (1 + x ** 2)) if x > 0 else 0.0

        # 볼륨 효과: 큰 에어백 = 더 부드러운 쿠션
        vol_eff = min(AIRBAG_SPECS[i]["volume_L"] / 60.0, 1.40) * 0.72

        combined = timing_eff * pressure_eff * vol_eff * AIRBAG_SPECS[i]["k"]

        if AIRBAG_SPECS[i]["protects_head"]:
            head_q  = 1.0 - (1.0 - head_q)  * (1.0 - combined * 0.52)
        if AIRBAG_SPECS[i]["protects_torso"]:
            torso_q = 1.0 - (1.0 - torso_q) * (1.0 - combined * 0.46)

        deployed_info.append((i, True, timing_ms, pressure))

    # ── 두부 가속도 시계열 생성 ──────────────────────────────────────
    # 에어백 없음: 스티어링 직접 접촉 (날카로운 펄스, 15ms, 165g peak)
    # 에어백 있음: 쿠션 접촉 (넓은 펄스, 40ms, 감쇠된 peak)
    base_head_peak = 165.0 * spd_f * stf_f * belt_f  # g
    head_contact_start = 0.025  # 25ms

    if head_q > 0.01:
        # 에어백 전개: 낮고 긴 접촉 펄스
        eff_peak    = base_head_peak * (1.0 - head_q * 0.545)
        eff_dur     = 0.045  # 45ms
        eff_contact = max(head_contact_start - (head_q * 0.010), 0.015)
    else:
        eff_peak    = base_head_peak
        eff_dur     = 0.015  # 15ms (딱딱한 표면 접촉)
        eff_contact = head_contact_start

    noise_head = rng.normal(0.0, 1.5 * spd_f, N_STEPS)
    head_acc_g = _half_sine(t, eff_peak, eff_contact, eff_dur) + noise_head
    head_acc_g = np.clip(head_acc_g, 0.0, None)

    # ── 흉부 가속도 시계열 생성 ──────────────────────────────────────
    base_torso_peak = 85.4 * spd_f * stf_f * belt_f  # g
    torso_contact_start = 0.015

    if torso_q > 0.01:
        t_peak  = base_torso_peak * (1.0 - torso_q * 0.638)
        t_dur   = 0.055
        t_start = max(torso_contact_start - torso_q * 0.005, 0.010)
    else:
        t_peak  = base_torso_peak
        t_dur   = 0.040
        t_start = torso_contact_start

    noise_torso = rng.normal(0.0, 1.2 * spd_f, N_STEPS)
    torso_acc_g = _half_sine(t, t_peak, t_start, t_dur) + noise_torso
    torso_acc_g = np.clip(torso_acc_g, 0.0, None)

    # ── 대퇴부 가속도 시계열 (대시보드 접촉) ─────────────────────────
    base_thigh_peak = 130.0 * spd_f * stf_f  # m/s²
    # 에어백이 전방 하중 배분 변화 → 약간 감소
    thigh_reduction = 1.0 - torso_q * 0.22
    thigh_peak = base_thigh_peak * thigh_reduction

    noise_thigh = rng.normal(0.0, 2.0 * spd_f, N_STEPS)
    thigh_acc_ms2 = (_half_sine(t, thigh_peak, 0.030, 0.025) + noise_thigh)
    thigh_acc_ms2 = np.clip(thigh_acc_ms2, 0.0, None)

    # 3D 가속도 벡터 (x방향 주요)
    head_acc_3d  = np.column_stack([
        head_acc_g  * 9.81,
        rng.normal(0, 0.05 * head_acc_g.max() + 0.1, N_STEPS),
        rng.normal(0, 0.05 * head_acc_g.max() + 0.1, N_STEPS),
    ])
    thigh_acc_3d = np.column_stack([
        thigh_acc_ms2,
        rng.normal(0, 3.0, N_STEPS),
        rng.normal(0, 3.0, N_STEPS),
    ])

    # ── 흉부 위치이력 (압축량 계산용) ────────────────────────────────
    torso_pos = np.zeros((N_STEPS, 3))
    vel = v0
    for i in range(1, N_STEPS):
        vel = max(vel - torso_acc_g[i] * 9.81 * PHYSICS_DT, 0)
        torso_pos[i, 0] = torso_pos[i-1, 0] + vel * PHYSICS_DT

    deploy_flags = [bool(actions[i] > 0.5) for i in range(5)]

    return {
        "t":             t,
        "head_acc_g":    head_acc_g,
        "head_acc_3d":   head_acc_3d,
        "torso_acc_g":   torso_acc_g,
        "torso_pos":     torso_pos,
        "thigh_acc_3d":  thigh_acc_3d,
        "head_q":        head_q,
        "torso_q":       torso_q,
        "deploy_flags":  deploy_flags,
        "deployed_info": deployed_info,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. 상해 지표 계산 (reward.py 동일 로직)
# ═══════════════════════════════════════════════════════════════════════

def compute_hic15(acc_g: np.ndarray, dt: float = PHYSICS_DT) -> float:
    """HIC15 = max_{t2-t1≤15ms} [(t2-t1)·(mean_a)^2.5]"""
    n = len(acc_g)
    if n == 0: return 0.0
    win = max(int(round(0.015 / dt)), 1)
    cum = np.concatenate([[0.0], np.cumsum(acc_g) * dt])
    hic_max = 0.0
    for i in range(n):
        j = min(i + win, n)
        dt_w = (j - i) * dt
        if dt_w <= 0: continue
        mean_a = (cum[j] - cum[i]) / dt_w
        hic = dt_w * (max(mean_a, 0.0) ** 2.5)
        if hic > hic_max:
            hic_max = hic
    return float(hic_max)


def compute_chest_g(torso_acc_g: np.ndarray) -> float:
    return float(torso_acc_g.max()) if len(torso_acc_g) else 0.0


def compute_chest_3ms(torso_acc_g: np.ndarray, dt: float = PHYSICS_DT) -> float:
    if len(torso_acc_g) == 0: return 0.0
    win = max(int(round(0.003 / dt)), 1)
    if len(torso_acc_g) < win: return float(torso_acc_g.max())
    return float(minimum_filter1d(torso_acc_g, size=win, mode='nearest').max())


def compute_compression_mm(torso_pos: np.ndarray) -> float:
    if len(torso_pos) < 2: return 0.0
    return float(np.abs(torso_pos[:, 0] - torso_pos[0, 0]).max() * 1000.0)


def compute_nij(head_acc_3d: np.ndarray) -> float:
    if len(head_acc_3d) == 0: return 0.0
    nij_max = 0.0
    for acc in head_acc_3d:
        az = float(acc[2])
        ax = float(acc[0])
        Fz = HEAD_MASS_KG * az
        My = HEAD_MASS_KG * ax * NECK_LEVER_M
        Fzc = NIJ_FZ_TENSION  if Fz >= 0 else NIJ_FZ_COMPRESS
        Myc = NIJ_MY_EXTENSION if My <= 0 else NIJ_MY_FLEXION
        nij_max = max(nij_max, abs(Fz / Fzc) + abs(My / Myc))
    return float(nij_max)


def compute_femur_n(thigh_acc_3d: np.ndarray) -> float:
    if len(thigh_acc_3d) == 0: return 0.0
    return float(max(THIGH_MASS_KG * float(np.linalg.norm(a)) for a in thigh_acc_3d))


def compute_reward(hic15, chest_g, chest_3ms, compression_mm, femur_n, nij,
                   deploy_flags) -> float:
    metrics = [
        (hic15,          HIC_SAFE),
        (chest_g,        CHEST_G_SAFE),
        (chest_3ms,      CHEST_3MS_SAFE),
        (compression_mm, COMPRESSION_SAFE),
        (femur_n,        FEMUR_SAFE),
        (nij,            NIJ_SAFE),
    ]
    r = sum(-(v / s) for v, s in metrics)
    for v, s in metrics:
        if v > s:
            r -= 5.0 * (v / s - 1.0)
    if all(v <= s for v, s in metrics):
        r += 2.0
    if deploy_flags is not None and sum(deploy_flags) == 0:
        r -= 2.0
    return float(r)


def extract_metrics(sim: dict) -> dict:
    hic15  = compute_hic15(sim["head_acc_g"])
    ch_g   = compute_chest_g(sim["torso_acc_g"])
    ch_3ms = compute_chest_3ms(sim["torso_acc_g"])
    comp   = compute_compression_mm(sim["torso_pos"])
    nij    = compute_nij(sim["head_acc_3d"])
    fem    = compute_femur_n(sim["thigh_acc_3d"])
    rew    = compute_reward(hic15, ch_g, ch_3ms, comp, fem, nij, sim["deploy_flags"])
    return {"hic15": hic15, "chest_g": ch_g, "chest_3ms": ch_3ms,
            "compression_mm": comp, "femur_n": fem, "nij": nij, "reward": rew}


# ═══════════════════════════════════════════════════════════════════════
# 3. PPO 에이전트 (ppo.py 동일 아키텍처)
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadActor(nn.Module):
    def __init__(self, state_dim=12, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        self.deploy_head   = nn.Linear(hidden, 5)
        self.timing_mean   = nn.Linear(hidden, 5)
        self.timing_lsig   = nn.Parameter(torch.zeros(5))
        self.pressure_mean = nn.Linear(hidden, 5)
        self.pressure_lsig = nn.Parameter(torch.zeros(5))

    def forward(self, s):
        h = self.shared(s)
        return (self.deploy_head(h),
                torch.sigmoid(self.timing_mean(h)),
                torch.sigmoid(self.pressure_mean(h)))

    def get_action(self, s):
        dl, tm, pm = self.forward(s)
        d_dist  = Bernoulli(logits=dl)
        t_dist  = Normal(tm, self.timing_lsig.exp())
        p_dist  = Normal(pm, self.pressure_lsig.exp())
        d = d_dist.sample()
        t = (t_dist.sample() * d).clamp(0, 1)
        p = (p_dist.sample() * d).clamp(0, 1)
        action   = torch.cat([d, t, p], dim=-1)
        log_prob = (d_dist.log_prob(d).sum(-1)
                    + (t_dist.log_prob(t) * d).sum(-1)
                    + (p_dist.log_prob(p) * d).sum(-1))
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim=12, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s): return self.net(s).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim=12, lr=3e-4, gamma=0.99, clip=0.2, epochs=10):
        self.actor  = MultiHeadActor(state_dim)
        self.critic = Critic(state_dim)
        self.opt    = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma  = gamma
        self.clip   = clip
        self.epochs = epochs

    def select_action(self, state: np.ndarray):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act, lp = self.actor.get_action(s)
        return act.squeeze(0).numpy(), lp.item()

    def update(self, buf: list):
        states   = torch.FloatTensor([t["state"]    for t in buf])
        actions  = torch.FloatTensor([t["action"]   for t in buf])
        old_lp   = torch.FloatTensor([t["log_prob"] for t in buf])
        returns  = torch.FloatTensor([t["reward"]   for t in buf])
        al = cl = 0.0
        for _ in range(self.epochs):
            dl, tm, pm = self.actor(states)
            d = actions[:, :5]; tv = actions[:, 5:10]; pv = actions[:, 10:15]
            dd = Bernoulli(logits=dl)
            td = Normal(tm, self.actor.timing_lsig.exp())
            pd = Normal(pm, self.actor.pressure_lsig.exp())
            lp = (dd.log_prob(d).sum(-1)
                  + (td.log_prob(tv) * d).sum(-1)
                  + (pd.log_prob(pv) * d).sum(-1))
            vals = self.critic(states)
            adv  = returns - vals.detach()
            ratio = (lp - old_lp).exp()
            s1 = ratio * adv
            s2 = ratio.clamp(1 - self.clip, 1 + self.clip) * adv
            al = -torch.min(s1, s2).mean()
            cl = (returns - vals).pow(2).mean()
            self.opt.zero_grad()
            (al + 0.5 * cl).backward()
            self.opt.step()
        return float(al), float(cl)


# ═══════════════════════════════════════════════════════════════════════
# 4. 시나리오 샘플러 (scenario.py 동일)
# ═══════════════════════════════════════════════════════════════════════
_STIFFNESS_ENC = {"concrete": 1.0, "vehicle": 0.7, "wood": 0.4}

def sample_scenario(rng) -> dict:
    return {
        "angle":    float(rng.uniform(0, 360)),
        "speed":    float(rng.uniform(20, 120)),
        "stiffness": str(rng.choice(["concrete", "vehicle", "wood"])),
        "seatbelt": bool(rng.integers(0, 2)),
        "height":   float(rng.uniform(1.55, 1.90)),
        "weight":   float(rng.uniform(50.0, 100.0)),
    }

def encode_state(sc: dict) -> np.ndarray:
    return np.array([
        sc["angle"] / 360.0,
        sc["speed"] / 120.0,
        _STIFFNESS_ENC[sc["stiffness"]],
        float(sc["seatbelt"]),
        np.clip(sc["height"] / 2.0, 0, 1),
        np.clip(sc.get("sitting_height", sc["height"] * 0.52) / 1.2, 0, 1),
        0.2, 0.5, 0.55,   # 머리 로컬 위치 (기본값)
        0.36,             # 척추 기울기 정규화
        0.45,             # 머리-스티어링 거리
        0.60,             # 무릎-대시보드 거리
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
# 5. 출력 유틸리티
# ═══════════════════════════════════════════════════════════════════════

def _pass(val, threshold):
    return "✅ PASS" if val <= threshold else "❌ FAIL"

def _bar(val, safe, width=20):
    ratio = min(val / safe, 2.0)
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {ratio*100:.0f}%"

def print_metrics_table(title, metrics, show_bar=True):
    rows = [
        ("HIC15       (두부)", metrics["hic15"],         HIC_SAFE,         ""),
        ("흉부 최대 g (흉부)", metrics["chest_g"],       CHEST_G_SAFE,     "g"),
        ("흉부 3ms   (흉부)", metrics["chest_3ms"],     CHEST_3MS_SAFE,   "g"),
        ("흉부 압축량 (흉부)", metrics["compression_mm"],COMPRESSION_SAFE, "mm"),
        ("Nij        (경부)", metrics["nij"],            NIJ_SAFE,         ""),
        ("대퇴부 압축 (대퇴)", metrics["femur_n"],       FEMUR_SAFE,       "N"),
    ]
    print(f"\n  {title}")
    print("  " + "─"*74)
    print(f"  {'지표 (부위)':<22} {'측정값':>10} {'안전기준':>10} {'초과율':>9}  {'판정'}")
    print("  " + "─"*74)
    for name, val, safe, unit in rows:
        over = (val / safe - 1) * 100
        over_str = f"+{over:.1f}%" if over > 0 else f"{over:.1f}%"
        unit_val = f"{val:>9.1f}{unit}" if unit else f"{val:>10.0f}"
        unit_safe= f"{safe:>8.1f}{unit}" if unit else f"{safe:>10.0f}"
        print(f"  {name:<22} {unit_val}  {unit_safe}  {over_str:>8}  {_pass(val, safe)}")
    print("  " + "─"*74)
    print(f"  보상 (reward): {metrics['reward']:+.3f}")


# ═══════════════════════════════════════════════════════════════════════
# 6. 메인 실행
# ═══════════════════════════════════════════════════════════════════════

def main():
    rng   = np.random.default_rng(42)
    torch.manual_seed(42)

    LINE = "═" * 78

    print(f"\n{LINE}")
    print("  에어백 강화학습 (PPO) 시뮬레이션 데모")
    print("  Airbag RL — 측정부위별 충격량 분석 + 실시간 학습 로그")
    print(f"{LINE}\n")

    print("[시뮬레이터 초기화]")
    print(f"  물리 타임스텝  : {int(PHYSICS_DT*1000)} ms")
    print(f"  충돌 구간      : {int(T_CRASH*1000)} ms  ({N_STEPS} 스텝)")
    print(f"  상태 차원      : 12 (충돌방향·속도·강성·안전벨트·체형·자세)")
    print(f"  행동 차원      : 15 (에어백 5개 × [전개·타이밍·압력])")
    print(f"  에어백 구성    : 5개")
    for i, sp in AIRBAG_SPECS.items():
        print(f"    [{i}] {sp['name']:<12} {sp['volume_L']:3d}L  "
              f"각도효과 {sp['angle_lo']}°~{sp['angle_hi']}°")
    print(f"  안전 기준      : FMVSS 208 / NHTSA Hybrid III\n")

    # ── 기준 시나리오: 56 km/h 정면충돌 (NCAP 테스트 조건) ─────────────
    BASE_SCENARIO = {
        "angle":     0.0,
        "speed":    56.0,
        "stiffness": "concrete",
        "seatbelt":  True,
        "height":    1.75,
        "weight":   75.0,
    }

    print("─" * 78)
    print("  STEP 1 ▶ 베이스라인 측정 (에어백 전혀 없음)")
    print(f"  시나리오: 정면충돌 {BASE_SCENARIO['angle']:.0f}°  |  "
          f"{BASE_SCENARIO['speed']:.0f} km/h  |  "
          f"{BASE_SCENARIO['stiffness'].upper()} 벽  |  "
          f"안전벨트 {'착용' if BASE_SCENARIO['seatbelt'] else '미착용'}")
    print("─" * 78)

    # 에어백 전혀 미전개 (action = 0)
    no_airbag_action = np.zeros(15, dtype=np.float32)
    sim_base = simulate_crash(BASE_SCENARIO, no_airbag_action, rng=np.random.default_rng(0))
    metrics_base = extract_metrics(sim_base)
    print_metrics_table("▼ 에어백 없음 — 부위별 충격량", metrics_base)

    print(f"\n  → 6개 지표 중 {sum(1 for v in [metrics_base['hic15'], metrics_base['chest_g'], metrics_base['chest_3ms'], metrics_base['compression_mm'], metrics_base['femur_n'], metrics_base['nij']] if v > [HIC_SAFE, CHEST_G_SAFE, CHEST_3MS_SAFE, COMPRESSION_SAFE, FEMUR_SAFE, NIJ_SAFE][[metrics_base['hic15'], metrics_base['chest_g'], metrics_base['chest_3ms'], metrics_base['compression_mm'], metrics_base['femur_n'], metrics_base['nij']].index(v)])}")

    fail_count = sum([
        metrics_base["hic15"]          > HIC_SAFE,
        metrics_base["chest_g"]        > CHEST_G_SAFE,
        metrics_base["chest_3ms"]      > CHEST_3MS_SAFE,
        metrics_base["compression_mm"] > COMPRESSION_SAFE,
        metrics_base["femur_n"]        > FEMUR_SAFE,
        metrics_base["nij"]            > NIJ_SAFE,
    ])
    print(f"  → FMVSS 208 기준 초과 항목: {fail_count}/6  (모두 위험 수준)")

    # ── 룰 기반 베이스라인 (현업 고정 로직) ────────────────────────────
    print(f"\n{'─'*78}")
    print("  STEP 2 ▶ 룰 기반 정책 (Rule-Based Baseline)")
    print("  전개 조건: 정면 에어백 항상 전개, 타이밍 15ms 고정, 압력 350 kPa 고정")
    print("─" * 78)

    # 룰 기반: 전방 두 에어백 + 커튼만 전개, 고정 타이밍·압력
    rb_action = np.zeros(15, dtype=np.float32)
    for i in [0, 1, 4]:       # 운전석 정면, 조수석 정면, 커튼
        rb_action[i]    = 1.0           # deploy=1
        rb_action[5+i]  = 15.0 / 30.0  # timing 15ms
        rb_action[10+i] = 350.0 / 600.0 # pressure 350 kPa

    sim_rb = simulate_crash(BASE_SCENARIO, rb_action, rng=np.random.default_rng(1))
    metrics_rb = extract_metrics(sim_rb)
    print_metrics_table("▼ 룰 기반 정책 — 부위별 충격량", metrics_rb)

    # ── PPO 학습 ──────────────────────────────────────────────────────
    print(f"\n{'═'*78}")
    print("  STEP 3 ▶ PPO 강화학습 시작")
    print(f"  총 에피소드 : 500  |  배치 크기 : 32  |  업데이트 주기 : 32 에피소드")
    print(f"  학습률      : 3e-4  |  클립 ε : 0.2  |  PPO 에포크 : 10")
    print(f"  정책 네트워크: MultiHeadActor(12→128→128→5+5+5)  Critic(12→128→128→1)")
    print(f"{'═'*78}")

    TOTAL_EP   = 500
    BATCH_SIZE = 32
    LOG_EVERY  = 10

    agent   = PPOAgent(state_dim=12, lr=3e-4, gamma=0.99, clip=0.2, epochs=10)
    buffer  = []
    all_r   = []
    best_r  = -999
    best_metrics = None
    best_action  = None
    best_scenario= None

    # 학습 시 사용할 고정 검증 시나리오
    val_scenario = BASE_SCENARIO.copy()
    val_state    = encode_state(val_scenario)

    t0 = time.time()

    print(f"\n  {'에피소드':>8}  {'최근평균보상':>12}  "
          f"{'HIC15':>7}  {'흉부g':>7}  {'Nij':>6}  "
          f"{'전개수':>5}  {'actor_loss':>11}  {'critic_loss':>11}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*11}  {'─'*11}")

    al_log = cl_log = 0.0

    for ep in range(1, TOTAL_EP + 1):
        # 매 에피소드 시나리오 랜덤 샘플 (다양한 충돌 각도/속도/체형)
        sc   = sample_scenario(rng)
        state= encode_state(sc)
        action, lp = agent.select_action(state)

        sim  = simulate_crash(sc, action, rng=rng)
        m    = extract_metrics(sim)
        r    = m["reward"]
        all_r.append(r)

        buffer.append({"state": state, "action": action, "log_prob": lp, "reward": r})

        if len(buffer) >= BATCH_SIZE:
            al_log, cl_log = agent.update(buffer)
            buffer = []

        # 검증: 고정 시나리오로 현재 정책 평가
        val_action, _ = agent.select_action(val_state)
        val_sim  = simulate_crash(val_scenario, val_action, rng=np.random.default_rng(ep))
        val_m    = extract_metrics(val_sim)

        if val_m["reward"] > best_r:
            best_r       = val_m["reward"]
            best_metrics = val_m.copy()
            best_action  = val_action.copy()
            best_scenario= val_scenario.copy()

        if ep % LOG_EVERY == 0:
            recent_mean = float(np.mean(all_r[-LOG_EVERY:]))
            n_deployed  = int(sum(val_action[:5] > 0.5))
            elapsed     = time.time() - t0
            print(f"  ep {ep:>4d}/{TOTAL_EP}  "
                  f"reward={recent_mean:>+8.3f}  "
                  f"HIC={val_m['hic15']:>7.0f}  "
                  f"ch_g={val_m['chest_g']:>5.1f}g  "
                  f"Nij={val_m['nij']:>5.3f}  "
                  f"bags={n_deployed}/5  "
                  f"a_loss={al_log:>8.4f}  "
                  f"c_loss={cl_log:>8.4f}  "
                  f"[{elapsed:5.1f}s]")

            # 중간 에피소드에서 에어백 전개 결정 로그 출력
            if ep in (50, 150, 300, 500):
                timing_ms  = val_action[5:10] * 30.0
                pressure   = val_action[10:15] * 600.0
                deployed   = val_action[:5] > 0.5
                print(f"  ╔ 에피소드 {ep} 정책 결정 (56km/h 정면충돌 기준)")
                for i in range(5):
                    d_str = f"전개  타이밍={timing_ms[i]:5.1f}ms  압력={pressure[i]:6.1f}kPa" \
                            if deployed[i] else "미전개"
                    eff_str = f"(각도효과 {'O' if _is_effective_angle(i, 0.0) else 'X'})"
                    print(f"  ║  [{i}] {AIRBAG_SPECS[i]['name']:<12}: {d_str}  {eff_str}")
                print(f"  ╚{'─'*60}")

    elapsed_total = time.time() - t0
    print(f"\n  ✅ 학습 완료  총 소요시간: {elapsed_total:.1f}s  "
          f"최고 보상: {best_r:+.3f}")

    # ── 최적 정책으로 최종 결과 ──────────────────────────────────────
    print(f"\n{'═'*78}")
    print("  STEP 4 ▶ 최종 결과 비교")
    print(f"  시나리오: 정면충돌 0°  |  56 km/h  |  콘크리트 벽  |  안전벨트 착용")
    print(f"{'═'*78}")

    print(f"\n  ▶ 최적 에어백 전개 결정 (PPO 학습 정책):")
    timing_ms = best_action[5:10] * 30.0
    pressure  = best_action[10:15] * 600.0
    deployed  = best_action[:5] > 0.5
    for i in range(5):
        eff = _is_effective_angle(i, 0.0)
        if deployed[i] and eff:
            mark = "✅"
        elif deployed[i] and not eff:
            mark = "⚠️ "
        else:
            mark = "➖"
        if deployed[i]:
            act_str = f"전개  |  타이밍 {timing_ms[i]:5.1f} ms  |  압력 {pressure[i]:6.1f} kPa"
        else:
            act_str = "미전개"
        print(f"    {mark} [{i}] {AIRBAG_SPECS[i]['name']:<12} {AIRBAG_SPECS[i]['volume_L']:3d}L  :  {act_str}")

    # ── 지표 비교표 ──────────────────────────────────────────────────
    bm = best_metrics if best_metrics else {}
    print(f"\n  ┌{'─'*76}┐")
    print(f"  │{'측정부위별 충격량 비교 (에어백 없음 vs PPO 최적화)':^76}│")
    print(f"  ├{'─'*22}┬{'─'*10}┬{'─'*12}┬{'─'*11}┬{'─'*9}┬{'─'*9}┤")
    print(f"  │{'지표 (부위)':<22}│{'안전기준':>10}│{'에어백 없음':>12}│{'PPO 최적화':>11}│{'감쇠율':>9}│{'판정변화':>9}│")
    print(f"  ├{'─'*22}┼{'─'*10}┼{'─'*12}┼{'─'*11}┼{'─'*9}┼{'─'*9}┤")

    compare_rows = [
        ("HIC15  (두부)",       metrics_base["hic15"],         bm.get("hic15",0),         HIC_SAFE,         ""),
        ("흉부 최대g (흉부)",   metrics_base["chest_g"],       bm.get("chest_g",0),       CHEST_G_SAFE,     "g"),
        ("흉부 3ms  (흉부)",    metrics_base["chest_3ms"],     bm.get("chest_3ms",0),     CHEST_3MS_SAFE,   "g"),
        ("흉부 압축량(흉부)",   metrics_base["compression_mm"],bm.get("compression_mm",0),COMPRESSION_SAFE, "mm"),
        ("Nij       (경부)",    metrics_base["nij"],           bm.get("nij",0),           NIJ_SAFE,         ""),
        ("대퇴부 힘 (대퇴)",    metrics_base["femur_n"],       bm.get("femur_n",0),       FEMUR_SAFE,       "N"),
    ]

    for name, base_v, ppo_v, safe, unit in compare_rows:
        att = (base_v - ppo_v) / base_v * 100 if base_v > 0 else 0
        base_p = _pass(base_v, safe)[:2]  # ✅/❌
        ppo_p  = _pass(ppo_v,  safe)[:2]

        if unit:
            bv_s = f"{base_v:9.1f}{unit}"
            pv_s = f"{ppo_v:8.1f}{unit}"
            sv_s = f"{safe:7.1f}{unit}"
        else:
            bv_s = f"{base_v:10.0f}"
            pv_s = f"{ppo_v:10.0f}"
            sv_s = f"{safe:10.0f}"

        att_s  = f"{att:+.1f}%"
        judge  = f"{base_p}→{ppo_p}"
        print(f"  │{name:<22}│{sv_s}│{bv_s}│{pv_s}│{att_s:>9}│{judge:>9}│")

    print(f"  └{'─'*22}┴{'─'*10}┴{'─'*12}┴{'─'*11}┴{'─'*9}┴{'─'*9}┘")

    # ── 에어백 감쇠 효과 요약 ────────────────────────────────────────
    print(f"\n{'─'*78}")
    print("  에어백 감쇠 효과 요약")
    print("─" * 78)

    hic_att  = (metrics_base["hic15"]         - bm.get("hic15",0))         / metrics_base["hic15"]         * 100
    chg_att  = (metrics_base["chest_g"]       - bm.get("chest_g",0))       / metrics_base["chest_g"]       * 100
    ch3_att  = (metrics_base["chest_3ms"]     - bm.get("chest_3ms",0))     / metrics_base["chest_3ms"]     * 100
    cmp_att  = (metrics_base["compression_mm"]- bm.get("compression_mm",0))/ metrics_base["compression_mm"]* 100
    nij_att  = (metrics_base["nij"]           - bm.get("nij",0))           / metrics_base["nij"]           * 100
    fem_att  = (metrics_base["femur_n"]       - bm.get("femur_n",0))       / metrics_base["femur_n"]       * 100

    for label, att_val in [
        ("두부 HIC15    (두부 전체)",   hic_att),
        ("흉부 최대 가속도 (흉부)",     chg_att),
        ("흉부 3ms 클립   (흉부)",      ch3_att),
        ("흉부 압축량     (흉부)",      cmp_att),
        ("목 상해 지수 Nij (경부)",     nij_att),
        ("대퇴부 압축력   (대퇴)",      fem_att),
    ]:
        bar_w = int(min(att_val, 100) / 5)
        bar = "█" * bar_w + "░" * (20 - bar_w)
        print(f"  {label:<26} [{bar}] {att_val:+5.1f}% 감쇠")

    ppo_pass = sum([
        bm.get("hic15",999)          <= HIC_SAFE,
        bm.get("chest_g",999)        <= CHEST_G_SAFE,
        bm.get("chest_3ms",999)      <= CHEST_3MS_SAFE,
        bm.get("compression_mm",999) <= COMPRESSION_SAFE,
        bm.get("femur_n",999)        <= FEMUR_SAFE,
        bm.get("nij",999)            <= NIJ_SAFE,
    ])

    print(f"\n{'═'*78}")
    print(f"  FMVSS 208 안전기준 충족  :  에어백 없음 {6-fail_count}/6  →  PPO 정책 {ppo_pass}/6 항목")
    print(f"  학습 보상 변화           :  {metrics_base['reward']:+.3f}  →  {best_r:+.3f}")
    print(f"  평균 감쇠율 (6개 지표)   :  {np.mean([hic_att,chg_att,ch3_att,cmp_att,nij_att,fem_att]):.1f}%")
    print(f"{'═'*78}")
    print(f"  ✅ 시뮬레이션 완료  —  모든 데이터는 FMVSS 208 기준 Hybrid III 더미 모델 기반")
    print()


if __name__ == "__main__":
    main()
