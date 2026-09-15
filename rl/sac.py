"""
SAC — Soft Actor-Critic (Haarnoja et al. 2018) for mixed action space.

Action layout (dim=15):
  [0:5]   deploy   — Bernoulli (이산×5)
  [5:10]  timing   — Squashed Gaussian → [0,1] (×5)
  [10:15] pressure — Squashed Gaussian → [0,1] (×5)

Deploy gradient:
  rollout → hard Bernoulli sample
  actor update → sigmoid soft-probability (straight-through) passed to critic
  이렇게 하면 reparameterization trick을 continuous 부분에만 적용 가능.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli, Normal

STATE_DIM  = 11
ACTION_DIM = 15
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-6


# ══════════════════════════════════════════════════════════════════════════
# Actor
# ══════════════════════════════════════════════════════════════════════════

class SACMixedActor(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.deploy_head  = nn.Linear(hidden, 5)
        self.t_mean_head  = nn.Linear(hidden, 5)
        self.t_ls_head    = nn.Linear(hidden, 5)   # log_std
        self.p_mean_head  = nn.Linear(hidden, 5)
        self.p_ls_head    = nn.Linear(hidden, 5)

    def _heads(self, state: torch.Tensor):
        h = self.net(state)
        return (
            self.deploy_head(h),
            self.t_mean_head(h),
            self.t_ls_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX),
            self.p_mean_head(h),
            self.p_ls_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX),
        )

    @staticmethod
    def _squash(mean: torch.Tensor, log_std: torch.Tensor):
        """
        u ~ N(μ, σ²)  →  a = (tanh(u)+1)/2 ∈ [0,1]
        log π(a) = log N(u|μ,σ) − Σ log(1−tanh²(u)) − log2
        reparameterizable (rsample)
        """
        std = log_std.exp()
        u   = Normal(mean, std).rsample()
        a   = (torch.tanh(u) + 1.0) / 2.0
        lp  = (
            Normal(mean, std).log_prob(u)
            - torch.log(1.0 - torch.tanh(u).pow(2) + EPS)
            - np.log(2.0)
        )
        return a, lp

    @staticmethod
    def _squash_mean(mean: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(mean) + 1.0) / 2.0

    # ── 롤아웃용 (hard deploy) ──────────────────────────────────────────
    def sample(self, state: torch.Tensor):
        d_logit, t_mu, t_ls, p_mu, p_ls = self._heads(state)
        d_dist  = Bernoulli(logits=d_logit)
        deploy  = d_dist.sample()                      # (B, 5) hard
        d_lp    = d_dist.log_prob(deploy)              # (B, 5)

        t_act, t_lp = self._squash(t_mu, t_ls)
        p_act, p_lp = self._squash(p_mu, p_ls)
        t_act = t_act * deploy
        p_act = p_act * deploy

        action   = torch.cat([deploy, t_act, p_act], dim=-1)
        log_prob = (d_lp + t_lp * deploy + p_lp * deploy).sum(-1)
        return action, log_prob

    # ── Actor 업데이트용 (soft deploy → reparameterizable) ──────────────
    def sample_for_actor(self, state: torch.Tensor):
        d_logit, t_mu, t_ls, p_mu, p_ls = self._heads(state)
        d_prob  = torch.sigmoid(d_logit)               # soft, differentiable
        d_dist  = Bernoulli(probs=d_prob)
        deploy_hard = d_dist.sample()                  # hard (for lp mask)
        d_lp    = d_dist.log_prob(deploy_hard)

        t_act, t_lp = self._squash(t_mu, t_ls)
        p_act, p_lp = self._squash(p_mu, p_ls)

        # soft masking: gradient flows through d_prob
        action = torch.cat([d_prob, t_act * d_prob, p_act * d_prob], dim=-1)
        log_prob = (d_lp + t_lp * deploy_hard + p_lp * deploy_hard).sum(-1)
        return action, log_prob

    # ── 평가용 (결정론적) ─────────────────────────────────────────────
    @torch.no_grad()
    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0)
        d_logit, t_mu, _, p_mu, _ = self._heads(s)
        deploy   = (torch.sigmoid(d_logit) > 0.5).float()
        timing   = self._squash_mean(t_mu) * deploy
        pressure = self._squash_mean(p_mu) * deploy
        return torch.cat([deploy, timing, pressure], dim=-1).squeeze(0).numpy()


# ══════════════════════════════════════════════════════════════════════════
# Twin Critic
# ══════════════════════════════════════════════════════════════════════════

class TwinCritic(nn.Module):
    """Two independent Q-networks (clipped double-Q)."""
    def __init__(self, state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM, hidden: int = 256):
        super().__init__()
        def _mlp():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ══════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Circular numpy buffer. RAM ≈ 39 floats × 4B × cap = 15 MB @ 100K."""
    def __init__(self, capacity: int = 100_000,
                 state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        self._s   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self._a   = np.zeros((capacity, action_dim), dtype=np.float32)
        self._r   = np.zeros(capacity,               dtype=np.float32)
        self._ns  = np.zeros((capacity, state_dim),  dtype=np.float32)
        self._d   = np.zeros(capacity,               dtype=np.float32)
        self._ptr  = 0
        self._size = 0
        self._cap  = capacity

    def push(self, state, action, reward, next_state, done):
        self._s[self._ptr]  = state
        self._a[self._ptr]  = action
        self._r[self._ptr]  = reward
        self._ns[self._ptr] = next_state
        self._d[self._ptr]  = float(done)
        self._ptr  = (self._ptr + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self._size, batch_size)
        return dict(
            state      = torch.FloatTensor(self._s[idx]),
            action     = torch.FloatTensor(self._a[idx]),
            reward     = torch.FloatTensor(self._r[idx]),
            next_state = torch.FloatTensor(self._ns[idx]),
            done       = torch.FloatTensor(self._d[idx]),
        )

    def __len__(self) -> int:
        return self._size


# ══════════════════════════════════════════════════════════════════════════
# SAC Agent
# ══════════════════════════════════════════════════════════════════════════

class SACAgent:
    def __init__(
        self,
        state_dim:        int   = STATE_DIM,
        action_dim:       int   = ACTION_DIM,
        lr:               float = 3e-4,
        gamma:            float = 0.99,
        tau:              float = 0.005,
        buffer_size:      int   = 100_000,
        batch_size:       int   = 256,
        warmup_steps:     int   = 1_000,
        update_every:     int   = 1,
        init_temperature: float = 0.2,
        target_entropy:   float = -15.0,
        hidden:           int   = 256,
    ):
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup_steps = warmup_steps
        self.update_every = update_every
        self._total_steps = 0
        self.target_entropy = target_entropy

        self.actor         = SACMixedActor(state_dim, hidden)
        self.critic        = TwinCritic(state_dim, action_dim, hidden)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.log_alpha = torch.tensor(np.log(init_temperature),
                                      dtype=torch.float32, requires_grad=True)

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt  = optim.Adam([self.log_alpha],          lr=lr)

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ── 환경 인터페이스 ────────────────────────────────────────────────
    def select_action(self, state: np.ndarray):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample(s)
        return action.squeeze(0).numpy(), log_prob.item()

    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.get_deterministic_action(state)

    def step(self, state, action, reward, next_state, done) -> dict:
        """각 env step 후 호출. 버퍼에 푸시하고 업데이트 여부 결정."""
        if np.isfinite(reward):
            self.buffer.push(state, action, reward, next_state, done)
        self._total_steps += 1

        if (self._total_steps < self.warmup_steps
                or len(self.buffer) < self.batch_size
                or self._total_steps % self.update_every != 0):
            return {}
        return self.update(self.buffer.sample(self.batch_size))

    # ── 핵심 업데이트 ──────────────────────────────────────────────────
    def update(self, batch: dict) -> dict:
        s, a, r, ns, d = (batch["state"], batch["action"], batch["reward"],
                          batch["next_state"], batch["done"])

        # ── Critic ──
        with torch.no_grad():
            na, nlp = self.actor.sample(ns)
            q1_t, q2_t = self.critic_target(ns, na)
            target_q = r + self.gamma * (1.0 - d) * (
                torch.min(q1_t, q2_t) - self.alpha * nlp
            )

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        # ── Actor ──
        sa, lp = self.actor.sample_for_actor(s)
        actor_loss = (self.alpha.detach() * lp - self.critic.q_min(s, sa)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # ── Alpha (자동 온도 조절) ──
        # alpha 발산/소멸 방지: log_alpha를 [-5, 2] 로 소프트 클리핑
        alpha_loss = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-5.0, 2.0)

        self._soft_update()

        return dict(
            critic_loss = critic_loss.item(),
            actor_loss  = actor_loss.item(),
            alpha_loss  = alpha_loss.item(),
            alpha       = self.alpha.item(),
        )

    def _soft_update(self):
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ── 저장 / 불러오기 ────────────────────────────────────────────────
    def save(self, path: str):
        torch.save(dict(
            actor      = self.actor.state_dict(),
            critic     = self.critic.state_dict(),
            log_alpha  = self.log_alpha.item(),
        ), path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        with torch.no_grad():
            self.log_alpha.fill_(ckpt["log_alpha"])
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False
