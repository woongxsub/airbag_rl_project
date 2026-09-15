import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Bernoulli, Normal


class MultiHeadActor(nn.Module):
    """
    deploy  → Bernoulli (이산, 5개)
    timing  → Normal    (연속, 5개), deploy=0이면 gradient 마스킹
    pressure→ Normal    (연속, 5개), deploy=0이면 gradient 마스킹
    """

    def __init__(self, state_dim=11, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.deploy_head = nn.Linear(hidden, 5)
        self.timing_mean = nn.Linear(hidden, 5)
        self.timing_log_std = nn.Parameter(torch.zeros(5))
        self.pressure_mean = nn.Linear(hidden, 5)
        self.pressure_log_std = nn.Parameter(torch.zeros(5))

    def forward(self, state):
        h = self.shared(state)
        deploy_logit = self.deploy_head(h)
        t_mean = torch.sigmoid(self.timing_mean(h))
        p_mean = torch.sigmoid(self.pressure_mean(h))
        return deploy_logit, t_mean, p_mean

    def get_action(self, state):
        deploy_logit, t_mean, p_mean = self.forward(state)

        deploy_dist = Bernoulli(logits=deploy_logit)
        deploy = deploy_dist.sample()

        t_dist = Normal(t_mean, self.timing_log_std.exp())
        p_dist = Normal(p_mean, self.pressure_log_std.exp())
        timing = t_dist.sample().clamp(0.0, 1.0)
        pressure = p_dist.sample().clamp(0.0, 1.0)

        timing = timing * deploy
        pressure = pressure * deploy

        action = torch.cat([deploy, timing, pressure], dim=-1)

        log_prob = (
            deploy_dist.log_prob(deploy).sum(-1)
            + (t_dist.log_prob(timing) * deploy).sum(-1)
            + (p_dist.log_prob(pressure) * deploy).sum(-1)
        )
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim=11, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        state_dim: int = 11,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip: float = 0.2,
        epochs: int = 10,
        lam: float = 0.95,         # GAE λ
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.actor  = MultiHeadActor(state_dim)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.gamma        = gamma
        self.clip         = clip
        self.epochs       = epochs
        self.lam          = lam
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

    def select_action(self, state: np.ndarray):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_t)
        return action.squeeze(0).numpy(), log_prob.item()

    def get_deterministic_action(self, state: np.ndarray) -> np.ndarray:
        """분포 평균값으로 결정론적 action 반환 (평가용)."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            deploy_logit, t_mean, p_mean = self.actor(state_t)
            deploy = (torch.sigmoid(deploy_logit) > 0.5).float()
            action = torch.cat([deploy, t_mean * deploy, p_mean * deploy], dim=-1)
        return action.squeeze(0).numpy()

    def update(self, transitions: list, debug: bool = False):
        transitions = [t for t in transitions if np.isfinite(t["reward"])]
        if not transitions:
            return {}

        states      = torch.FloatTensor(np.array([t["state"]      for t in transitions]))
        next_states = torch.FloatTensor(np.array([t["next_state"] for t in transitions]))
        actions     = torch.FloatTensor(np.array([t["action"]     for t in transitions]))
        old_log_probs = torch.FloatTensor([t["log_prob"] for t in transitions])
        rewards     = torch.FloatTensor([t["reward"] for t in transitions])
        dones       = torch.FloatTensor([float(t["done"]) for t in transitions])

        if debug:
            print(f"[DEBUG-reward] RAW   : min={rewards.min().item():.3e}"
                  f"  max={rewards.max().item():.3e}"
                  f"  mean={rewards.mean().item():.3e}"
                  f"  std={rewards.std().item():.3e}"
                  f"  nan={torch.isnan(rewards).any().item()}"
                  f"  n={len(transitions)}")

        # ── 보상 정규화 (학습용 신호만 변환; raw reward/info 로깅값은 환경에서 불변) ──
        # raw reward 스케일이 수십억에 달해 critic_loss 폭발 방지.
        # 정규화 범위 epsilon은 advantage 정규화(1e-8)와 동일하게 맞춤.
        r_mean = rewards.mean()
        r_std  = rewards.std() + 1e-8
        rewards = (rewards - r_mean) / r_std

        if debug:
            print(f"[DEBUG-reward] NORM  : min={rewards.min().item():.3e}"
                  f"  max={rewards.max().item():.3e}"
                  f"  mean={rewards.mean().item():.3e}"
                  f"  std={rewards.std().item():.3e}")

        # GAE(λ) advantage 계산
        with torch.no_grad():
            values      = self.critic(states)
            next_values = self.critic(next_states)

        gae        = 0.0
        advantages = torch.zeros(len(transitions))
        for i in reversed(range(len(transitions))):
            delta = rewards[i] + self.gamma * next_values[i] * (1.0 - dones[i]) - values[i]
            gae   = delta + self.gamma * self.lam * (1.0 - dones[i]) * gae
            advantages[i] = gae

        returns = advantages + values.detach()

        if debug:
            ret_nan = torch.isnan(returns).any().item()
            ret_inf = torch.isinf(returns).any().item()
            print(f"[DEBUG-update] returns: min={returns.min().item():.3e}"
                  f"  max={returns.max().item():.3e}"
                  f"  nan={ret_nan}  inf={ret_inf}")

        # 어드밴티지 정규화
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss_val = critic_loss_val = 0.0
        params = list(self.actor.parameters()) + list(self.critic.parameters())

        for epoch_i in range(self.epochs):
            deploy_logit, t_mean, p_mean = self.actor(states)
            deploy   = actions[:, :5]
            timing   = actions[:, 5:10]
            pressure = actions[:, 10:15]

            deploy_dist = Bernoulli(logits=deploy_logit)
            t_dist      = Normal(t_mean, self.actor.timing_log_std.exp())
            p_dist      = Normal(p_mean, self.actor.pressure_log_std.exp())

            log_probs = (
                deploy_dist.log_prob(deploy).sum(-1)
                + (t_dist.log_prob(timing)   * deploy).sum(-1)
                + (p_dist.log_prob(pressure) * deploy).sum(-1)
            )

            # 엔트로피 보너스: 탐색 장려
            entropy = (
                deploy_dist.entropy().sum(-1)
                + (t_dist.entropy() * deploy).sum(-1)
                + (p_dist.entropy() * deploy).sum(-1)
            )

            values_pred = self.critic(states)

            ratio  = (log_probs - old_log_probs).exp()
            surr1  = ratio * advantages
            surr2  = ratio.clamp(1 - self.clip, 1 + self.clip) * advantages
            actor_loss  = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()
            critic_loss = (returns - values_pred).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            if debug and epoch_i == 0:
                print(f"[DEBUG-update] epoch=0  actor_loss={actor_loss.item():.4f}"
                      f"  critic_loss={critic_loss.item():.4e}"
                      f"  loss={loss.item():.4e}"
                      f"  loss_finite={torch.isfinite(loss).item()}")

            if not torch.isfinite(loss):
                if debug:
                    print(f"[DEBUG-update] epoch={epoch_i}  loss not finite → skip")
                continue

            self.optimizer.zero_grad()
            loss.backward()

            # ── 클리핑 전 grad norm (계측만, 실제 클리핑 없음) ──────────────
            norm_before = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))

            if debug and epoch_i == 0:
                print(f"[DEBUG-grad]  epoch=0 BEFORE clip:"
                      f"  norm={norm_before.item():.4e}"
                      f"  is_nan={torch.isnan(norm_before).item()}"
                      f"  is_inf={torch.isinf(norm_before).item()}")

            # ── 실제 클리핑 적용 ─────────────────────────────────────────────
            norm_after = torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)

            if debug and epoch_i == 0:
                print(f"[DEBUG-grad]  epoch=0 AFTER  clip (target={self.max_grad_norm}):"
                      f"  norm={norm_after.item():.4e}"
                      f"  is_nan={torch.isnan(norm_after).item()}"
                      f"  is_inf={torch.isinf(norm_after).item()}")

            self.optimizer.step()

            # ── 파라미터 NaN 오염 여부 ───────────────────────────────────────
            if debug and epoch_i == 0:
                actor_nan  = any(torch.isnan(p).any().item() for p in self.actor.parameters())
                critic_nan = any(torch.isnan(p).any().item() for p in self.critic.parameters())
                print(f"[DEBUG-param] epoch=0 AFTER step:"
                      f"  actor_nan={actor_nan}  critic_nan={critic_nan}")

            actor_loss_val  = actor_loss.item()
            critic_loss_val = critic_loss.item()

        return {"actor_loss": actor_loss_val, "critic_loss": critic_loss_val}

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
