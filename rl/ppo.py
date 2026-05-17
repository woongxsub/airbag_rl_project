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

    def __init__(self, state_dim=12, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.deploy_head = nn.Linear(hidden, 5)       # Bernoulli logit
        self.timing_mean = nn.Linear(hidden, 5)
        self.timing_log_std = nn.Parameter(torch.zeros(5))
        self.pressure_mean = nn.Linear(hidden, 5)
        self.pressure_log_std = nn.Parameter(torch.zeros(5))

    def forward(self, state):
        h = self.shared(state)
        deploy_logit = self.deploy_head(h)
        t_mean = torch.sigmoid(self.timing_mean(h))     # [0,1]
        p_mean = torch.sigmoid(self.pressure_mean(h))   # [0,1]
        return deploy_logit, t_mean, p_mean

    def get_action(self, state):
        deploy_logit, t_mean, p_mean = self.forward(state)

        deploy_dist = Bernoulli(logits=deploy_logit)
        deploy = deploy_dist.sample()

        t_dist = Normal(t_mean, self.timing_log_std.exp())
        p_dist = Normal(p_mean, self.pressure_log_std.exp())
        timing = t_dist.sample().clamp(0.0, 1.0)
        pressure = p_dist.sample().clamp(0.0, 1.0)

        # deploy=0이면 timing·pressure gradient 마스킹
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
    def __init__(self, state_dim=12, hidden=128):
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
    def __init__(self, state_dim=12, lr=3e-4, gamma=0.99, clip=0.2, epochs=10):
        self.actor = MultiHeadActor(state_dim)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs

    def select_action(self, state: np.ndarray):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_t)
        return action.squeeze(0).numpy(), log_prob.item()

    def update(self, transitions: list):
        states = torch.FloatTensor([t["state"] for t in transitions])
        actions = torch.FloatTensor([t["action"] for t in transitions])
        old_log_probs = torch.FloatTensor([t["log_prob"] for t in transitions])
        rewards = torch.FloatTensor([t["reward"] for t in transitions])

        # 단순 Monte Carlo return (에피소드 단위)
        returns = rewards

        for _ in range(self.epochs):
            deploy_logit, t_mean, p_mean = self.actor(states)
            deploy = actions[:, :5]
            timing = actions[:, 5:10]
            pressure = actions[:, 10:15]

            deploy_dist = Bernoulli(logits=deploy_logit)
            t_dist = Normal(t_mean, self.actor.timing_log_std.exp())
            p_dist = Normal(p_mean, self.actor.pressure_log_std.exp())

            log_probs = (
                deploy_dist.log_prob(deploy).sum(-1)
                + (t_dist.log_prob(timing) * deploy).sum(-1)
                + (p_dist.log_prob(pressure) * deploy).sum(-1)
            )

            values = self.critic(states)
            advantages = returns - values.detach()

            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip, 1 + self.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
