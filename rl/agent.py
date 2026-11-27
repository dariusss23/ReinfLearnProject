
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

OBS_DIM = 210
ACTION_DIM = 242
NUM_AGENTS = 5
STATE_DIM = OBS_DIM + NUM_AGENTS


class PolicyNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=256):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.policy(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden=256):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.value(x)


def build_state(obs: np.ndarray, agent_idx: int) -> np.ndarray:
    one_hot = np.zeros(NUM_AGENTS, dtype=np.float32)
    one_hot[agent_idx] = 1.0
    return np.concatenate([np.asarray(obs, dtype=np.float32), one_hot])


class PPOAgent:
    def __init__(self,
                 state_dim=STATE_DIM,
                 action_dim=ACTION_DIM,
                 hidden=256,
                 gamma=0.99,
                 lam=0.95,
                 clip_eps=0.2,
                 lr=3e-4,
                 train_iters=10,
                 minibatch_size=256,
                 entropy_coef=0.01,
                 max_grad_norm=0.5):

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.train_iters = train_iters
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.config = dict(state_dim=state_dim, action_dim=action_dim, hidden=hidden,
                           gamma=gamma, lam=lam, clip_eps=clip_eps, lr=lr,
                           train_iters=train_iters, minibatch_size=minibatch_size,
                           entropy_coef=entropy_coef, max_grad_norm=max_grad_norm)

        self.policy = PolicyNet(state_dim, action_dim, hidden)
        self.value_fn = ValueNet(state_dim, hidden)

        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.opt_value = optim.Adam(self.value_fn.parameters(), lr=lr)

    @staticmethod
    def _masked_dist(logits, mask):
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def act(self, states, masks, greedy=False):
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        m = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
        single = s.dim() == 1
        if single:
            s, m = s.unsqueeze(0), m.unsqueeze(0)

        dist = self._masked_dist(self.policy(s), m)
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        value = self.value_fn(s).squeeze(-1)

        if single:
            return action.item(), logp.item(), value.item()
        return action.numpy(), logp.numpy(), value.numpy()

    @torch.no_grad()
    def value(self, state):
        s = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        return self.value_fn(s).item()

    def compute_gae(self, rewards, values, next_value, dones):
        values = list(values) + [next_value]
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return (torch.tensor(advantages, dtype=torch.float32),
                torch.tensor(returns, dtype=torch.float32))

    def update(self, batch):
        """batch: dict cu obs, masks, actions, logp_old, advantages, returns (tensori)."""
        obs = batch["obs"]
        masks = batch["masks"]
        actions = batch["actions"]
        logp_old = batch["logp_old"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        dataset_size = len(obs)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}

        for _ in range(self.train_iters):
            idx = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.minibatch_size):
                batch_idx = idx[start:start + self.minibatch_size]

                b_obs = obs[batch_idx]
                b_masks = masks[batch_idx]
                b_actions = actions[batch_idx]
                b_logp_old = logp_old[batch_idx]
                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]

                dist = self._masked_dist(self.policy(b_obs), b_masks)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - b_logp_old)
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

                policy_loss = -torch.min(ratio * b_adv, clipped * b_adv).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy

                value_pred = self.value_fn(b_obs).squeeze(-1)
                value_loss = (b_returns - value_pred).pow(2).mean()

                self.opt_policy.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                self.opt_value.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_fn.parameters(), self.max_grad_norm)
                self.opt_value.step()

                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.item())
                    stats["approx_kl"].append((b_logp_old - new_logp).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > self.clip_eps).float().mean().item())

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def save(self, path, **extra):
        torch.save({
            "config": self.config,
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "opt_policy": self.opt_policy.state_dict(),
            "opt_value": self.opt_value.state_dict(),
            **extra,
        }, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        agent = cls(**ckpt["config"])
        agent.policy.load_state_dict(ckpt["policy"])
        agent.value_fn.load_state_dict(ckpt["value_fn"])
        agent.opt_policy.load_state_dict(ckpt["opt_policy"])
        agent.opt_value.load_state_dict(ckpt["opt_value"])
        return agent, ckpt
