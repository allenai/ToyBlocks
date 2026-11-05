import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class BaseRLAgent(nn.Module): 
    critic: nn.Module
    actor_mean: nn.Module
    actor_logstd: nn.Parameter
    def __init__(self, n_obs, n_act, priv_dim=0, device=None):
        super().__init__()
        self.n_obs = n_obs
        self.n_act = n_act
        # Privileged tail width for asymmetric actor-critic. The flat obs is laid out as
        # [actor_obs | privileged(priv_dim)]; the critic sees the full obs while the actor strips the tail.
        # priv_dim == 0 => symmetric (actor and critic share the same input), backward compatible.
        self.priv_dim = priv_dim
        self.device = device

    def _actor_obs(self, obs):
        """Strip the privileged tail so the actor never sees domain-randomization params."""
        if self.priv_dim > 0:
            return obs[:, : obs.shape[1] - self.priv_dim]
        return obs

    def get_deterministic_action(self, obs):
        return self.actor_mean(self._actor_obs(obs))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(self._actor_obs(obs))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

class MLPAgent(BaseRLAgent):
    def __init__(self, n_obs, n_act, priv_dim=0, device=None):
        super().__init__(n_obs, n_act, priv_dim=priv_dim, device=device)
        n_actor = n_obs - priv_dim  # actor never sees the privileged tail
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_actor, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))


class BigMLPAgent(BaseRLAgent):
    def __init__(self, n_obs, n_act, priv_dim=0, device=None):
        super().__init__(n_obs, n_act, priv_dim=priv_dim, device=device)
        n_actor = n_obs - priv_dim  # actor never sees the privileged tail
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 512, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_actor, 512, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(512, n_act, device=device), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))


class AttentionMLPAgent(BaseRLAgent):
    """
    PPO agent with a small self-attention front-end for permutation-invariant
    processing of the K nearest distractor observations.

    The distractor corner features occupy a contiguous slice of the flat obs
    starting at `dist_start_idx`. Everything outside that slice is treated as
    non-permuted context passed straight to the MLP.

    Expected flat obs layout (push_stacked):
        robot_joints(...) | distractors(K_obs*24) | object_pos(3) | ...
    → dist_start_idx = size of robot obs group ("agent" key in raw obs dict)

    Args:
        n_obs: total flat observation dimension.
        n_act: action dimension.
        K_obs: number of distractor slots (must match env cfg num_obs_distractors).
        dist_start_idx: index in the flat obs where distractor features begin.
        dist_token_dim: features per distractor slot (8 corners × 3 xyz = 24).
        embed_dim: per-token attention embedding dimension.
        num_heads: number of attention heads.
        device: torch device.
    """
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        K_obs: int,
        dist_start_idx: int,
        dist_token_dim: int = 24,
        embed_dim: int = 64,
        num_heads: int = 1,
        priv_dim: int = 0,
        device=None,
    ):
        super().__init__(n_obs, n_act, priv_dim=priv_dim, device=device)
        self.K_obs = K_obs
        self.dist_token_dim = dist_token_dim
        self.dist_start_idx = dist_start_idx
        self.dist_end_idx = dist_start_idx + K_obs * dist_token_dim
        # Context = everything outside the distractor slice (includes the privileged tail, if any).
        # The privileged params sit at the very end of the flat obs (appended last), and distractors are
        # an interior slice [i:j], so the priv tail lands at the end of ctx (= end of obs[:, j:]).
        n_ctx = n_obs - K_obs * dist_token_dim

        # Per-token encoder: project each 24-d distractor token → embed_dim
        self.token_encoder = nn.Sequential(
            layer_init(nn.Linear(dist_token_dim, embed_dim, device=device)),
            nn.Tanh(),
        )

        # Small self-attention over K_obs tokens
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, device=device
        )

        # MLP input: context + mean-pooled distractor aggregate. Critic sees full context; actor drops priv.
        mlp_in = n_ctx + embed_dim
        mlp_in_actor = mlp_in - priv_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(mlp_in, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(mlp_in_actor, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract distractor slice, run attention, rejoin with context.

        Returns ``(enc_actor, enc_critic)``. The critic encoding keeps the full context (including the
        privileged tail); the actor encoding drops the trailing ``priv_dim`` context features. When
        ``priv_dim == 0`` the two are identical.
        """
        i, j = self.dist_start_idx, self.dist_end_idx
        tokens = obs[:, i:j].reshape(-1, self.K_obs, self.dist_token_dim)  # (N, K_obs, 24)
        ctx = torch.cat([obs[:, :i], obs[:, j:]], dim=-1)                   # (N, n_ctx)

        # Inactive distractor slots are zeroed out in the obs. Exclude them as
        # keys/values so they don't pollute active-slot attention outputs.
        active = tokens.abs().sum(dim=-1) > 0                               # (N, K_obs) bool
        key_padding_mask = ~active                                           # True = ignore slot
        # If every slot is masked (push_stacked allows 0 active distractors),
        # softmax over all-(-inf) → NaN. Allow full attention; zeroed agg handles it.
        all_inactive = ~active.any(dim=1)                                    # (N,)
        key_padding_mask[all_inactive] = False

        tok_emb = self.token_encoder(tokens)                               # (N, K_obs, embed_dim)
        attn_out, _ = self.attn(tok_emb, tok_emb, tok_emb,
                                key_padding_mask=key_padding_mask)         # (N, K_obs, embed_dim)

        # Mean pool only over active slots so inactive outputs don't dilute the aggregate.
        active_f = active.float().unsqueeze(-1)                            # (N, K_obs, 1)
        agg = (attn_out * active_f).sum(dim=1) / active_f.sum(dim=1).clamp(min=1)  # (N, embed_dim)

        enc_critic = torch.cat([ctx, agg], dim=-1)                         # (N, n_ctx + embed_dim)
        if self.priv_dim > 0:
            # Drop the privileged tail (sits at the end of ctx) before concatenating the aggregate.
            ctx_actor = ctx[:, : ctx.shape[1] - self.priv_dim]
            enc_actor = torch.cat([ctx_actor, agg], dim=-1)
        else:
            enc_actor = enc_critic
        return enc_actor, enc_critic

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        enc_actor, _ = self._encode(obs)
        return self.actor_mean(enc_actor)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, enc_critic = self._encode(x)
        return self.critic(enc_critic)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        enc_actor, enc_critic = self._encode(obs)
        action_mean = self.actor_mean(enc_actor)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(enc_critic)


_AGENTS = {
    "mlp": MLPAgent,
    "big_mlp": BigMLPAgent,
    "attention": AttentionMLPAgent,
}
