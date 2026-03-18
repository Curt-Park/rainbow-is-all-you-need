# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gymnasium[classic-control]",
#     "marimo",
#     "matplotlib",
#     "moviepy",
#     "numpy",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 09. Rainbow IQN

    This tutorial builds a **Rainbow IQN** agent — replacing the Categorical DQN (C51) distributional component in Rainbow with **Implicit Quantile Networks (IQN)**.

    > "Creating a Rainbow-IQN agent could yield even greater improvements on Atari-57."
    > — Dabney et al., *Implicit Quantile Networks for Distributional Reinforcement Learning*, 2018

    The IQN paper explicitly suggests this combination, and implementations like [medipixel/rl_algorithms](https://github.com/medipixel/rl_algorithms) have demonstrated that Rainbow-IQN reaches perfect score (+21) on Atari Pong within ~100 episodes — dramatically faster than standard Rainbow (~500K vs ~10M steps).

    ---

    ## Background: Distributional RL Progression

    **C51** (tutorial 06) models the return distribution with **fixed discrete atoms** on a support $[V_{\min}, V_{\max}]$, learning probability masses over those atoms. This works well but requires choosing the support range and number of atoms in advance.

    **QR-DQN** (Dabney et al., 2018, [arXiv:1710.10044](https://arxiv.org/abs/1710.10044)) flips the representation: instead of fixed locations with learned probabilities (C51), it uses **fixed uniform probabilities with learned locations** (quantiles). It uses quantile regression loss to learn the quantile values. However, it still uses a fixed number of quantile fractions (e.g., $\tau = 1/2N, 3/2N, \ldots$).

    ---

    ## IQN — The Key Innovation

    IQN goes further by learning a **continuous quantile function** $F^{-1}(\tau)$ that maps *any* $\tau \in [0, 1]$ to the corresponding quantile value. The network takes $\tau$ as input and outputs quantile values — hence "implicit."

    ### 1. Cosine Basis Embedding

    To condition the network on $\tau$, IQN uses a cosine embedding:

    $$\phi_j(\tau) = \text{ReLU}\!\Big(\sum_{i=1}^{n} w_{ij} \cos(\pi \, i \, \tau) + b_j\Big)$$

    The cosine terms $\cos(\pi \, i \, \tau)$ for $i = 1, \ldots, n$ form a basis that captures different frequency components of the quantile function. This embedding is passed through a learned linear layer to produce a feature vector of the same dimension as the state features.

    ### 2. Hadamard Product

    The quantile features (from cosine embedding) are combined with state features via **element-wise multiplication**. This allows the network to modulate its state representation based on which quantile level it's estimating.

    ### 3. Quantile Huber Loss

    Training uses an asymmetric loss. For quantile $\tau$, overestimation errors (positive) are weighted by $\tau$, underestimation errors (negative) by $(1 - \tau)$. Combined with Huber loss for robustness:

    $$\rho_\tau^\kappa(u) = |\tau - \mathbf{1}(u < 0)| \cdot \frac{\mathcal{L}_\kappa(u)}{\kappa}$$

    where $\mathcal{L}_\kappa$ is the Huber loss with threshold $\kappa$.

    ### 4. Action Selection

    Sample $N$ quantile fractions, compute quantile values for each, average to get $Q(s, a)$, then take $\arg\max$.

    ---

    ## Rainbow IQN — What Changes

    The seven components become:

    1. DQN
    2. Double DQN
    3. Prioritized Experience Replay
    4. Dueling Network
    5. Noisy Network
    6. **IQN** (replacing C51)
    7. N-step Learning

    IQN replaces C51 as the distributional component; all other Rainbow components remain. The key advantage: **no need to specify $V_{\min}$, $V_{\max}$, or atom size** — the network learns the full quantile function implicitly.

    **References:**
    - W. Dabney et al., ["Implicit Quantile Networks for Distributional Reinforcement Learning,"](https://arxiv.org/abs/1806.06923) ICML 2018.
    - W. Dabney et al., ["Distributional Reinforcement Learning with Quantile Regression,"](https://arxiv.org/abs/1710.10044) AAAI 2018. (QR-DQN)
    - M. G. Bellemare et al., ["A Distributional Perspective on Reinforcement Learning,"](https://arxiv.org/abs/1707.06887) ICML 2017. (C51)
    """)
    return


@app.cell
def _():
    import math
    import os
    import random
    import warnings
    from collections import deque

    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.nn.utils import clip_grad_norm_

    return (
        F,
        clip_grad_norm_,
        deque,
        gym,
        math,
        nn,
        np,
        optim,
        os,
        plt,
        random,
        torch,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Replay buffer

    Same as the basic N-step buffer.

    (Please see *01_dqn.py*, *07_n_step_learning.py* for detailed description about the basic (n-step) replay buffer.)
    """)
    return


@app.cell
def _(deque, np):
    class ReplayBuffer:
        """A simple numpy replay buffer."""

        def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            n_step: int = 1,
            gamma: float = 0.99,
        ):
            self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
            self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
            self.acts_buf = np.zeros([size], dtype=np.float32)
            self.rews_buf = np.zeros([size], dtype=np.float32)
            self.terminated_buf = np.zeros(size, dtype=np.float32)
            self.max_size, self.batch_size = size, batch_size
            (
                self.ptr,
                self.size,
            ) = 0, 0

            # for N-step Learning
            self.n_step_buffer = deque(maxlen=n_step)
            self.n_step = n_step
            self.gamma = gamma

        def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            terminated: bool,
        ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
            transition = (obs, act, rew, next_obs, terminated)
            self.n_step_buffer.append(transition)

            # single step transition is not ready
            if len(self.n_step_buffer) < self.n_step:
                return ()

            # make a n-step transition
            rew, next_obs, terminated = self._get_n_step_info(self.n_step_buffer, self.gamma)
            obs, act = self.n_step_buffer[0][:2]

            self.obs_buf[self.ptr] = obs
            self.next_obs_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.terminated_buf[self.ptr] = terminated
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

            return self.n_step_buffer[0]

        def sample_batch(self) -> dict[str, np.ndarray]:
            idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

            return dict(
                obs=self.obs_buf[idxs],
                next_obs=self.next_obs_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                terminated=self.terminated_buf[idxs],
                # for N-step Learning
                indices=idxs,
            )

        def sample_batch_from_idxs(self, idxs: np.ndarray) -> dict[str, np.ndarray]:
            # for N-step Learning
            return dict(
                obs=self.obs_buf[idxs],
                next_obs=self.next_obs_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                terminated=self.terminated_buf[idxs],
            )

        def _get_n_step_info(
            self, n_step_buffer: deque, gamma: float
        ) -> tuple[np.int64, np.ndarray, bool]:
            """Return n step rew, next_obs, and terminated."""
            # info of the last transition
            rew, next_obs, terminated = n_step_buffer[-1][-3:]

            for transition in reversed(list(n_step_buffer)[:-1]):
                r, n_o, d = transition[-3:]

                rew = r + gamma * rew * (1 - d)
                next_obs, terminated = (n_o, d) if d else (next_obs, terminated)

            return rew, next_obs, terminated

        def __len__(self) -> int:
            return self.size

    return (ReplayBuffer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Segment Tree

    Same segment tree implementation used in *03_per.py*. See that notebook for a detailed explanation of the data structure.

    - **SegmentTree**: Base class supporting any associative binary operation with O(log n) updates and range queries.
    - **SumSegmentTree**: Tracks cumulative priorities for proportional sampling.
    - **MinSegmentTree**: Tracks minimum priority for importance-sampling weight normalization.
    """)
    return


@app.cell
def _():
    import operator
    from collections.abc import Callable

    class SegmentTree:
        """Create SegmentTree.

        Taken from OpenAI baselines github repository:
        https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
        """

        def __init__(self, capacity: int, operation: Callable, init_value: float):
            assert capacity > 0 and capacity & (capacity - 1) == 0, (
                "capacity must be positive and a power of 2."
            )
            self.capacity = capacity
            self.tree = [init_value for _ in range(2 * capacity)]
            self.operation = operation

        def _operate_helper(
            self, start: int, end: int, node: int, node_start: int, node_end: int
        ) -> float:
            if start == node_start and end == node_end:
                return self.tree[node]
            mid = (node_start + node_end) // 2
            if end <= mid:
                return self._operate_helper(start, end, 2 * node, node_start, mid)
            elif mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

        def operate(self, start: int = 0, end: int = 0) -> float:
            if end <= 0:
                end += self.capacity
            end -= 1
            return self._operate_helper(start, end, 1, 0, self.capacity - 1)

        def __setitem__(self, idx: int, val: float):
            idx += self.capacity
            self.tree[idx] = val
            idx //= 2
            while idx >= 1:
                self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
                idx //= 2

        def __getitem__(self, idx: int) -> float:
            assert 0 <= idx < self.capacity
            return self.tree[self.capacity + idx]

    class SumSegmentTree(SegmentTree):
        def __init__(self, capacity: int):
            super().__init__(capacity=capacity, operation=operator.add, init_value=0.0)

        def sum(self, start: int = 0, end: int = 0) -> float:
            return super().operate(start, end)

        def retrieve(self, upperbound: float) -> int:
            assert 0 <= upperbound <= self.sum() + 1e-5, f"upperbound: {upperbound}"
            idx = 1
            while idx < self.capacity:
                left = 2 * idx
                right = left + 1
                if self.tree[left] > upperbound:
                    idx = 2 * idx
                else:
                    upperbound -= self.tree[left]
                    idx = right
            return idx - self.capacity

    class MinSegmentTree(SegmentTree):
        def __init__(self, capacity: int):
            super().__init__(capacity=capacity, operation=min, init_value=float("inf"))

        def min(self, start: int = 0, end: int = 0) -> float:
            return super().operate(start, end)

    return MinSegmentTree, SumSegmentTree


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prioritized Replay Buffer

    Combines PER with N-step learning. The `store` method returns a tuple to indicate whether an N-step transition has been completed — only then is it added to the priority buffer.

    (Please see *03_per.py* for detailed description about PER.)
    """)
    return


@app.cell
def _(MinSegmentTree, ReplayBuffer, SumSegmentTree, np, random):
    class PrioritizedReplayBuffer(ReplayBuffer):
        """Prioritized Replay buffer.

        Attributes:
            max_priority (float): max priority
            tree_ptr (int): next index of tree
            alpha (float): alpha parameter for prioritized replay buffer
            sum_tree (SumSegmentTree): sum tree for prior
            min_tree (MinSegmentTree): min tree for min prior to get max weight

        """

        def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            alpha: float = 0.6,
            n_step: int = 1,
            gamma: float = 0.99,
        ):
            """Initialization."""
            assert alpha >= 0

            super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size, n_step, gamma)
            self.max_priority, self.tree_ptr = 1.0, 0
            self.alpha = alpha

            # capacity must be positive and a power of 2.
            tree_capacity = 1
            while tree_capacity < self.max_size:
                tree_capacity *= 2

            self.sum_tree = SumSegmentTree(tree_capacity)
            self.min_tree = MinSegmentTree(tree_capacity)

        def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            terminated: bool,
        ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
            """Store experience and priority."""
            transition = super().store(obs, act, rew, next_obs, terminated)

            if transition:
                self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
                self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
                self.tree_ptr = (self.tree_ptr + 1) % self.max_size

            return transition

        def sample_batch(self, beta: float = 0.4) -> dict[str, np.ndarray]:
            """Sample a batch of experiences."""
            assert len(self) >= self.batch_size
            assert beta > 0

            indices = self._sample_proportional()

            obs = self.obs_buf[indices]
            next_obs = self.next_obs_buf[indices]
            acts = self.acts_buf[indices]
            rews = self.rews_buf[indices]
            terminated = self.terminated_buf[indices]
            weights = np.array([self._calculate_weight(i, beta) for i in indices])

            return dict(
                obs=obs,
                next_obs=next_obs,
                acts=acts,
                rews=rews,
                terminated=terminated,
                weights=weights,
                indices=indices,
            )

        def update_priorities(self, indices: list[int], priorities: np.ndarray):
            """Update priorities of sampled transitions."""
            assert len(indices) == len(priorities)

            for idx, priority in zip(indices, priorities):
                assert priority > 0
                assert 0 <= idx < len(self)

                self.sum_tree[idx] = priority**self.alpha
                self.min_tree[idx] = priority**self.alpha

                self.max_priority = max(self.max_priority, priority)

        def _sample_proportional(self) -> list[int]:
            """Sample indices based on proportions."""
            indices = []
            p_total = self.sum_tree.sum(0, len(self))
            segment = p_total / self.batch_size

            for i in range(self.batch_size):
                a = segment * i
                b = segment * (i + 1)
                upperbound = random.uniform(a, b)
                idx = self.sum_tree.retrieve(upperbound)
                indices.append(idx)

            return indices

        def _calculate_weight(self, idx: int, beta: float):
            """Calculate the weight of the experience at idx."""
            # get max weight
            p_min = self.min_tree.min() / self.sum_tree.sum()
            max_weight = (p_min * len(self)) ** (-beta)

            # calculate weights
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weight = weight / max_weight

            return weight

    return (PrioritizedReplayBuffer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Noisy Layer

    Please see *05_noisy_net.py* for detailed description.

    **References:**

    - https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
    - https://github.com/Kaixhin/Rainbow/blob/master/model.py
    """)
    return


@app.cell
def _(F, math, nn, torch):
    class NoisyLinear(nn.Module):
        """Noisy linear module for NoisyNet.



        Attributes:
            in_features (int): input size of linear module
            out_features (int): output size of linear module
            std_init (float): initial std value
            weight_mu (nn.Parameter): mean value weight parameter
            weight_sigma (nn.Parameter): std value weight parameter
            bias_mu (nn.Parameter): mean value bias parameter
            bias_sigma (nn.Parameter): std value bias parameter

        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
        ):
            """Initialization."""
            super(NoisyLinear, self).__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init

            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
            self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer("bias_epsilon", torch.Tensor(out_features))

            self.reset_parameters()
            self.reset_noise()

        def reset_parameters(self):
            """Reset trainable network parameters (factorized gaussian noise)."""
            mu_range = 1 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

        def reset_noise(self):
            """Make new noise."""
            epsilon_in = self.scale_noise(self.in_features)
            epsilon_out = self.scale_noise(self.out_features)

            # outer product
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward method implementation.

            In eval mode, use only the mean weights (no noise) for
            deterministic action selection, following Google Dopamine.
            """
            if self.training:
                return F.linear(
                    x,
                    self.weight_mu + self.weight_sigma * self.weight_epsilon,
                    self.bias_mu + self.bias_sigma * self.bias_epsilon,
                )
            return F.linear(x, self.weight_mu, self.bias_mu)

        @staticmethod
        def scale_noise(size: int) -> torch.Tensor:
            """Set scale to make noise (factorized gaussian noise)."""
            x = torch.randn(size)

            return x.sign().mul(x.abs().sqrt())

    return (NoisyLinear,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NoisyNet + DuelingNet + IQN

    The three components integrate as follows:

    - **NoisyNet + DuelingNet**: Same as tutorial 08 — NoisyLinear layers in the advantage and value streams.
    - **DuelingNet + IQN**: The dueling architecture applies **per-quantile** — each quantile sample gets its own value + advantage decomposition.
    - **IQN integration**: The cosine embedding + Hadamard product happen between the shared feature layer and the dueling split.

    ```
    state_features = feature_layer(x)                    # (batch, 128)
    cos_embed = cos(pi * i * tau)                        # (batch*n_tau, embed_dim)
    tau_features = relu(cos_linear(cos_embed))           # (batch*n_tau, 128)
    combined = state_features * tau_features             # (batch*n_tau, 128) via Hadamard
    -> advantage_stream -> value_stream -> dueling aggregation -> quantile values
    ```
    """)
    return


@app.cell
def _(F, NoisyLinear, nn, torch):
    class Network(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, quantile_embedding_dim: int = 64):
            """Initialization."""
            super(Network, self).__init__()

            self.out_dim = out_dim
            self.quantile_embedding_dim = quantile_embedding_dim

            # set common feature layer
            self.feature_layer = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
            )

            # cosine embedding layer for IQN
            self.cos_embedding_layer = nn.Linear(quantile_embedding_dim, 128)
            self.register_buffer(
                "i_pi", torch.pi * torch.arange(1, quantile_embedding_dim + 1, dtype=torch.float32)
            )

            # set advantage layer
            self.advantage_hidden_layer = NoisyLinear(128, 128)
            self.advantage_layer = NoisyLinear(128, out_dim)

            # set value layer
            self.value_hidden_layer = NoisyLinear(128, 128)
            self.value_layer = NoisyLinear(128, 1)

        def forward(self, x: torch.Tensor, n_tau_samples: int = 32) -> torch.Tensor:
            """Forward method: sample taus, get quantile values, average for Q-values."""
            if x.dim() == 1:
                x = x.unsqueeze(0)
            batch_size = x.size(0)
            taus = torch.rand(batch_size, n_tau_samples, device=x.device)
            quantile_values = self.quantile_forward(x, taus)
            # Average over quantiles to get Q-values: (batch, n_tau, out_dim) -> (batch, out_dim)
            q = quantile_values.mean(dim=1)
            return q

        def quantile_forward(
            self, x: torch.Tensor, taus: torch.Tensor
        ) -> torch.Tensor:
            """Compute quantile values for given taus.

            Args:
                x: state input, shape (batch_size, in_dim)
                taus: quantile fractions, shape (batch_size, n_tau)

            Returns:
                quantile values, shape (batch_size, n_tau, out_dim)
            """
            batch_size = x.size(0)
            n_tau = taus.size(1)

            # State features: (batch, 128)
            features = self.feature_layer(x)

            # Cosine embedding: (batch, n_tau) -> (batch * n_tau, embed_dim)
            # cos(pi * i * tau) for i = 1, ..., quantile_embedding_dim
            cos_input = taus.unsqueeze(2) * self.i_pi.unsqueeze(0).unsqueeze(0)
            # (batch, n_tau, embed_dim)
            cos_embed = torch.cos(cos_input)
            # (batch * n_tau, 128)
            tau_features = F.relu(self.cos_embedding_layer(cos_embed.view(batch_size * n_tau, -1)))

            # Hadamard product: expand features to (batch * n_tau, 128)
            features_expanded = features.unsqueeze(1).expand(-1, n_tau, -1).reshape(batch_size * n_tau, -1)
            combined = features_expanded * tau_features

            # Dueling streams
            adv_hid = F.relu(self.advantage_hidden_layer(combined))
            val_hid = F.relu(self.value_hidden_layer(combined))

            advantage = self.advantage_layer(adv_hid)  # (batch * n_tau, out_dim)
            value = self.value_layer(val_hid)  # (batch * n_tau, 1)
            q = value + advantage - advantage.mean(dim=1, keepdim=True)

            # Reshape to (batch, n_tau, out_dim)
            q = q.view(batch_size, n_tau, self.out_dim)
            return q

        def reset_noise(self):
            """Reset all noisy layers."""
            self.advantage_hidden_layer.reset_noise()
            self.advantage_layer.reset_noise()
            self.value_hidden_layer.reset_noise()
            self.value_layer.reset_noise()

    return (Network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rainbow IQN Agent

    Here is a summary of DQNAgent class.

    | Method           | Note                                                 |
    | ---              | ---                                                  |
    |select_action     | select an action from the input state.               |
    |step              | take an action and return the response of the env.   |
    |compute_dqn_loss  | return IQN quantile Huber loss.                      |
    |update_model      | update the model by gradient descent.                |
    |target_hard_update| hard update from the local model to the target model.|
    |train             | train the agent during num_frames.                   |
    |test              | test the agent (1 episode).                          |
    |plot              | plot the training progresses.                        |

    #### IQN + Double DQN

    Double DQN decomposes the max operation in the target into action selection and action evaluation. Here, the **online network** selects the best next action (via averaged quantile values), and the **target network** evaluates quantile values for that action.

    ```
            # Double DQN: online net selects action, target net evaluates quantiles
            next_action = self.dqn(next_state, self.n_quantile_samples).argmax(1)
            next_quantile_values = self.dqn_target.quantile_forward(next_state, tau_primes)
            next_quantile_values = next_quantile_values[range(batch_size), :, next_action]
    ```
    """)
    return


@app.cell
def _(
    F,
    Network,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    clip_grad_norm_,
    gym,
    mo,
    np,
    optim,
    plt,
    torch,
    warnings,
):
    class DQNAgent:
        """DQN Agent interacting with environment.

        Attribute:
            env (gym.Env): openAI Gym environment
            memory (PrioritizedReplayBuffer): replay memory to store transitions
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            gamma (float): discount factor
            dqn (Network): model to train and select actions
            dqn_target (Network): target model to update
            optimizer (torch.optim): optimizer for training dqn
            transition (list): transition information including
                               state, action, reward, next_state, done
            n_tau_samples (int): number of quantile samples for online network
            n_tau_prime_samples (int): number of quantile samples for target network
            n_quantile_samples (int): number of quantile samples for action selection
            kappa (float): threshold for Huber loss in quantile regression
            use_n_step (bool): whether to use n_step memory
            n_step (int): step number to calculate n-step td error
            memory_n (ReplayBuffer): n-step replay buffer
        """

        def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            seed: int,
            gamma: float = 0.99,
            # PER parameters
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            # IQN parameters
            n_tau_samples: int = 32,
            n_tau_prime_samples: int = 32,
            n_quantile_samples: int = 32,
            quantile_embedding_dim: int = 64,
            kappa: float = 1.0,
            # N-step Learning
            n_step: int = 3,
        ):
            """Initialization.

            Args:
                env (gym.Env): openAI Gym environment
                memory_size (int): length of memory
                batch_size (int): batch size for sampling
                target_update (int): period for target model's hard update
                seed (int): random seed
                gamma (float): discount factor
                alpha (float): determines how much prioritization is used
                beta (float): determines how much importance sampling is used
                prior_eps (float): guarantees every transition can be sampled
                n_tau_samples (int): number of quantile samples for online net
                n_tau_prime_samples (int): number of quantile samples for target net
                n_quantile_samples (int): number of quantile samples for action selection
                quantile_embedding_dim (int): dimension of cosine embedding
                kappa (float): Huber loss threshold for quantile regression
                n_step (int): step number to calculate n-step td error
            """
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            self.env = env
            self.batch_size = batch_size
            self.target_update = target_update
            self.seed = seed
            self.gamma = gamma
            # NoisyNet: All attributes related to epsilon are removed

            # IQN parameters
            self.n_tau_samples = n_tau_samples
            self.n_tau_prime_samples = n_tau_prime_samples
            self.n_quantile_samples = n_quantile_samples
            self.kappa = kappa

            # device: cpu / gpu
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(self.device)

            # PER
            # memory for 1-step Learning
            self.beta = beta
            self.beta_initial = beta
            self.prior_eps = prior_eps
            self.memory = PrioritizedReplayBuffer(
                obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
            )

            # memory for N-step Learning
            self.use_n_step = True if n_step > 1 else False
            if self.use_n_step:
                self.n_step = n_step
                self.memory_n = ReplayBuffer(
                    obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
                )

            # networks: dqn, dqn_target
            self.dqn = Network(obs_dim, action_dim, quantile_embedding_dim).to(self.device)
            self.dqn_target = Network(obs_dim, action_dim, quantile_embedding_dim).to(self.device)
            self.dqn_target.load_state_dict(self.dqn.state_dict())
            self.dqn_target.eval()

            # optimizer
            self.optimizer = optim.Adam(self.dqn.parameters())

            # transition to store in memory
            self.transition = list()

            # mode: train / test
            self.is_test = False

        def select_action(self, state: np.ndarray) -> np.ndarray:
            """Select an action from the input state."""
            # NoisyNet: no epsilon greedy action selection
            # Disable noise during test for deterministic evaluation
            if self.is_test:
                self.dqn.eval()
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device), self.n_quantile_samples
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

            if not self.is_test:
                self.transition = [state, selected_action]

            return selected_action

        def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float64, bool]:
            """Take an action and return the response of the env."""
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if not self.is_test:
                self.transition += [reward, next_state, terminated]

                # N-step transition
                if self.use_n_step:
                    one_step_transition = self.memory_n.store(*self.transition)
                # 1-step transition
                else:
                    one_step_transition = self.transition

                # add a single step transition
                if one_step_transition:
                    self.memory.store(*one_step_transition)

            return next_state, reward, done

        def update_model(self) -> torch.Tensor:
            """Update the model by gradient descent."""
            # PER needs beta to calculate weights
            samples = self.memory.sample_batch(self.beta)
            weights = torch.FloatTensor(samples["weights"]).to(self.device)
            indices = samples["indices"]

            # 1-step Learning loss
            elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

            # N-step Learning loss
            # we are gonna combine 1-step loss and n-step loss so as to
            # prevent high-variance. The original rainbow employs n-step loss only.
            if self.use_n_step:
                gamma = self.gamma**self.n_step
                samples = self.memory_n.sample_batch_from_idxs(indices)
                elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
                elementwise_loss += elementwise_loss_n_loss

                # PER: importance sampling before average
                loss = torch.mean(elementwise_loss * weights)

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.dqn.parameters(), 10.0)
            self.optimizer.step()

            # PER: update priorities
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)

            # NoisyNet: reset noise
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

            return loss.item()

        def train(self, num_frames: int, plotting_interval: int = 200):
            """Train the agent."""
            self.is_test = False

            state, _ = self.env.reset(seed=self.seed)
            update_cnt = 0
            losses = []
            scores = []
            score = 0

            for frame_idx in range(1, num_frames + 1):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

                # NoisyNet: removed decrease of epsilon

                # PER: increase beta
                fraction = min(frame_idx / num_frames, 1.0)
                self.beta = self.beta_initial + fraction * (1.0 - self.beta_initial)

                # if episode ends
                if done:
                    state, _ = self.env.reset(seed=self.seed)
                    scores.append(score)
                    score = 0

                # if training is ready
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # plotting
                if frame_idx % plotting_interval == 0:
                    self._plot(frame_idx, scores, losses)

            self.env.close()

        def test(self, video_folder: str) -> None:
            """Test the agent."""
            self.is_test = True

            # for recording a video
            naive_env = self.env
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

            state, _ = self.env.reset(seed=self.seed)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            self.env.close()

            # reset
            self.env = naive_env
            self.dqn.train()

            return score

        def _compute_dqn_loss(self, samples: dict[str, np.ndarray], gamma: float) -> torch.Tensor:
            """Return IQN quantile Huber loss."""
            device = self.device
            state = torch.FloatTensor(samples["obs"]).to(device)
            next_state = torch.FloatTensor(samples["next_obs"]).to(device)
            action = torch.LongTensor(samples["acts"]).to(device)
            reward = torch.FloatTensor(samples["rews"]).to(device)
            terminated = torch.FloatTensor(samples["terminated"]).to(device)

            batch_size = state.size(0)

            # Sample taus for online network
            taus = torch.rand(batch_size, self.n_tau_samples, device=device)

            # Current quantile values for taken actions: (batch, n_tau)
            current_quantile_values = self.dqn.quantile_forward(state, taus)
            current_quantile_values = current_quantile_values.gather(
                2, action.unsqueeze(1).unsqueeze(2).expand(-1, self.n_tau_samples, 1)
            ).squeeze(2)  # (batch, n_tau)

            with torch.no_grad():
                # Sample tau_primes for target network
                tau_primes = torch.rand(batch_size, self.n_tau_prime_samples, device=device)

                # Double DQN: online net selects action, target net evaluates quantiles
                next_action = self.dqn(next_state, self.n_quantile_samples).argmax(1)

                # Target quantile values: (batch, n_tau_prime)
                next_quantile_values = self.dqn_target.quantile_forward(next_state, tau_primes)
                next_quantile_values = next_quantile_values.gather(
                    2, next_action.unsqueeze(1).unsqueeze(2).expand(-1, self.n_tau_prime_samples, 1)
                ).squeeze(2)  # (batch, n_tau_prime)

                # Bellman target: r + (1 - terminated) * gamma * Z'
                target_quantile_values = reward.unsqueeze(1) + (
                    1.0 - terminated.unsqueeze(1)
                ) * gamma * next_quantile_values  # (batch, n_tau_prime)

            # Pairwise TD errors: (batch, n_tau, n_tau_prime)
            td_errors = target_quantile_values.unsqueeze(1) - current_quantile_values.unsqueeze(2)

            # Quantile Huber loss
            huber_loss = F.smooth_l1_loss(
                current_quantile_values.unsqueeze(2).expand_as(td_errors),
                target_quantile_values.unsqueeze(1).expand_as(td_errors),
                reduction="none",
            )

            # Asymmetric quantile weighting: |tau - 1(error < 0)|
            quantile_weights = torch.abs(
                taus.unsqueeze(2) - (td_errors < 0).float()
            )

            # Quantile Huber loss: (batch, n_tau, n_tau_prime)
            quantile_huber_loss = (quantile_weights * huber_loss / self.kappa)

            # Average over both tau and tau_prime for per-sample loss
            elementwise_loss = quantile_huber_loss.mean(dim=2).mean(dim=1)

            return elementwise_loss

        def _target_hard_update(self):
            """Hard update: target <- local."""
            self.dqn_target.load_state_dict(self.dqn.state_dict())

        def _plot(
            self,
            frame_idx: int,
            scores: list[float],
            losses: list[float],
        ):
            """Plot the training progresses."""
            plt.close("all")
            plt.figure(figsize=(20, 5))
            plt.subplot(131)
            plt.title("frame %s. score: %s" % (frame_idx, np.mean(scores[-10:])))
            plt.plot(scores)
            plt.subplot(132)
            plt.title("loss")
            plt.plot(losses)
            mo.output.replace(mo.as_html(plt.gcf()))

    return (DQNAgent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Environment

    You can see the [code](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py) and [configurations](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py#L91) of CartPole-v1 from Farama Gymnasium's repository.
    """)
    return


@app.cell
def _(gym):
    # environment
    env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")
    return (env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set random seed
    """)
    return


@app.cell
def _(np, random, torch):
    seed = 777

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    return (seed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize
    """)
    return


@app.cell
def _(DQNAgent, env, seed):
    # parameters
    num_frames = 20000
    memory_size = 5000
    batch_size = 32
    target_update = 100

    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update, seed)
    return agent, num_frames


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train
    """)
    return


@app.cell
def _(agent, num_frames):
    agent.train(num_frames)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test

    Run the trained agent (1 episode).
    """)
    return


@app.cell
def _(agent, mo):
    video_folder = "videos/rainbow_iqn"
    score = agent.test(video_folder=video_folder)
    mo.output.replace(mo.md(f"**Test score: {score}**"))
    return (video_folder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Render
    """)
    return


@app.cell
def _(mo, os, video_folder):
    import glob

    def show_latest_video(video_folder: str):
        list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    latest_file = show_latest_video(video_folder=video_folder)
    mo.output.replace(mo.video(src=open(latest_file, "rb").read()))
    return


if __name__ == "__main__":
    app.run()
