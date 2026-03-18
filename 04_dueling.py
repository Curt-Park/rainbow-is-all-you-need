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
    # 04. Dueling Network

    [Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning." arXiv preprint arXiv:1511.06581, 2015.](https://arxiv.org/pdf/1511.06581.pdf)

    The proposed network architecture, which is named *dueling architecture*, explicitly separates the representation of state values and (state-dependent) action advantages.

    ![fig1](https://user-images.githubusercontent.com/14961526/60322956-c2f0b600-99bb-11e9-9ed4-443bd14bc3b0.png)

    The dueling network automatically produces separate estimates of the state value function and advantage function, without any extra supervision. Intuitively, the dueling architecture can learn which states are (or are not) valuable, without having to learn the effect of each action for each state. This is particularly useful in states where its actions do not affect the environment in any relevant way.

    The dueling architecture represents both the value $V(s)$ and advantage $A(s, a)$ functions with a single deep model whose output combines the two to produce a state-action value $Q(s, a)$. Unlike in advantage updating, the representation and algorithm are decoupled by construction.

    $$A^\pi (s, a) = Q^\pi (s, a) - V^\pi (s).$$

    The value function $V$ measures the how good it is to be in a particular state $s$. The $Q$ function, however, measures the the value of choosing a particular action when in this state. Now, using the definition of advantage, we might be tempted to construct the aggregating module as follows:

    $$Q(s, a; \theta, \alpha, \beta) = V (s; \theta, \beta) + A(s, a; \theta, \alpha),$$

    where $\theta$ denotes the parameters of the convolutional layers, while $\alpha$ and $\beta$ are the parameters of the two streams of fully-connected layers.

    Unfortunately, the above equation is unidentifiable in the sense that given $Q$ we cannot recover $V$ and $A$ uniquely; for example, there are uncountable pairs of $V$ and $A$ that make $Q$ values to zero. To address this issue of identifiability, we can force the advantage function estimator to have zero advantage at the chosen action. That is, we let the last module of the network implement the forward mapping.

    $$
    Q(s, a; \theta, \alpha, \beta) = V (s; \theta, \beta) + \big( A(s, a; \theta, \alpha) - \max_{a' \in |\mathcal{A}|} A(s, a'; \theta, \alpha) \big).
    $$

    This formula guarantees that we can recover the unique $V$ and $A$, but the optimization is not so stable because the advantages have to compensate any change to the optimal action’s advantage. Due to the reason, an alternative module that replaces the max operator with an average is proposed:

    $$
    Q(s, a; \theta, \alpha, \beta) = V (s; \theta, \beta) + \big( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \big).
    $$

    Unlike the max advantage form, in this formula, the advantages only need to change as fast as the mean, so it increases the stability of optimization.
    """)
    return


@app.cell
def _():
    import os

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
        gym,
        nn,
        np,
        optim,
        os,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Replay buffer

    Please see *01_dqn.py* for detailed description.
    """)
    return


@app.cell
def _(np):
    class ReplayBuffer:
        """A simple numpy replay buffer."""

        def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
            self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
            self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
            self.acts_buf = np.zeros([size], dtype=np.float32)
            self.rews_buf = np.zeros([size], dtype=np.float32)
            self.done_buf = np.zeros(size, dtype=np.float32)
            self.max_size, self.batch_size = size, batch_size
            (
                self.ptr,
                self.size,
            ) = 0, 0

        def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
        ):
            self.obs_buf[self.ptr] = obs
            self.next_obs_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

        def sample_batch(self) -> dict[str, np.ndarray]:
            idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
            return dict(
                obs=self.obs_buf[idxs],
                next_obs=self.next_obs_buf[idxs],
                acts=self.acts_buf[idxs],
                rews=self.rews_buf[idxs],
                done=self.done_buf[idxs],
            )

        def __len__(self) -> int:
            return self.size

    return (ReplayBuffer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dueling Network

    Carefully take a look at advantage and value layers separated from feature layer.
    """)
    return


@app.cell
def _(nn, torch):
    class Network(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            """Initialization."""
            super(Network, self).__init__()

            # set common feature layer
            self.feature_layer = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
            )

            # set advantage layer
            self.advantage_layer = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim),
            )

            # set value layer
            self.value_layer = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward method implementation."""
            feature = self.feature_layer(x)

            value = self.value_layer(feature)
            advantage = self.advantage_layer(feature)

            q = value + advantage - advantage.mean(dim=-1, keepdim=True)

            return q

    return (Network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DQN + DuelingNet Agent (w/o Double-DQN & PER)

    Here is a summary of DQNAgent class.

    | Method           | Note                                                 |
    | ---              | ---                                                  |
    |select_action     | select an action from the input state.               |
    |step              | take an action and return the response of the env.   |
    |compute_dqn_loss  | return dqn loss.                                     |
    |update_model      | update the model by gradient descent.                |
    |target_hard_update| hard update from the local model to the target model.|
    |train             | train the agent during num_frames.                   |
    |test              | test the agent (1 episode).                          |
    |plot              | plot the training progresses.                        |


    Aside from the dueling network architecture, the authors suggest to use Double-DQN and Prioritized Experience Replay as extra components for better performance. However, we don't implement them to simplify the tutorial. There is only one diffrence between DQNAgent here and the one from *01_dqn.py* and that is the usage of clip_grad_norm_ to prevent gradient exploding.
    """)
    return


@app.cell
def _(
    F,
    Network,
    ReplayBuffer,
    clip_grad_norm_,
    gym,
    np,
    optim,
    plt,
    torch,
):
    class DQNAgent:
        """DQN Agent interacting with environment.

        Attribute:
            env (gym.Env): openAI Gym environment
            memory (ReplayBuffer): replay memory to store transitions
            batch_size (int): batch size for sampling
            epsilon (float): parameter for epsilon greedy policy
            epsilon_decay (float): step size to decrease epsilon
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            target_update (int): period for target model's hard update
            gamma (float): discount factor
            dqn (Network): model to train and select actions
            dqn_target (Network): target model to update
            optimizer (torch.optim): optimizer for training dqn
            transition (list): transition information including
                               state, action, reward, next_state, done
        """

        def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            seed: int,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
        ):
            """Initialization.

            Args:
                env (gym.Env): openAI Gym environment
                memory_size (int): length of memory
                batch_size (int): batch size for sampling
                target_update (int): period for target model's hard update
                epsilon_decay (float): step size to decrease epsilon
                lr (float): learning rate
                max_epsilon (float): max value of epsilon
                min_epsilon (float): min value of epsilon
                gamma (float): discount factor
            """
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            self.env = env
            self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
            self.batch_size = batch_size
            self.epsilon = max_epsilon
            self.epsilon_decay = epsilon_decay
            self.seed = seed
            self.max_epsilon = max_epsilon
            self.min_epsilon = min_epsilon
            self.target_update = target_update
            self.gamma = gamma

            # device: cpu / gpu
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(self.device)

            # networks: dqn, dqn_target
            self.dqn = Network(obs_dim, action_dim).to(self.device)
            self.dqn_target = Network(obs_dim, action_dim).to(self.device)
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
            # epsilon greedy policy
            if self.epsilon > np.random.random():
                selected_action = self.env.action_space.sample()
            else:
                selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
                selected_action = selected_action.detach().cpu().numpy()

            if not self.is_test:
                self.transition = [state, selected_action]

            return selected_action

        def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float64, bool]:
            """Take an action and return the response of the env."""
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if not self.is_test:
                self.transition += [reward, next_state, done]
                self.memory.store(*self.transition)

            return next_state, reward, done

        def update_model(self) -> torch.Tensor:
            """Update the model by gradient descent."""
            samples = self.memory.sample_batch()

            loss = self._compute_dqn_loss(samples)

            self.optimizer.zero_grad()
            loss.backward()
            # DuelingNet: we clip the gradients to have their norm less than or equal to 10.
            clip_grad_norm_(self.dqn.parameters(), 10.0)
            self.optimizer.step()

            return loss.item()

        def train(self, num_frames: int, plotting_interval: int = 200):
            """Train the agent."""
            self.is_test = False

            state, _ = self.env.reset(seed=self.seed)
            update_cnt = 0
            epsilons = []
            losses = []
            scores = []
            score = 0

            for frame_idx in range(1, num_frames + 1):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

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

                    # linearly decrease epsilon
                    self.epsilon = max(
                        self.min_epsilon,
                        self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                    )
                    epsilons.append(self.epsilon)

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # plotting
                if frame_idx % plotting_interval == 0:
                    self._plot(frame_idx, scores, losses, epsilons)

            self.env.close()

        def test(self, video_folder: str) -> None:
            """Test the agent."""
            self.is_test = True

            # for recording a video
            naive_env = self.env
            self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

            state, _ = self.env.reset(seed=self.seed)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            print("score: ", score)
            self.env.close()

            # reset
            self.env = naive_env

        def _compute_dqn_loss(self, samples: dict[str, np.ndarray]) -> torch.Tensor:
            """Return dqn loss."""
            device = self.device  # for shortening the following lines
            state = torch.FloatTensor(samples["obs"]).to(device)
            next_state = torch.FloatTensor(samples["next_obs"]).to(device)
            action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
            reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
            done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

            # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
            #       = r                       otherwise
            curr_q_value = self.dqn(state).gather(1, action)
            next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
            mask = 1 - done
            target = (reward + self.gamma * next_q_value * mask).to(self.device)

            # calculate dqn loss
            loss = F.smooth_l1_loss(curr_q_value, target)

            return loss

        def _target_hard_update(self):
            """Hard update: target <- local."""
            self.dqn_target.load_state_dict(self.dqn.state_dict())

        def _plot(
            self,
            frame_idx: int,
            scores: list[float],
            losses: list[float],
            epsilons: list[float],
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
            plt.subplot(133)
            plt.title("epsilons")
            plt.plot(epsilons)
            plt.show()

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
def _(np, torch):
    seed = 777

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
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
    num_frames = 10000
    memory_size = 1000
    batch_size = 32
    target_update = 100
    epsilon_decay = 1 / 2000

    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay, seed)
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
def _(agent):
    video_folder = "videos/dueling"
    agent.test(video_folder=video_folder)
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


if __name__ == "__main__":
    app.run()
