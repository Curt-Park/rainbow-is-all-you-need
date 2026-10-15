"""Exercise notebook agents against real CartPole episode boundaries.

Run with: python -m unittest test_n_step_truncation.py
The notebook class cells are loaded without executing the 20,000-frame UI cells.
"""

import ast
import math
import random
import unittest
import warnings
from collections import deque
from pathlib import Path

import gymnasium as gym
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

NOTEBOOKS = ("07_n_step_learning.py", "08_rainbow.py", "09_rainbow_iqn.py")


def load_agent(filename):
    namespace = dict(
        math=math,
        random=random,
        warnings=warnings,
        deque=deque,
        gym=gym,
        mo=mo,
        plt=plt,
        np=np,
        torch=torch,
        nn=nn,
        F=F,
        optim=optim,
        clip_grad_norm_=clip_grad_norm_,
    )
    tree = ast.parse(Path(__file__).with_name(filename).read_text(encoding="utf-8"))
    for cell in tree.body:
        if isinstance(cell, ast.FunctionDef):
            definitions = [
                node
                for node in cell.body
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
            ]
            exec(
                compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec"), namespace
            )
    return namespace["DQNAgent"]


class BalancePolicy(nn.Module):
    """Fixed policy to reach the notebook's 200-step time limit without training."""

    def forward(self, state, *args):
        direction = state[..., 2] + state[..., 3]
        return torch.stack((-direction, direction), dim=-1)


class TruncationTests(unittest.TestCase):
    def make_agent(self, filename, n_step=3):
        env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="rgb_array")
        kwargs = dict(env=env, memory_size=1000, batch_size=500, target_update=100, seed=777)
        if filename.startswith("07"):
            kwargs.update(epsilon_decay=0, max_epsilon=0, min_epsilon=0)
        agent = load_agent(filename)(**kwargs, n_step=n_step)
        agent.dqn = BalancePolicy().to(agent.device)
        return agent

    def test_time_limit_does_not_join_episodes(self):
        for filename in NOTEBOOKS:
            with self.subTest(notebook=filename):
                agent = self.make_agent(filename)
                # train uses select_action -> step -> reset, exactly as the notebook does.
                agent.train(202, plotting_interval=1000)
                self.assertEqual(len(agent.memory_n), 198)
                self.assertEqual(len(agent.memory), 198)
                self.assertEqual(agent.memory_n.ptr, agent.memory.ptr)
                # The full window ending at truncation still bootstraps.
                self.assertEqual(agent.memory_n.terminated_buf[197], 0)
                self.assertEqual(agent.memory_n.rews_buf[197], np.float32(1 + 0.99 + 0.99**2))
                self.assertEqual(len(agent.memory_n.n_step_buffer), 2)

    def test_next_full_window_keeps_paired_indices(self):
        for filename in NOTEBOOKS:
            with self.subTest(notebook=filename):
                agent = self.make_agent(filename)
                agent.train(203, plotting_interval=1000)
                self.assertEqual(len(agent.memory_n), 199)
                self.assertEqual(len(agent.memory), 199)
                np.testing.assert_array_equal(
                    agent.memory.obs_buf[:199], agent.memory_n.obs_buf[:199]
                )
                np.testing.assert_array_equal(
                    agent.memory.acts_buf[:199], agent.memory_n.acts_buf[:199]
                )
                # Both episodes reset with seed 777 in train().
                np.testing.assert_array_equal(
                    agent.memory_n.obs_buf[0], agent.memory_n.obs_buf[198]
                )

    def test_true_termination_still_cuts_the_return(self):
        for filename in NOTEBOOKS:
            with self.subTest(notebook=filename):
                agent = self.make_agent(filename)
                state, _ = agent.env.reset(seed=777)
                done = False
                while not done:
                    # A valid constant CartPole action reaches physical termination.
                    agent.transition = [state, 0]
                    state, _, done = agent.step(0)
                self.assertLess(agent.env._elapsed_steps, 200)
                final_state = state.copy()
                state, _ = agent.env.reset(seed=777)
                for _ in range(2):
                    agent.transition = [state, 0]
                    state, _, _ = agent.step(0)
                row = agent.memory_n.ptr - 1
                self.assertEqual(agent.memory_n.terminated_buf[row], 1)
                self.assertEqual(agent.memory_n.rews_buf[row], 1)
                np.testing.assert_array_equal(agent.memory_n.next_obs_buf[row], final_state)
                agent.env.close()

    def test_one_step_path_is_unchanged(self):
        for filename in NOTEBOOKS:
            with self.subTest(notebook=filename):
                agent = self.make_agent(filename, n_step=1)
                agent.train(202, plotting_interval=1000)
                self.assertEqual(len(agent.memory), 202)
                self.assertEqual(agent.memory.terminated_buf[199], 0)


if __name__ == "__main__":
    unittest.main()
