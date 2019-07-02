# Somewhere over the Rainbow, from Cartpole to Pong

Step-by-step tutorials from DQN to Rainbow.
Every tutorial contains both of simple theoretical backgrounds and object-oriented implementation. Just pick any topic in which you are interested, and learn and run it!

## Contents

01. [DQN](https://nbviewer.jupyter.org/github/Curt-Park/2nd_dlcat_rainbow/blob/master/01.dqn.ipynb)
02. [DoubleDQN](https://nbviewer.jupyter.org/github/Curt-Park/2nd_dlcat_rainbow/blob/master/02.double_q.ipynb)
03. [PrioritizedExperienceReplay](https://nbviewer.jupyter.org/github/Curt-Park/2nd_dlcat_rainbow/blob/master/03.per.ipynb)
04. [DuelingNet](https://nbviewer.jupyter.org/github/Curt-Park/2nd_dlcat_rainbow/blob/master/04.dueling.ipynb)
05. [NoisyNet](https://nbviewer.jupyter.org/github/Curt-Park/2nd_dlcat_rainbow/blob/master/05.noisy_net.ipynb)
06. [CategoricalDQN](https://nbviewer.jupyter.org/github/Curt-Park/2nd_dlcat_rainbow/blob/master/06.categorical_dqn.ipynb)
07. [N-stepTD](https://nbviewer.jupyter.org/github/Curt-Park/2nd_dlcat_rainbow/blob/master/07.n_step_td.ipynb)
08. [Rainbow](https://nbviewer.jupyter.org/github/Curt-Park/2nd_dlcat_rainbow/blob/master/08.rainbow.ipynb)

## Prerequisites
This repository is tested on [Anaconda](https://www.anaconda.com/distribution/) virtual environment with python 3.6.1+
```
$ conda create -n rainbow_is_all_you_need python=3.6.1
$ conda activate rainbow_is_all_you_need
```

## Installation
First, clone the repository.
```
git clone https://github.com/Curt-Park/rainbow-is-all-you-need.git
cd rainbow-is-all-you-need
```

Secondly, install packages required to execute the code. Just type:
```
make dep
```

## References

00. [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518
(7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
01. [van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv preprint arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)
02. [T. Schaul et al., "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
03. [Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning." arXiv preprint arXiv:1511.06581, 2015.](https://arxiv.org/pdf/1511.06581.pdf)
04. [M. Fortunato et al., "Noisy Networks for Exploration." arXiv preprint arXiv:1706.10295, 2017.](https://arxiv.org/pdf/1706.10295.pdf)
05. [M. G. Bellemare et al., "A Distributional Perspective on Reinforcement Learning." arXiv preprint arXiv:1707.06887, 2017.](https://arxiv.org/pdf/1707.06887.pdf)
06. [M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning." arXiv preprint arXiv:1710.02298, 2017.](https://arxiv.org/pdf/1710.02298.pdf)
07. [T. Hester et al., "Deep Q-learning from Demonstrations." arXiv preprint arXiv:1704.03732, 2017.](https://arxiv.org/pdf/1704.03732.pdf)
