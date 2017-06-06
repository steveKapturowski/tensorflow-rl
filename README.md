# Tensorflow-RL

[![Join the chat at https://gitter.im/tensorflow-rl/Lobby](https://badges.gitter.im/tensorflow-rl/Lobby.svg)](https://gitter.im/tensorflow-rl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Tensorflow based implementations of [A3C](https://arxiv.org/abs/1602.01783),
[PGQ](https://arxiv.org/abs/1611.01626),
[TRPO](https://arxiv.org/abs/1502.05477),
[DQN+CTS](https://arxiv.org/abs/1606.01868),
and [CEM](http://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf) 
originally based on https://github.com/traai/async-deep-rl. I extensively refactored most of the code and beyond the new algorithms added several additional options including the a3c-lstm architecture, a fully-connected architecture to allow training on non-image-based gym environments, and support for continuous action spaces.

The code also includes some experimental ideas I'm toying with and I'm planning on adding the following implementations
in the near future:
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/abs/1611.02247)
- [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)
- [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161)\*
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Neural Episodic Control](https://arxiv.org/abs/1703.01988)\*

\**currently in progress*

# Notes
- You can find a number of my evaluations for the A3C, TRPO, and DQN+CTS algorithms at https://gym.openai.com/users/steveKapturowski. As I'm working on lots of refactoring at the moment it's possible I could break things. Please open an issue if you discover any bugs.
- I'm in the process of swapping out most of the multiprocessing code in favour of distributed tensorflow which should simplify a lot of the training code and allow to distribute actor-learner processes across multiple machines.
- I tried to match my PGQ implementation as closely as possible to what they describe in the paper but I've noticed the average episode reward can exhibit a pathological oscillatory behavior or suddenly collapse during training. If someone spots a flaw in my implementation I'd be extremely grateful to get your feedback.
- There's also an implementation of the A3C+ model from [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868) but I've been focusing on improvements to the DQN variant so this hasn't gotten much love

# Running the code
First you'll need to install the cython extensions needed for the hog updates and CTS density model:
```bash
./setup.py install build_ext --inplace
```

To train an a3c agent on Pong run:
```bash
python main.py Pong-v0 --alg_type a3c -n 8
```

To evaluate a trained agent simply add the --test flag:
```bash
python main.py Pong-v0 --alg_type a3c -n 1 --test --restore_checkpoint
```
DQN+CTS after 50M agent steps

![Montezuma's Revenge](/images/montezumas-revenge-3600.gif)

A3C run on Pong-v0 with default parameters and frameskip sampled uniformly over 3-4

<img src="/images/pong-a3c-reward.png" alt="alt text" width="75%" height="75%">

# Requirements
- python 2.7
- tensorflow 1.0
- scikit-image
- Cython
- pyaml
- gym
