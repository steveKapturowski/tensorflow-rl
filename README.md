# What is in this repo?

[![Join the chat at https://gitter.im/traai/async-deep-rl](https://badges.gitter.im/traai/async-deep-rl.svg)](https://gitter.im/traai/async-deep-rl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A [Tensorflow](https://www.tensorflow.org/)-based implementation of all algorithms presented in [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783).

This implementation uses processes instead of threads to achieve real concurrency. Each process has a local replica of the network(s) used, implemented in Tensorflow, and runs its own Tensorflow session. In addition, a copy of the network parameters are kept in a shared memory space. At runtime, each process uses its own local network(s) to choose actions and compute gradients (with Tensorflow). The shared network parameters are updated periodically in an asynchronous manner, by applying the grads obtained from Tensorflow into the shared memory space. 

All algorithms have been implemented and all of them converge, although the 1-step Q-learning and Sarsa ones show a medium-low learning rate (more testing is needed). 

Both [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) and [Open AI GYM](https://gym.openai.com/) are supported for the environments.


(1) Go to the algorithms folder 
(`<some path to this repo>/async-deep-rl/algorithms`) and choose which 
algorithm to run via the configuration options in `main.py`.

(2) If you want to run the algorithms using [Open AI GYM](https://gym.openai.com/) with 16 processes and visualize the games, e.g.:
```shell
$ python main.py BeamRider-v0 --env GYM -n 16 -v 1 
```
