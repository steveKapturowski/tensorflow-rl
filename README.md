# Tensorflow-RL

Tensorflow based implementations of [A3C](https://arxiv.org/abs/1602.01783),
[PGQ](https://arxiv.org/abs/1611.01626), 
[TRPO](https://arxiv.org/abs/1502.05477), and
[CEM](http://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf)
originally based on https://github.com/traai/async-deep-rl. I did some heavy refactoring and added several
additional options including the a3c-lstm model, a fully-connected architecture to allow training on
non-image-based gym environments, and support for the AdaMax optimizer.

There's also an implementation of the A3C+ model from [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868) but I'm still in the process of verifying that it can at least roughly reproduce the results of the paper on Montezuma's Revenge.

The code also includes some experimental ideas I'm toying with and I'm planning on adding the following implementations
in the near future:
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/abs/1611.02247)
- [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)
- [Neural Episodic Control](https://arxiv.org/abs/1703.01988)

I've tested the implementations based on the A3C paper pretty extensively and some of my agent evaluations can be
found at https://gym.openai.com/users/steveKapturowski. They *should* work but I can't guarantee I won't accidentally
break something as I'm planning on doing a lot more refactoring.

I tried to match my PGQ implementation as closely as possible to what they describe in the paper but I've noticed the
average episode reward consitently exhibits a pathological oscillatory behavior during training and the entropy gets
quite small far quicker than it should. If someone spots a flaw in my implementation I'd be extremely grateful to get 
your feedback.

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
python main.py Pong-v0 --alg_type a3c -n 1 --test
```

# Requirements
- python 2.7
- tensorflow 1.0
- scikit-image
- Cython
- gym
