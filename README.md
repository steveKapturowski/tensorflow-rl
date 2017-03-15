# async-deep-rl

Tensorflow based implementations of [A3C](https://arxiv.org/abs/1602.01783) and
[PGQ](https://arxiv.org/abs/1611.01626) originally based on https://github.com/traai/async-deep-rl.
The [Cross-Entropy Method](http://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf) is implemented
for use as a baseline on simple environments like Cart-Pole. 

I did some heavy refactoring and added some additional options including the a3c-lstm model, a fully-connected 
architecture to allow training on non-image-based gym environments, and support for the AdaMax optimizer.

The code also includes some experimental ideas I'm toying with and I'm planning on adding the following implementations
in the near future:
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868)
- [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)

I've tested the implementations based on the A3C paper pretty extensively and some of my agent evaluations can be
found at https://gym.openai.com/users/steveKapturowski. They *should* work but I can't guarantee I won't accidentally
break something as I'm planning on doing a lot more refactoring.

I tried to match my PGQ implementation as closely as possible to what they describe in the paper but I've noticed the
average episode reward consitently exhibits a pathological oscillatory behavior during training and the entropy gets
quite small far quicker than it should. If someone spots a flaw in my implementation I'd be extremely grateful to get 
your feedback.
