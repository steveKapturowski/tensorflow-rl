"""
Code based on: https://github.com/coreylynch/async-rl/blob/master/atari_environment.py
"""

"""
The MIT License (MIT)

Copyright (c) 2016 Corey Lynch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import gym

from gym.spaces import Box, Discrete
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque


RESIZE_WIDTH = 84
RESIZE_HEIGHT = 84


def get_actions(game_or_env):
    if isinstance(game_or_env, str):
        env = gym.make(game_or_env)
    else:
        env = game_or_env

    if isinstance(env.action_space, Discrete):
        num_actions = env.action_space.n
    elif isinstance(env.action_space, Box):
        num_actions = np.prod(env.action_space.shape)
    else:
        raise Exception('Unsupported Action Space \'{}\''.format(
            type(env.action_space).__name__))

    indices = range(num_actions)
    if env.spec.id in ['Pong-v0', 'Breakout-v0']:
        # Gym currently specifies 6 actions for pong and breakout when only 3 are needed
        # TODO: patch the environments instead
        num_actions = 3
        indices = [1 ,2, 3]

    return num_actions, env.action_space, indices


def get_input_shape(game):
    env = gym.make(game)
    if isinstance(env.observation_space, Discrete):
        return [env.observation_space.n]
    elif len(env.observation_space.shape) == 1:
        return list(env.observation_space.shape)
    else:
        return [RESIZE_WIDTH, RESIZE_HEIGHT]


class AtariEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer 
    of size agent_history_length from which environment state
    is constructed.
    """
    def __init__(self, game, seed, visualize=False, use_rgb=False, resized_width=RESIZE_WIDTH,
                 resized_height=RESIZE_HEIGHT, agent_history_length=4, frame_skip=4,
                 max_episode_steps=None, single_life_episodes=False):
        self.game = game
        self.env = gym.make(game)
        self.env.seed(seed)
        self.env.frameskip = frame_skip
        if max_episode_steps:
            self.env.spec.max_episode_steps = max_episode_steps

        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.single_life_episodes = single_life_episodes
        self.visualize = visualize
        self.use_rgb = use_rgb

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, width, height]
        self.state_buffer = deque(maxlen=self.agent_history_length-1)
        self.gym_actions = get_actions(self.env)[2]

        
    def get_lives(self):
        if hasattr(self.env.env, 'ale'):
            return self.env.env.ale.lives()
        else:
            return 0


    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer.clear()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)

        if self.use_rgb:
            s_t = x_t
        else:
            s_t = np.stack([x_t]*self.agent_history_length, axis=len(x_t.shape))
        
        self.current_lives = self.get_lives()
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)

        return s_t

    def get_preprocessed_frame(self, observation):
        if isinstance(self.env.observation_space, Discrete):
            expanded_obs = np.zeros(self.env.observation_space.n, dtype=np.float32)
            expanded_obs[observation] = 1
            return expanded_obs
        elif len(observation.shape) > 1:
            if not self.use_rgb:
                observation = rgb2gray(observation)
            return resize(observation, (self.resized_width, self.resized_height))
        else:
            return observation

    def get_state(self, frame):
        if self.use_rgb:
            state = frame
        else:
            state = np.empty(list(frame.shape)+[self.agent_history_length])
            for i in range(self.agent_history_length-1):
                state[..., i] = self.state_buffer[i] 
            state[..., self.agent_history_length-1] = frame

        return state

    def next(self, action):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        if self.visualize:
            self.env.render()
        
        if isinstance(self.env.action_space, Discrete):
            action_index = np.argmax(action)
            action = self.gym_actions[action_index]

        frame, reward, terminal, info = self.env.step(action)

        frame = self.get_preprocessed_frame(frame)
        state = self.get_state(frame)

        self.state_buffer.append(frame)

        if self.single_life_episodes:
            terminal |= self.get_lives() < self.current_lives
        self.current_lives = self.get_lives()

        return state, reward, terminal


