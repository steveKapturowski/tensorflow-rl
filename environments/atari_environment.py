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


from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque
import gym


RESIZE_WIDTH = 84
RESIZE_HEIGHT = 84


def get_num_actions(game):
    env = gym.make(game)
    if hasattr(env.action_space, 'n'):
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.num_discrete_space

    if game in ['Pong-v0', 'Breakout-v0']:
        # Gym currently specifies 6 actions for pong
        # and breakout when only 3 are needed. This
        # is a lame workaround.
        num_actions = 3
    return num_actions


def get_input_shape(game):
    env = gym.make(game)
    if len(env.observation_space.shape) == 1:
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
    def __init__(self, game, visualize=False, resized_width=RESIZE_WIDTH, resized_height=RESIZE_HEIGHT, agent_history_length=4, frame_skip=4, single_life_episodes=False):
        self.game = game
        self.env = gym.make(game)
        self.env.frameskip = frame_skip
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.single_life_episodes = single_life_episodes

        if hasattr(self.env.action_space, 'n'):
            num_actions = self.env.action_space.n
        else:
            num_actions = self.env.action_space.num_discrete_space

        self.gym_actions = range(num_actions)
        if self.env.spec.id in ['Pong-v0', 'Breakout-v0']:
            print 'Doing workaround for pong or breakout'
            # Gym returns 6 possible actions for breakout and pong.
            # Only three are used, the rest are no-ops. This just lets us
            # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
            self.gym_actions = [1,2,3]

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, width, height]
        self.state_buffer = deque()
        
        self.visualize = visualize
        
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
        self.state_buffer = deque()

        x_t = self.env.reset()

        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=len(x_t.shape))
        
        self.current_lives = self.get_lives()
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)

        return s_t

    def get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        if len(observation.shape) > 1:
            return resize(rgb2gray(observation), (self.resized_width, self.resized_height))
        else:
            return observation

    def next(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        if self.visualize:
            self.env.render()
        
        action_index = np.argmax(action_index)
        
        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        s_t1 = np.empty(list(x_t1.shape)+[self.agent_history_length])
        for i in range(self.agent_history_length-1):
            s_t1[..., i] = self.state_buffer[i] 
        
        s_t1[..., self.agent_history_length-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        if self.single_life_episodes:
            terminal |= self.get_lives() < self.current_lives
        self.current_lives = self.get_lives()

        return s_t1, r_t, terminal


