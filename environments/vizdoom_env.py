# wrapper for vizdoom envronment "https://github.com/mwydmuch/ViZDoom"

import numpy as np
from vizdoom import *
from skimage.transform import resize
from skimage.color import rgb2gray
import itertools as it
from os import path

RESIZE_H = 120
RESIZE_W = 160
REPEAT = 12

class VizDoomEnv(object):
  def __init__(self, cfg_path, game_name):
    self.env = DoomGame()
    self.env.load_config(path.join(cfg_path, game_name)+'.cfg')
    self.height = RESIZE_H
    self.width = RESIZE_W
    self.frame_repeat = REPEAT
    self.action_size = self.env.get_available_buttons_size()
    self.actions = [list(a) for a in it.product([0, 1], repeat=self.frame_repeat)]
    self.env.init()

  def get_actions(self):
    return self.action_size, self.actions

  def get_input_shape(self):
    return [self.height, self.width]

  def get_initial_state(self):
    self.env.new_episode()
    screen = self.env.get_state().screen_buffer
    
    return resize(screen, (self.height, self.width), mode='constant')

  def get_preprocess_frame(self, observation):
    if not self.use_rgb:
      observation = rgb2gray(observation)
    observation = resize(observation, (self.height, self.width), mode='constant')

    return observation

  def set_visible(self, visible=True):
    self.env.set_window_visible(visible)

  def next(self, action):
    prev_ammo, prev_health, prev_kill = self.env.get_state().game_variables
    a = action.nonzero()[0][0]
    reward = self.env.make_action(self.actions[a], self.frame_repeat)
    terminal = self.env.is_episode_finished()
    if not terminal:
      state = self.env.get_state()
      screen_buffer = resize(state.screen_buffer, (self.height, self.width), mode='constant')
      curr_ammo, curr_health, curr_kill = state.game_variables
      reward = reward + (curr_ammo - prev_ammo) + (curr_health - prev_health) + 20*(curr_kill - prev_kill)
    else:
      screen_buffer = None

    return screen_buffer, reward, terminal
    