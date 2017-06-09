# wrapper for vizdoom envronment "https://github.com/mwydmuch/ViZDoom"

import numpy as np
from vizdoom import *
from skimage.transform import resize
from skimage.color import rgb2gray
import itertools as it

RESIZE_H = 160
RESIZE_W = 120
REPEAT = 12

class VizDoomEnv(object):
  def __init__(self, args):
    self.game = DoomGame()
    self.game.load_config(args.doom_cfg)
    self.height = RESIZE_H
    self.width = RESIZE_W
    self.frame_repeat = REPEAT

  def get_actions(self):
    self.action_size = self.game.get_available_buttons_size()
    self.actions = [list(a) for a in it.product([0, 1], repeat=n)]

    return self.action_size, self.actions

  def get_input_shape(self):
    return [self.height, self.width]

  def get_initial_state(self):
    self.game.init()
    screen = self.game.get_state().screen_buffer
    
    return resize(screen, (self.height, self.width))

  def next(self, action):
    prev_health, prev_ammo, prev_kill = self.game.get_state().game_variables
    reward = self.game.make_action(action, self.frame_repeat)
    terminal = self.game.is_episode_finished()
    if not terminal:
      curr_health, curr_ammo, curr_kill = 
    