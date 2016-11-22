from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import atari_py
import utils.logger
logger = utils.logger.getLogger('emulator')

import matplotlib.pyplot as plt

IMG_SCALE = 255.0
IMG_SIZE_X = 84
IMG_SIZE_Y = 84
NR_IMAGES = 4


def get_num_actions(rom_path, rom_name):
    #import os
    #print os.path.abspath(atari_py.__file__)
    game_path = atari_py.get_game_path(rom_name)
    ale = atari_py.ALEInterface()
    ale.loadROM(game_path)
    return ale.getMinimalActionSet()

class Emulator:
    def __init__(self, rom_path, rom_name, visualize, actor_id, rseed, single_life_episode = False):
        
        self.ale = atari_py.ALEInterface()

        self.ale.setInt("random_seed", rseed * (actor_id +1))

        # For fuller control on explicit action repeat (>= ALE 0.5.0) 
        self.ale.setFloat("repeat_action_probability", 0.0)
        
        # See: http://is.gd/tYzVpj
        self.ale.setInt("frame_skip", 4)
        #self.ale.setBool("color_averaging", False)
        self.ale.loadROM(atari_py.get_game_path(rom_name))
        self.legal_actions = self.ale.getMinimalActionSet()        
        self.single_life_episode = single_life_episode
        self.initial_lives = self.ale.lives()
        
        # Processed frames that will be fed in to the network 
        # (i.e., four 84x84 images)
        self.processed_imgs = np.zeros((IMG_SIZE_X, IMG_SIZE_Y, 
            NR_IMAGES), dtype=np.uint8) 

        self.screen_width,self.screen_height = self.ale.getScreenDims()
        self.rgb_screen = np.zeros((self.screen_height,self.screen_width, 4), dtype=np.uint8)
        self.gray_screen = np.zeros((self.screen_height,self.screen_width,1), dtype=np.uint8)
        
        self.visualize = visualize
        self.visualize_processed = False
        rendering_imported = False
#         if self.visualize:
#             from gym.envs.classic_control import rendering
#             rendering_imported = True
#             logger.debug("Opening emulator window...")
#             self.viewer = rendering.SimpleImageViewer()
#             self.render()
#             logger.debug("Emulator window opened")
#             
#         if self.visualize_processed:
#             if not rendering_imported:
#                 from gym.envs.classic_control import rendering
#             logger.debug("Opening emulator window...")
#             self.viewer2 = rendering.SimpleImageViewer()
#             self.render()
#             logger.debug("Emulator window opened")



    def render(self):
        if self.visualize:
            self.ale.getScreenRGB(self.rgb_screen)
            #self.viewer.imshow(self.rgb_screen)
            plt.imshow(self.processed_imgs[:,:,3])
            plt.savefig("test.png")
        if self.visualize_processed:
            self.viewer2.imshow(self.processed_imgs[:,:,3])
            
            
    """
        Resets the atari game, clears the state buffer
    """
    def get_initial_state(self):
        self.ale.reset_game()
        
        s_t = np.squeeze(self.get_new_preprocessed_frame())
        for i in xrange(NR_IMAGES):
            self.processed_imgs[:,:,i] = s_t

        self.render()
        
        return np.copy(self.processed_imgs)

    
    def next(self, action):
        
        action_index = np.argmax(action)
        reward = self.ale.act(action_index)
        episode_over = self.is_terminal()
        
        self.processed_imgs[:, :, 0:NR_IMAGES-1] = \
            self.processed_imgs[:, :, 1:NR_IMAGES] 
        
        self.processed_imgs[:, :, NR_IMAGES-1] = self.get_new_preprocessed_frame()

        return np.copy(self.processed_imgs), reward, episode_over
    
    
    """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
    """
    def get_new_preprocessed_frame(self):
        #self.ale.getScreenGrayscale(self.gray_screen)
        self.ale.getScreenRGB(self.rgb_screen) # says rgb but actually bgr
        #return resize(np.squeeze(self.gray_screen), (IMG_SIZE_X, IMG_SIZE_Y))
               
        #return arr[:,:,[2, 1, 0]].copy()
        return resize(rgb2gray(self.rgb_screen[:,:,[2, 1, 0]]), (IMG_SIZE_X, IMG_SIZE_Y))
    


    def is_terminal(self):
        if self.single_life_episode:
            # Declare the episode finished if we lost one life or the game is over 
            return (self.ale.game_over() or (self.initial_lives > self.ale.lives()))
        else:
            return self.ale.game_over()


