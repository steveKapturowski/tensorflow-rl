# Emulator code inspired by:
# https://github.com/Jabberwockyll/deep_rl_ale/
import numpy as np
from ale_python_interface import ALEInterface
import cv2
#from skimage.transform import resize
#from skimage.color import rgb2gray
import sys
import random
import logging_utils
logger = logging_utils.getLogger('emulator')

import matplotlib.pyplot as plt

IMG_SCALE = 255.0
IMG_SIZE_X = 84
IMG_SIZE_Y = 84
NR_IMAGES = 4
ACTION_REPEAT = 4
MAX_START_WAIT = 30
FRAMES_IN_POOL = 2
BLEND_METHOD = 'max_pool'
RANDOM_PLAY_STEPS = 1000000

class Emulator:
    def __init__(self, rom_path, rom_name, visualize, actor_id, rseed, single_life_episodes = False):
        
        self.ale = ALEInterface()

        self.ale.setInt("random_seed", rseed * (actor_id +1))

        # For fuller control on explicit action repeat (>= ALE 0.5.0) 
        self.ale.setFloat("repeat_action_probability", 0.0)
        
        # Disable frame_skip and color_averaging
        # See: http://is.gd/tYzVpj
        self.ale.setInt("frame_skip", 1)
        self.ale.setBool("color_averaging", False)
        self.ale.loadROM(rom_path + "/" + rom_name + ".bin")
        self.legal_actions = self.ale.getMinimalActionSet()        
        self.screen_width,self.screen_height = self.ale.getScreenDims()
        #self.ale.setBool('display_screen', True)
        
        # Processed historcal frames that will be fed in to the network 
        # (i.e., four 84x84 images)
        self.screen_images_processed = np.zeros((IMG_SIZE_X, IMG_SIZE_Y, 
            NR_IMAGES)) 
        self.rgb_screen = np.zeros((self.screen_height,self.screen_width, 3), dtype=np.uint8)
        self.gray_screen = np.zeros((self.screen_height,self.screen_width,1), dtype=np.uint8)

        self.frame_pool = np.empty((2, self.screen_height, self.screen_width))
        self.current = 0
        self.lives = self.ale.lives()

        self.visualize = visualize
        self.visualize_processed = False
        self.windowname = rom_name + ' ' + str(actor_id)
        if self.visualize:
            logger.debug("Opening emulator window...")
            #from skimage import io
            #io.use_plugin('qt')
            cv2.startWindowThread()
            cv2.namedWindow(self.windowname)
            logger.debug("Emulator window opened")
            
        if self.visualize_processed:
            logger.debug("Opening processed frame window...")
            cv2.startWindowThread()
            logger.debug("Processed frame window opened")
            cv2.namedWindow(self.windowname + "_processed")
            
        self.single_life_episodes = single_life_episodes

    def get_screen_image(self):
        """ Add screen (luminance) to frame pool """
        # [screen_image, screen_image_rgb] = [self.ale.getScreenGrayscale(), 
        #     self.ale.getScreenRGB()]
        self.ale.getScreenGrayscale(self.gray_screen)
        self.ale.getScreenRGB(self.rgb_screen)
        self.frame_pool[self.current] = np.squeeze(self.gray_screen)
        self.current = (self.current + 1) % FRAMES_IN_POOL
        return self.rgb_screen

    def new_game(self):
        """ Restart game """
        self.ale.reset_game()
        self.lives = self.ale.lives()

        if MAX_START_WAIT < 0:
            logger.debug("Cannot time travel yet.")
            sys.exit()
        elif MAX_START_WAIT > 0:
            wait = random.randint(0, MAX_START_WAIT)
        else:
            wait = 0
        for _ in xrange(wait):
            self.ale.act(self.legal_actions[0])

    def process_frame_pool(self):
        """ Preprocess frame pool """
        
        img = None
        if BLEND_METHOD == "max_pool":
            img = np.amax(self.frame_pool, axis=0)
        
        #return resize(img[:210, :], (84, 84))
        img = cv2.resize(img[:210, :], (84, 84), 
            interpolation=cv2.INTER_LINEAR)
        
        img = img.astype(np.float32)
        img *= (1.0/255.0)
        
        return img
        # Reduce height to 210, if not so
        #cropped_img = img[:210, :]
        # Downsample to 110x84
        #down_sampled_img = resize(cropped_img, (84, 84))
        
        # Crop to 84x84 playing area
        #stackable_image = down_sampled_img[:, 26:110]
        #return stackable_image

    def action_repeat(self, a):
        """ Repeat action and grab screen into frame pool """
        reward = 0
        for i in xrange(ACTION_REPEAT):
            reward += self.ale.act(self.legal_actions[a])
            new_screen_image_rgb = self.get_screen_image()
        return reward, new_screen_image_rgb

    def get_reshaped_state(self, state):
        return np.reshape(state, 
            (1, IMG_SIZE_X, IMG_SIZE_Y, NR_IMAGES))
        #return np.reshape(self.screen_images_processed, 
        #    (1, IMG_SIZE_X, IMG_SIZE_Y, NR_IMAGES))

    def get_initial_state(self):
        """ Get the initial state """
        self.new_game()
        for step in xrange(NR_IMAGES):
            reward, new_screen_image_rgb = self.action_repeat(0)
            self.screen_images_processed[:, :, step] = self.process_frame_pool()
            self.show_screen(new_screen_image_rgb)
        if self.is_terminal():
            MAX_START_WAIT -= 1
            return self.get_initial_state()
        return np.copy(self.screen_images_processed) #get_reshaped_state()      

    def next(self, action):
        """ Get the next state, reward, and game over signal """
        reward, new_screen_image_rgb = self.action_repeat(np.argmax(action))
        self.screen_images_processed[:, :, 0:3] = \
            self.screen_images_processed[:, :, 1:4]
        self.screen_images_processed[:, :, 3] = self.process_frame_pool()
        self.show_screen(new_screen_image_rgb)
        terminal = self.is_terminal()
        self.lives = self.ale.lives()
        return np.copy(self.screen_images_processed), reward, terminal #get_reshaped_state(), reward, terminal
    
    def show_screen(self, image):
        """ Show visuals for raw and processed images """
        if self.visualize:
            #io.imshow(image[:210, :], fancy=True)
            cv2.imshow(self.windowname, image[:210, :])
            # plt.imshow(image[:210, :])
            # plt.savefig("test.png")
        if self.visualize_processed:
            #io.imshow(self.screen_images_processed[:, :, 3], fancy=True)
            cv2.imshow(self.windowname + "_processed", self.screen_images_processed[:, :, 3])
            
    def is_terminal(self):
        if self.single_life_episodes:
            return (self.is_over() or (self.lives > self.ale.lives()))
        else:
            return self.is_over()

    def is_over(self):
        return self.ale.game_over()


if __name__ == "__main__":
    emulator = Emulator("../atari_roms", "breakout", True, 0, 1)
    emulator.get_initial_state()
    n_actions = len(emulator.legal_actions)
    one_hot_action = np.zeros(n_actions)
    for _ in xrange(RANDOM_PLAY_STEPS):
        one_hot_action[random.randint(0, n_actions - 1)] = 1
        _, _, over = emulator.next(one_hot_action, False)
        if over:
            break
        one_hot_action.fill(0)