#!/usr/env/bin python
#   
# Game Screen
# 

import pygame 
import sys
import os
import inspect
import pickle
from environment import Environment
from spritesheet import SpriteSheet
from Background import Backgrounds
from King import King
from Babe import Babe
from Level import Levels
from Menu import Menus
from jump_king_ai import JumpKingTrainer

from Start import Start

class JKGame:
	""" Overall class to manga game aspects """
        
	def __init__(self):

		pygame.init()

		self.environment = Environment()

		self.clock = pygame.time.Clock()

		self.fps = int(os.environ.get("fps"))
 
		self.bg_color = (0, 0, 0)

		self.screen = pygame.display.set_mode((int(os.environ.get("screen_width")) * int(os.environ.get("window_scale")), int(os.environ.get("screen_height")) * int(os.environ.get("window_scale"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen = pygame.Surface((int(os.environ.get("screen_width")), int(os.environ.get("screen_height"))), pygame.HWSURFACE|pygame.DOUBLEBUF)#|pygame.SRCALPHA)

		self.game_screen_x = 0

		pygame.display.set_icon(pygame.image.load("images\\sheets\\JumpKingIcon.ico"))

		self.levels = Levels(self.game_screen)

		self.king = King(self.game_screen, self.levels)

		self.babe = Babe(self.game_screen, self.levels)

		self.menus = Menus(self.game_screen, self.levels, self.king)

		self.start = Start(self.game_screen, self.menus)

		pygame.display.set_caption('Jump King At Home XD')

	def run_game(self, mode="play", model_path=None):
		ai_trainer = JumpKingTrainer(self)
		
		if mode == "train":
			# Training mode with 8-jump episodes
			self.fps = 60  # Higher FPS for faster training
			ai_trainer.train()
		elif mode == "run_model" and model_path:
			# Run pre-trained model in 8-jump episodes
			ai_trainer.load_and_run(model_path)
		else:
			# Normal gameplay mode
			while True:
				self.clock.tick(self.fps)
				self._check_events()

				if not os.environ["pause"]:
					self._update_gamestuff()

				self._update_gamescreen()
				self._update_guistuff()
				pygame.display.update()
				self._update_audio()

	def _check_events(self):

		for event in pygame.event.get():

			if event.type == pygame.QUIT:

				self.environment.save()

				self.menus.save()

				sys.exit()

			if event.type == pygame.KEYDOWN:

				self.menus.check_events(event)

				if event.key == pygame.K_c:

					if os.environ["mode"] == "creative":

						os.environ["mode"] = "normal"

					else:

						os.environ["mode"] = "creative"
					
			if event.type == pygame.VIDEORESIZE:

				self._resize_screen(event.w, event.h)

	def _update_gamestuff(self):

		self.levels.update_levels(self.king, self.babe)

	def _update_guistuff(self):

		if self.menus.current_menu:

			self.menus.update()

		if not os.environ["gaming"]:

			self.start.update()

	def _update_gamescreen(self):

		pygame.display.set_caption(f"Jump King At Home XD - {self.clock.get_fps():.2f} FPS")

		self.game_screen.fill(self.bg_color)

		if os.environ["gaming"]:

			self.levels.blit1()

		if os.environ["active"]:

			self.king.blitme()

		if os.environ["gaming"]:

			self.babe.blitme()

		if os.environ["gaming"]:

			self.levels.blit2()

		if os.environ["gaming"]:

			self._shake_screen()

		if not os.environ["gaming"]:

			self.start.blitme()

		self.menus.blitme()

		self.screen.blit(pygame.transform.scale(self.game_screen, self.screen.get_size()), (self.game_screen_x, 0))

	def _resize_screen(self, w, h):

		self.screen = pygame.display.set_mode((w, h), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.SRCALPHA)

	def _shake_screen(self):

		try:

			if self.levels.levels[self.levels.current_level].shake:

				if self.levels.shake_var <= 150:

					self.game_screen_x = 0

				elif self.levels.shake_var // 8 % 2 == 1:

					self.game_screen_x = -1

				elif self.levels.shake_var // 8 % 2 == 0:

					self.game_screen_x = 1

			if self.levels.shake_var > 260:

				self.levels.shake_var = 0

			self.levels.shake_var += 1

		except Exception as e:

			print("SHAKE ERROR: ", e)

	def _update_audio(self):

		for channel in range(pygame.mixer.get_num_channels()):

			if not os.environ["music"]:

				if channel in range(0, 2):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["ambience"]:

				if channel in range(2, 7):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			if not os.environ["sfx"]:

				if channel in range(7, 16):

					pygame.mixer.Channel(channel).set_volume(0)

					continue

			pygame.mixer.Channel(channel).set_volume(float(os.environ.get("volume")))
			
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Jump King Game with AI')
    parser.add_argument('--mode', type=str, default='play', 
                        choices=['play', 'train', 'run_model'],
                        help='Game mode: play, train, or run_model')
    parser.add_argument('--model', type=str, default='AI_Models/jumpking_ai_best.h5',
                        help='Path to the model file for run_model mode')
    
    args = parser.parse_args()
    
    Game = JKGame()
    Game.run_game(mode=args.mode, model_path=args.model)
