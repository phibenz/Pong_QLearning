import pygame 
import pypong 
import sys
import os

from pypong.player import BasicAIPlayer, KeyboardPlayer, MousePlayer
from pypong.NeuralAgent import NeuralAgent

def launch(config):
    
    if config.SOUND:
        pygame.mixer.pre_init(22050, -16, 2, 1024)
    
    # Initialize pygame
    pygame.init()
    display_surface = pygame.display.set_mode(config.SCREEN_SIZE)
    output_surface = display_surface.copy().convert_alpha()
    output_surface.fill((0,0,0))
    #~ debug_surface = output_surface.copy()
    #~ debug_surface.fill((0,0,0,0))
    debug_surface = None
    clock = pygame.time.Clock()
    input_state = {'key': None, 'mouse': None}
    
    # Prepare game
    player_left = BasicAIPlayer()
    player_right = NeuralAgent(config)
    #player_left = KeyboardPlayer(input_state, pygame.K_w, pygame.K_s)
    
    game = pypong.Game(player_left, player_right, config)
    
    # Main game loop
    timestamp = 1
    frame_number=0
    while game.running:
        
        if config.TRAINING or config.SHOW_TESTING:
            clock.tick(10000)
        else:
            clock.tick(60)
        
        now = pygame.time.get_ticks()
        if timestamp > 0 and timestamp < now:
            timestamp = now + 5000
            # print(clock.get_fps())
        # TODO: ONLY for Mouse or Keyboard Player
        input_state['key'] = pygame.key.get_pressed()
        #input_state['mouse'] = pygame.mouse.get_pos()
        game.update()
        
        if not config.TRAINING and not config.SHOW_TESTING:
            game.draw(output_surface)
            #~ pygame.surfarray.pixels_alpha(output_surface)[:,::2] = 12
            display_surface.blit(output_surface, (0,0))
            if debug_surface:
                display_surface.blit(debug_surface, (0,0))        
            pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                game.running = False

    pygame.display.quit()
    # close Files
    #TODO no hardcode for File closing
    if config.SHOW_TESTING:
        player_right.rewardFile.close()
    if config.TRAINING:
        player_right.learningFile.close()
        
if __name__ == '__main__': run()
