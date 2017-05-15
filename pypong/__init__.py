import pygame
import random 
import math
import os

import pypong.entity

def load_image(path):
    surface = pygame.image.load(path)
    surface.convert()
    pygame.surfarray.pixels3d(surface)[:,:,0:1:] = 0
    return surface

def line_line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # Taken from http://paulbourke.net/geometry/lineline2d/
    # Denominator for ua and ub are the same, so store this calculation
    d = float((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    # n_a and n_b are calculated as seperate values for readability
    n_a = float((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3))
    n_b = float((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3))
    # Make sure there is not a division by zero - this also indicates that
    # the lines are parallel.  
    # If n_a and n_b were both equal to zero the lines would be on top of each 
    # other (coincidental).  This check is not done because it is not 
    # necessary for this implementation (the parallel check accounts for this).
    if d == 0:
        return False
    # Calculate the intermediate fractional point that the lines potentially intersect.
    ua = n_a / d
    ub = n_b / d
    # The fractional point will be between 0 and 1 inclusive if the lines
    # intersect.  If the fractional calculation is larger than 1 or smaller
    # than 0 the lines would need to be longer to intersect.
    if ua >= 0. and ua <= 1. and ub >= 0. and ub <= 1.:
        return [x1 + (ua * (x2 - x1)), y1 + (ua * (y2 - y1))]
    return False
    
class Game(object):
    def __init__(self, player_left, player_right, config):
        self.player_left = player_left
        self.player_right = player_right
        self.config = config
        self.background = pygame.Surface(config.SCREEN_SIZE)
        self.sprites = pygame.sprite.OrderedUpdates()
        line = entity.Line(load_image(config.LINE_IMAGE), self.sprites)
        line.rect.topleft = ((config.SCREEN_SIZE[0]-line.rect.width)/2, 0)
        paddle_image = load_image(config.PADDLE_IMAGE)
        self.paddle_left = entity.Paddle(config.PADDLE_VELOCITY, paddle_image, 
                                         config.PADDLE_BOUNDS, self.sprites)
        self.paddle_right = entity.Paddle(config.PADDLE_VELOCITY, paddle_image, 
                                          config.PADDLE_BOUNDS, self.sprites)
        self.paddle_left.rect.topleft = (config.PADDLE_LEFT_POSITION,
                                        (config.SCREEN_SIZE[1]-self.paddle_left.rect.height)/2)
        self.paddle_right.rect.topleft = (config.PADDLE_RIGHT_POSITION,
                                         (config.SCREEN_SIZE[1]-self.paddle_left.rect.height)/2)
        digit_images = [load_image(config.DIGIT_IMAGE % n) for n in range(10)]
        self.score_left = entity.Score(digit_images, self.sprites)
        self.score_left.rect.topleft = config.SCORE_LEFT_POSITION
        self.score_right = entity.Score(digit_images, self.sprites)
        self.score_right.rect.topleft = config.SCORE_RIGHT_POSITION
        ball_image = load_image(config.BALL_IMAGE)
        self.ball = entity.Ball(self.config.BALL_VELOCITY, ball_image, self.sprites)
        self.bounds = pygame.Rect(20, 0, config.SCREEN_SIZE[0]-ball_image.get_width()-20,
                                  config.SCREEN_SIZE[1]-ball_image.get_height())
        if config.SOUND:
            self.sound_missed = pygame.mixer.Sound(config.SOUND_MISSED)
            self.sound_paddle = pygame.mixer.Sound(config.SOUND_PADDLE)
            self.sound_wall = pygame.mixer.Sound(config.SOUND_WALL)
       
	#Added by Philipp Benz
        self.first_state=True
        self.num_rounds=0
        self.epsilon_trigger=True
        self.trigger=True
        self.player_right_hit=False
        self.player_left_hit=False
        self.num_matches=0
        self.rightWonMatches=0
        self.leftWonMatches=0
        self.reset_game(random.random()<0.5)
        self.running = True

        self.rightHitCounter=0

    def play_sound(self, sound):
        if self.config.SOUND:
            sound.play()
        
    def reset_game(self, serveLeft=True):
        y = self.config.SCREEN_SIZE[1] - self.ball.rect.height
        self.ball.position_x = (self.config.SCREEN_SIZE[0]-self.ball.rect.width)/2.0
        self.ball.position_y = y * random.random()
        self.ball.velocity = self.config.BALL_VELOCITY
        a = random.random() * math.pi / 2. - math.pi / 4.
        self.ball.velocity_vec[0] = self.ball.velocity * math.cos(a)
        self.ball.velocity_vec[1] = self.ball.velocity * math.sin(a)
        if random.random() < 0.5:
            self.ball.velocity_vec[1] = -self.ball.velocity_vec[1]
        if serveLeft:
            self.ball.velocity_vec[0] *= -1

        self.first_state=True
        self.epsilon_trigger=True
        self.trigger=True
        self.num_rounds+=1
        if self.num_rounds>self.config.NUM_ROUNDS_PER_MATCH:
            self.num_matches+=1
            if self.score_right.score>self.score_left.score:
               # print('Player Right won this match')
                self.rightWonMatches+=1
            elif self.score_right.score<self.score_left.score:
               # print('Player Left won this match')
                self.leftWonMatches+=1
            #else:
               # print('Draw!\nThis should not happen!!!')
            '''
            print('Total Matches:'+str(self.num_matches) + \
                    ' Right:'+str(self.rightWonMatches) + \
                    ' Left:'+str(self.leftWonMatches))
            print('Right Hits: '+ str(self.rightHitCounter))
            '''
            self.rightHitCounter=0
            self.num_rounds=1
            self.score_right.score=0
            self.score_left.score=0

    def update(self):
        # Store previous ball position for line-line intersect test later
        ball_position_x = self.ball.position_x
        ball_position_y = self.ball.position_y
        # Update sprites and players
        self.sprites.update()
        self.player_left.update(self.paddle_left, self)
        self.player_right.update(self.paddle_right, self)
        # Paddle collision check. Could probably just do a line-line intersect,
        # but I think I prefer having the pixel-pefect result of a 
        # rect-rect intersect test as well.
        if self.ball.rect.x < self.bounds.centerx:
            # Left side bullet-through-paper check on ball and paddle
            if self.ball.velocity_vec[0] < 0:
                intersect_point = line_line_intersect(
                    self.paddle_left.rect.right, self.paddle_left.rect.top,
                    self.paddle_left.rect.right, self.paddle_left.rect.bottom,
                    ball_position_x-self.ball.rect.width/2, 
                    ball_position_y+self.ball.rect.height/2,
                    self.ball.position_x-self.ball.rect.width/2, 
                    self.ball.position_y+self.ball.rect.height/2
                )
                if intersect_point:
                    self.ball.position_y = intersect_point[1]-self.ball.rect.height/2
                if intersect_point or (self.paddle_left.rect.colliderect(self.ball.rect)
                and self.ball.rect.right > self.paddle_left.rect.right):
                    
                    self.ball.position_x = self.paddle_left.rect.right
                    velocity = self.paddle_left.calculate_bounce(min(1,
                                max(0,(self.ball.rect.centery - self.paddle_left.rect.y)/
                                float(self.paddle_left.rect.height))))

                    self.ball.velocity = min(self.config.BALL_VELOCITY_MAX, 
                                        self.ball.velocity * 
                                        self.config.BALL_VELOCITY_BOUNCE_MULTIPLIER)

                    self.ball.velocity_vec[0] = velocity[0] * self.ball.velocity
                    self.ball.velocity_vec[1] = velocity[1] * self.ball.velocity
                    self.player_left.hit()
                    self.player_left_hit=True
                    if self.config.SOUND:
                        self.play_sound(self.sound_paddle)
        else:
            # Right side bullet-through-paper check on ball and paddle.
            if self.ball.velocity_vec[0] > 0:
                intersect_point = line_line_intersect(
                    self.paddle_right.rect.left, self.paddle_right.rect.top,
                    self.paddle_right.rect.left, self.paddle_right.rect.bottom,
                    ball_position_x-self.ball.rect.width/2, 
                    ball_position_y+self.ball.rect.height/2,
                    self.ball.position_x-self.ball.rect.width/2, 
                    self.ball.position_y+self.ball.rect.height/2
                )
                if intersect_point:
                    self.ball.position_y = intersect_point[1]-self.ball.rect.height/2

                if intersect_point or \
                (self.paddle_right.rect.colliderect(self.ball.rect) and \
                self.ball.rect.x < self.paddle_right.rect.x):
                    
                    self.ball.position_x = self.paddle_right.rect.x - self.ball.rect.width
                    
                    velocity = self.paddle_right.calculate_bounce(min(1,
                            max(0,(self.ball.rect.centery - 
                            self.paddle_right.rect.y)/float(self.paddle_right.rect.height))))

                    self.ball.velocity = min(self.config.BALL_VELOCITY_MAX, 
                                        self.ball.velocity * 
                                        self.config.BALL_VELOCITY_BOUNCE_MULTIPLIER)

                    self.ball.velocity_vec[0] = -velocity[0] * self.ball.velocity
                    self.ball.velocity_vec[1] = velocity[1] * self.ball.velocity
                    self.player_right.hit()
                    self.rightHitCounter+=1
                    self.player_right_hit=True
                    if self.config.SOUND:
                        self.play_sound(self.sound_paddle)
        
        # Bounds collision check
        if self.ball.rect.y <= self.bounds.top:
            self.ball.position_y = float(self.bounds.top)
            self.ball.velocity_vec[1] = -self.ball.velocity_vec[1]
            if self.config.SOUND:
                self.play_sound(self.sound_wall)
        
        elif self.ball.rect.y >= self.bounds.bottom:
            self.ball.position_y = float(self.bounds.bottom)
            self.ball.velocity_vec[1] = -self.ball.velocity_vec[1]
            if self.config.SOUND:
                self.play_sound(self.sound_wall)
        # Check the ball is still in play
        if self.ball.rect.x < self.bounds.x:
            self.player_left.lost()
            self.player_right.won()
            self.score_right.score += 1
            self.reset_game(False)    
            if self.config.SOUND:
                self.play_sound(self.sound_missed)
        
        if self.ball.rect.x > self.bounds.right:
            self.player_left.won()
            self.player_right.lost()
            self.score_left.score += 1
            self.reset_game(True)
            if self.config.SOUND:
                self.play_sound(self.sound_missed)

    def draw(self, display_surface):
        self.sprites.clear(display_surface, self.background)
        return self.sprites.draw(display_surface)
