import launcher
import numpy, os

class Configuration:
    
    #----------------------
    # Environment
    #---------------------
    SYSTEM='Linux'
    FOLDER=os.getcwd()
    DATA_FOLDER= os.getcwd()

    #----------------------
    # Pygame Specific
    #---------------------
    SCREEN_SIZE=(686,488)
    PADDLE_IMAGE= 'assets/paddle.png'
    PADDLE_LEFT_POSITION= 84.
    PADDLE_RIGHT_POSITION= 594.
    PADDLE_VELOCITY= 6.
    PADDLE_BOUNDS= (0, 488) # This sets the upper and lower 
                            # paddle boundary.The original game 
                            # didn't allow the paddle to touch the edge, 
    LINE_IMAGE= 'assets/dividing-line.png'
    BALL_IMAGE= 'assets/ball.png'
    BALL_VELOCITY= 4.
    BALL_VELOCITY_BOUNCE_MULTIPLIER= 1.105
    BALL_VELOCITY_MAX=  32.
    SCORE_LEFT_POSITION= (141, 30)
    SCORE_RIGHT_POSITION= (473, 30)
    DIGIT_IMAGE= 'assets/digit_%i.png'
    SOUND_MISSED= 'assets/missed-ball.wav'
    SOUND_PADDLE= 'assets/bounce-paddle.wav'
    SOUND_WALL= 'assets/bounce-wall.wav'
    TRAINING=False
    SOUND=True
    NUM_ROUNDS_PER_MATCH=9
    SAVE_EPOCHS=50
    
    SHOW_TESTING=False
    if not TRAINING:
        SHOW_TESTING=False
    SIZE_TEST_EPOCH=5000
    TESTING_EPSILON=0.01
    SHOW_TESTING_EPOCH=100

    #------------------------
    # Agent/Network parameters:
    #------------------------
    EPSILON_START= 1.
    EPSILON_MIN= 0.1
    EPSILON_DECAY=401
    REPLAY_MEMORY_SIZE= 10000000
    RNG= numpy.random.RandomState()
    PHI_LENGTH=4
    FRAME_SKIP=4
    STATE_SIZE=3
    ACTION_SIZE=3
    BATCH_SIZE=32
    DISCOUNT=0.99
    RHO=0.95 #RMS_DECAY
    MOMENTUM=-1
    LEARNING_RATE=0.00025
    RMS_EPSILON=0.01
    UPDATE_RULE='deepmind_rmsprop'
    BATCH_ACCUMULATOR='sum'
    LOAD_NET_NUMBER=100000000 #50000000 #0
    SIZE_EPOCH=250000
    REPLAY_START_SIZE=SIZE_EPOCH/2
    FREEZE_INTERVAL=10000


    
if __name__=='__main__':

    launcher.launch(Configuration)

    # TODO: Write Test method to check parameters
