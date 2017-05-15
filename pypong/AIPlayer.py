
class AIQlearning_NN(object):
    def __init__(self, input_state, configuration):
       
        self.config=configuration

        self.input_state = input_state
        
        paddle_image = load_image(configuration['paddle_image'])
        paddle_rect = paddle_image.get_rect()
        ball_image = load_image(configuration['ball_image'])
        self.ball_rect = ball_image.get_rect()
        
        # Parameters for Learning
        self.reward=0
        self.gamma=0.9
        self.alpha=0.1
        
        # Rewards
        self.lostReward=0
        self.hitReward=0
        self.lastDistReward=0
        self.lastDirReward=0

        #Action
        self.lastAction=0
        
        # Ball Bounds
        self.ball_bound_left=0
        self.ball_bound_right=configuration['screen_size'][0]-self.ball_rect[2]
        self.ball_range_horizontal = self.ball_bound_right - self.ball_bound_left
        self.ball_bound_top = 0 
        self.ball_bound_bottom = configuration['screen_size'][1]-self.ball_rect[3]
        self.ball_range_vertical = self.ball_bound_bottom - self.ball_bound_top
        
        # Paddle Bounds
        self.paddle_bound_top = configuration['paddle_bounds'][0]
        self.paddle_bound_bottom = configuration['paddle_bounds'][1]-paddle_rect[3]
        self.paddle_range_vertical = self.paddle_bound_bottom - self.paddle_bound_top

        # States of previous Round
        self.lastPaddlePos = 0
        self.lastBallPositionVec = list([0,0])
        self.lastBallVelocityVecNorm = list([0,0])

        self.paddleVelocity = configuration['paddle_velocity']
        
        # Epsilon from insert    
        self.epsilon=configuration['epsilon']
        try:
            self.epsilon=float(self.epsilon)
        except:
            print('Epsilon is no number!\nEpsilon = 1')
            self.epsilon=1

        if self.epsilon>1:
            print('Epsilon = 1')
            self.epsilon=1
        elif self.epsilon<0:
            print('Epsilon = 0')
            self.epsilon=0

        self.endPositionTrigger=True
        self.predictedEndPosition=[0,0]
        self.predictedEndVelocity=[0,0]
        
        """
        One state is represented as:
        (BallPosx, BallPosy, BallVelNormx, BallVelNromy, PaddlePosY ,
        action,  reward, 
        NextBallPosy, NextBallPosY, NextBallVelNormx, NextBallVelNormy, NextPaddlePosy)
        
        Actions are 
         1: Down
         0: Stay
        -1: Up
        """ 
        # Initialize the ReplayMemory
        self.ReplayMemory=theano.shared(
                value=numpy.zeros(
                    (configuration['replayMemorySize'],12),
                    dtype=theano.config.floatX
                ),
                name='ReplayMemory',
                borrow=True
        )

        self.replayMemoryIndex=0
        self.replayMemoryFull=False

    def update(self, paddle, game):
        """
        if self.endPositionTrigger==True and game.ball.velocity_vec[0]>0:
            self.predictedEndPosition, self.predictedEndVelocity=self.predictEnd(game.ball.position_vec, game.ball.velocity_vec, paddle.rect[0] - paddle.rect[2])
            self.endPositionTrigger=False
        """
        
        #nextBallPositionVec, nextBallVelocityVec =self.predictNextState(game.ball.position_vec, game.ball.velocity_vec) 
        #This is the reward we got from the last action
        reward=self.lostReward+self.hitReward
        ballVelocityVecNorm=game.ball.velocity_vec/numpy.linalg.norm(game.ball.velocity_vec)
        paddle.direction=1
        transition=[self.lastBallPositionVec[0],self.lastBallPositionVec[1],
                    self.lastBallVelocityVecNorm[0],self.lastBallVelocityVecNorm[1],
                    self.lastPaddlePos,
                    self.lastAction, reward,
                    game.ball.position_vec[0],game.ball.position_vec[1],
                    ballVelocityVecNorm[0],ballVelocityVecNorm[1],
                    paddle.rect[1]
                    ]
        self.storeTransition(transition)

        # Safe this state for the next round, to apply the rewad
        self.lastPaddlePos =paddle.rect[1]
        self.lastBallPositionVec = list(game.ball.position_vec)
        self.lastBallVelocityVecNorm = list(game.ball.velocity_vec)/numpy.linalg.norm(list(game.ball.velocity_vec))
        self.lastAction=paddle.direction 

        # Reset Values
        self.hitReward=0
        self.lostReward=0
       
    def hit(self):
        print('Hit')
        self.hitReward=10

    def lost(self):
        self.lostReward=-10
        print('Lost')

    def won(self):
        print('Won')
        pass
    
    def getDistanceBallPaddleY(self, ballPositionVec, paddleRect):
        # Distance from Ball to Paddle in Y
        distance=abs(ballPositionVec[1]-(paddleRect[1]+paddleRect[3]/2))
        return distance/1000  
       
    def getDistanceReward(self, ballPositionVec, paddleRect):
        numRewards=50
        maxDistance=self.paddle_range_vertical
        divisor=maxDistance/(numRewards+1)
        distance=abs(ballPositionVec[1]-(paddleRect[1]+paddleRect[3]/2))
        distReward=-round(distance/divisor,2)
        return distReward

    def getDirectionReward(self, ballPositionVec, paddle, ballRect, endBallPositionY):
        dirReward=0
        distance=abs(paddle.rect[1]+paddle.rect[3]/2-endBallPositionY)
        if ballPositionVec[0]+ballRect[2]<=paddle.rect[0]+8:
            if distance<=paddle.rect[3]/2+3:
                dirReward=20.-round(distance/100,2)
            else:
                dirReward=-round(distance/100,2)
        else:
            dirReward=-5
        return dirReward
       
    def predictNextState(self, ballPositionVec, ballVelocityVec):
        # Prediction of the next state
        predictionBallPositionVec = list(ballPositionVec)
        predictionBallVelocityVec = list(ballVelocityVec)

        predictionBallPositionVec[0] += predictionBallVelocityVec[0]
        predictionBallPositionVec[1] += predictionBallVelocityVec[1]
        
        if ballPositionVec[1] <= self.ball_bound_top:
            predictionBallVelocityVec[1] = -predictionBallVelocityVec[1]
            predictionBallPositionVec[1] = float(self.ball_bound_top) + predictionBallVelocityVec[1]
        elif ballPositionVec[1] >= self.ball_bound_bottom:
            predictionBallVelocityVec[1] = -predictionBallVelocityVec[1]
            predictionBallPositionVec[1] = float(self.ball_bound_bottom) + predictionBallVelocityVec[1]
        return predictionBallPositionVec, predictionBallVelocityVec   

    def predictEnd(self, ballPositionVec, ballVelocityVec, endBallPositionX):
        currentPositionVec=list(ballPositionVec)
        currentVelocityVec=list(ballVelocityVec)
        nextPositionVec=[0,0]
        nextVelocityVec=[0,0]
        while(currentPositionVec[0]<=endBallPositionX):
            nextPositionVec, nextVelocityVec=self.predictNextState(currentPositionVec, currentVelocityVec)
            currentPositionVec=list(nextPositionVec)
            currentVelocityVec=list(nextVelocityVec)
        return nextPositionVec, nextVelocityVec

    def storeTransition(self, transition):
        if self.replayMemoryIndex==self.config['replayMemorySize']:
            self.replayMemoryFull=True
            print('Replay Memory full!')
            self.replayMemoryIndex=0
        self.ReplayMemory[self.replayMemoryIndex]=transition
        self.replayMemoryIndex+=1
