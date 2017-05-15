import numpy
import pickle
import pygame
import sys

from pypong.q_network import DeepQLearner
from  pypong.data_set import DataSet


def load_image(path):
    surface = pygame.image.load(path)
    surface.convert()
    pygame.surfarray.pixels3d(surface)[:,:,0:1:] = 0
    return surface

class NeuralAgent(object):
    def __init__(self, config):
        # Set recursionlimit for save dataSet
        sys.setrecursionlimit(2000)       
        
        # Config
        self.config=config
        
        # Ranges of the Field
        self.XRange=config.SCREEN_SIZE[0]
        self.YRange=config.SCREEN_SIZE[1]

        # Rewards
        self.hitReward=0
        self.lostReward=0
        self.winReward=0
        self.addUpReward=0
        
        # Last State
        self.lastState=numpy.zeros((1,self.config.STATE_SIZE), dtype='float32') 
        
        # Actions
        self.action=1
        self.lastAction=1

        # Terminal 
        self.terminal=False
        
        # Counter
        self.count=0
        self.countUpdates=0
        self.epochCount=0
        self.testEpochCount=0
        self.batchCount=0
        self.testUpdateCount=0

        #Epsilon
        self.epsilon=max(config.EPSILON_START, config.EPSILON_MIN)
        if config.EPSILON_DECAY>0:
            self.epsilonRate=(config.EPSILON_START - config.EPSILON_MIN) /  \
                              config.EPSILON_DECAY
        else:
            self.epsilonRate=0.01

        # Initialize TestDataSet
        self.testDataSet=DataSet(config.STATE_SIZE, 
                            2*config.PHI_LENGTH,
                            config.PHI_LENGTH,
                            config.RNG)
        
        # Initialize or Load Network
        if config.LOAD_NET_NUMBER>0:
            self.network=self.loadNetwork(config.LOAD_NET_NUMBER)
            self.dataSet=self.loadDataSet(config.LOAD_NET_NUMBER)
        else:
            self.network=DeepQLearner(config.STATE_SIZE,
                                    config.ACTION_SIZE,
                                    config.PHI_LENGTH,
                                    config.BATCH_SIZE,
                                    config.DISCOUNT,
                                    config.RHO,
                                    config.MOMENTUM,
                                    config.LEARNING_RATE,
                                    config.RMS_EPSILON,
                                    config.RNG,
                                    config.UPDATE_RULE,
                                    config.BATCH_ACCUMULATOR,
                                    config.FREEZE_INTERVAL)
            # Initialize DataSet
            self.dataSet=DataSet(config.STATE_SIZE, 
                                config.REPLAY_MEMORY_SIZE,
                                config.PHI_LENGTH,
                                config.RNG)
        
        
        #Initialize Matrix for loss Average
        self.lossAverages=numpy.empty([0])

        # Open Files
        if self.config.SHOW_TESTING:
            self.openRewardFile()
        if config.TRAINING:
            self.openLearningFile()

    def update(self, paddle, game):
        
        if self.config.TRAINING:
            # Choose every k(FRAME_SKIP) Frames an action
            if self.countUpdates % self.config.FRAME_SKIP==0:
                # Collect the reward
                lastRoundReward=self.observeReward(game, paddle)

                # Get current state
                currentState=self.observeState(paddle, game)
                
                # Store last Sample
                self.dataSet.addSample(self.lastState,
                                    self.lastAction,
                                    lastRoundReward,
                                    self.terminal)
                
                phi=self.dataSet.phi(currentState)
                self.action=self.network.choose_action(phi, self.epsilon)
                paddle.direction=self.action-1
                # When minimum start size is reached, begin training
                if self.countUpdates>self.config.REPLAY_START_SIZE:
                    loss=self.trainNetwork()
                    # count How many trainings had been done
                    self.batchCount+=1
                    # add loss to lossAverages
                    self.lossAverages=numpy.append(self.lossAverages,loss)
                # Some Postprocessing: 
                # The reward comes always after the action,
                # so the current state has to be safed until the 
                # the next round
                self.lastState=currentState
                self.lastAction=self.action
                self.terminal=False
                self.resetRewards()
            else:
                paddle.direction=self.action-1
        
            self.countUpdates+=1

            # After every epochs do the following:
            # update Epsilon
            # save Network
            # calculate the mean loss average
            # store how many Epochs had been made
            if self.countUpdates % self.config.SIZE_EPOCH==0:
                # Number of Epochs
                self.epochCount+=1
              
                # update Learning File
                self.updateLearningFile()
              
                # Update Epsilon
                self.epsilon=max(self.epsilon - self.epsilonRate, 
                                self.config.EPSILON_MIN)        
                print('Epsilon updated to: ', self.epsilon)
                if self.epsilon==self.config.EPSILON_MIN:
                    game.running=False

                # Only after ever n*epoch save the network
                if self.countUpdates % (self.config.SIZE_EPOCH * self.config.SAVE_EPOCHS)==0:
                    # Save Network
                    self.saveNetwork(self.config.LOAD_NET_NUMBER + self.countUpdates) 
                    self.saveDataSet(self.config.LOAD_NET_NUMBER + self.countUpdates)
                #Show testing ever n*epoch
                if (self.countUpdates % (self.config.SIZE_EPOCH * \
                   self.config.SHOW_TESTING_EPOCH)==0) and self.config.SHOW_TESTING:
                    # Show Testing
                    self.config.TRAINING=False
                
                # Reset Loss averages
                self.lossAverages=numpy.empty([0])

        else: 
            # Choose every k(FRAME_SKIP) Frames an action
            if self.testUpdateCount % self.config.FRAME_SKIP==0:
                # Collect the reward
                lastRoundReward=self.observeReward(game, paddle)
                self.addUpReward=self.addUpReward+lastRoundReward
                # Get current state
                currentState=self.observeState(paddle, game)
            
                # Store last Sample
                self.testDataSet.addSample(self.lastState,
                                    self.lastAction,
                                    lastRoundReward,
                                    self.terminal)
                
                phi=self.testDataSet.phi(currentState)
                self.action=self.network.choose_action(phi, self.config.TESTING_EPSILON)
                paddle.direction=self.action-1
                
                # Some Postprocessing: 
                # The reward comes always after the action,
                # so the current state has to be safed until the 
                # the next round
                self.lastState=currentState
                self.lastAction=self.action
                self.terminal=False
                self.resetRewards()
            else:
                paddle.direction=self.action-1
            if self.testUpdateCount >= self.config.SIZE_TEST_EPOCH and \
                self.config.SHOW_TESTING:
                self.testUpdateCount=0
                self.config.TRAINING=True
                self.testEpochCount+=1
                self.updateRewardFile()
                self.addUpReward=0

            self.testUpdateCount+=1

    def hit(self):
        self.hitReward=0.2
        pass

    def lost(self):
        self.lostReward=-0.7
        self.terminal=True
        pass
        
    def won(self):
        self.winReward=0.7
        self.terminal=True
        pass

    def observeState(self, paddle, game):
        '''
        Observes and returns the current State
        '''
        state=numpy.zeros((1,self.config.STATE_SIZE), dtype='float32')
        
        # xVelNorm=game.ball.velocity_vec[0]/numpy.linalg.norm(game.ball.velocity_vec)
        # yVelNorm=game.ball.velocity_vec[1]/numpy.linalg.norm(game.ball.velocity_vec)
        # Map the states to range [-1,1]
        state[0,0]=(game.ball.position_vec[0]-self.XRange/2)/(self.XRange/2)
        state[0,1]=(game.ball.position_vec[1]-self.YRange/2)/(self.YRange/2)
        # state[0,2]=xVelNorm
        # state[0,3]=yVelNorm
        state[0,2]=(paddle.rect[1]-self.YRange/2)/(self.YRange/2)
        return state

    def observeReward(self, game, paddle):
        reward=self.hitReward + self.lostReward + self.winReward
        return reward
    def resetRewards(self):
        self.hitReward=0
        self.lostReward=0
        self.winReward=0
    
    def trainNetwork(self):
        batchStates, batchActions, batchRewards, batchTerminals= \
                    self.dataSet.randomBatch(self.config.BATCH_SIZE)
        '''        
        print('batchStates: ', batchStates)
        print('BatchActions: ', batchActions)
        print('batchRewards: ', batchRewards)
        print('batchTerminals: ', batchTerminals)
        '''
        loss=self.network.train(batchStates, 
                                batchActions, 
                                batchRewards, 
                                batchTerminals)
        return loss
    
    #---------FILE-MANAGEMENT------------#    
    
    def saveNetwork(self, numRounds):
        print('...Writing Network to: \n' + \
                self.config.DATA_FOLDER + '/NetFiles/' + 'Net_' + \
                str(numRounds) + '.pkl')
        netFile=open(self.config.DATA_FOLDER + '/NetFiles/' + 'Net_' + \
                    str(numRounds) + '.pkl','wb')
        pickle.dump(self.network, netFile)
        netFile.close()
        print('Network written successfully.')

    def loadNetwork(self, numRounds):
        #TODO: Add exceptions for files which are nonexistent
        print('...Reading Network from: \n' + \
            self.config.DATA_FOLDER + '/NetFiles/' + 'Net_' + \
            str(numRounds) + '.pkl')
        netFile=open(self.config.DATA_FOLDER + '/NetFiles/' + 'Net_'+ \
                    str(numRounds) + '.pkl', 'rb')
        network=pickle.load(netFile)
        netFile.close()
        print('Network read succesfully.')
        return network
    
    
    def saveDataSet(self, numRounds):
        print('...Writing DataSet to: \n' + \
                self.config.DATA_FOLDER + '/DataSets/' + 'DataSet_' + \
                str(numRounds) + '.pkl')
        dataFile=open(self.config.DATA_FOLDER + '/DataSets/' + 'DataSet_' + \
                    str(numRounds) + '.pkl','wb')
        pickle.dump(self.dataSet, dataFile)
        dataFile.close()
        print('DataSet written successfully.')

    def loadDataSet(self, numRounds):
        #TODO: Add exceptions for files which are nonexistent
        print('...Reading DataSet from: \n' + \
            self.config.DATA_FOLDER + '/DataSets/' + 'DataSet_' + \
            str(numRounds) + '.pkl')
        dataFile=open(self.config.DATA_FOLDER + '/DataSets/' + 'DataSet_'+ \
                    str(numRounds) + '.pkl', 'rb')
        dataSet=pickle.load(dataFile)
        dataFile.close()
        print('DataSet read succesfully.')
        return dataSet
        
    
    def openLearningFile(self):
        self.learningFile=open(self.config.DATA_FOLDER + '/LearningFile/' + \
                            'learning.csv','w')
        self.learningFile.write('mean Loss, Epoch\n')
        self.learningFile.flush()

    def updateLearningFile(self):
        out="{},{}\n".format(numpy.mean(self.lossAverages),
                                        self.epochCount)
        self.learningFile.write(out)
        self.learningFile.flush()

    def openRewardFile(self):
        self.rewardFile=open(self.config.DATA_FOLDER + '/LearningFile/' + \
                            'reward.csv','w')
        self.rewardFile.write('Epoch, Reward\n')
        self.rewardFile.flush()
    
    def updateRewardFile(self):
        out="{},{}\n".format(self.epochCount,
                             self.addUpReward)
        self.rewardFile.write(out)
        self.rewardFile.flush()

#---------- Tests -------------#
    def testDataStorage(self):
        self.count+=1
        if self.count%100==0:
            print('Last Stored Sample')
            print(self.lastState, self.lastAction, 
                  lastRoundReward, self.terminal)
            print('Last Phi from Storage:')
            lastPhi=self.dataSet.lastPhi()
            print(lastPhi)
            print('Most Recent Phi + state')
            MRPhi=self.dataSet.phi(self.observeState(paddle,game))
            print(MRPhi)
            print('Random Batch with size 32')
            S, A, R, T=self.dataSet.randomBatch(32)
            print('S', S)
            print('A', A)
            print('R', R)
            print('T', T)
            print()
