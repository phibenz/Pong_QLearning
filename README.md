# Py-Pong Q-Network
Q-Learning approach for the Atari classic game Pong. The learning process is realized by means of a
neural net

## Dependencies
[Python3](https://www.python.org/download/releases/3.0/)  
[SciPy](https://www.scipy.org/)  
[Numpy](http://www.numpy.org/)  
[PyGame](http://pygame.org/)  
[Theano](http://www.deeplearning.net/software/theano/)
[Lasagne](http://lasagne.readthedocs.io/en/latest/#)
[Py-Pong](http://pygame.org/project-py-pong-2040-.html)  
[Theano-based implementation of Deep Q-learning](https://github.com/spragunr/deep_q_rl)

## Basics
[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)  
[Demystifying Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)  

## File Structure
runPyPong.py serves as a configuration file.  
In pypong/q_network.py the network parameters like the number of nodes and hidden layers can be changed
pypong/data_set.py explains the used data set and its functions
pypong/NeuralAgent.py shows the implemented agent.

## Usage
If all parameters are set the training or game can be started with `python3 runPyPong.py`.  
In the current configuration (see runPyPong.py), this command loads a state after 100.000.000 training steps. The neural network is set up to three hidden layers with 40 nodes each (see pypong/q_network.py). 

## Game structure
Q-Learning is based on Markov Decision Processes ([MDP](https://en.wikipedia.org/wiki/Markov_decision_process)). Therefore states, actions, and rewards have
to be defined.

### States
The states consist of the x- and y-position of the ball as well as the y-position of the paddle.
According to *PHI_LENGTH* subsequent states are the input of the network. The states are recorded
every *FRAME_SKIP*.

### Rewards
Rewards are distributed for the following situations

* Paddle hit the ball = +0.5  
* Agent wins the round = +1
* Agent loses the round = -1

### Actions
The possible actions for the agent are to move up, down or to stay at the current position, resulting
in an action space of 3.

## Video
[![Pong QLearning](http://img.youtube.com/vi/qSGvAzfSmDM/2.jpg)](https://youtu.be/qSGvAzfSmDM "Pong QLearning")