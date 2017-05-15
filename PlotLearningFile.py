import numpy
import matplotlib.pyplot as plt


graph=numpy.loadtxt(open('LearningFile/learning.csv','rb'), delimiter=',', skiprows=2)
fig1=plt.figure()
plt.plot(graph[:,1], graph[:,0])
plt.ylabel('Loss Average')
plt.xlabel('Epochs')
plt.title('Loss Average')
plt.savefig('LearningFile.png')


graph=numpy.loadtxt(open('LearningFile/reward.csv','rb'), delimiter=',', skiprows=2)
fig2=plt.figure()
plt.plot(graph[:,0],graph[:,1])
plt.ylabel('Add up Reward')
plt.xlabel('Epochs')
plt.title('Rewards')
plt.savefig('RewardFile.png')

