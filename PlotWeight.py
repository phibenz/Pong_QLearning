import numpy
import pypong
import pickle
import sys
import lasagne.layers
import matplotlib.pyplot as plt
import os


files=os.listdir('NetFiles/')
for i in range(len(files)):
    
    filename=files[i].split('_')
    filename=filename[1].split('.')
    filename=filename[0]
    
    if len(filename)<10:
        for j in range(10-len(filename)):
            filename='0'+filename
     
    if filename+'.png' not in os.listdir('WeightPics/'):
    
        print('...Processing Picture ' + str(i+1) + ' of ' + str(len(files)) )
        net_file = open('NetFiles/' + files[i], 'rb')
        network = pickle.load(net_file)
        net_file.close()
    
        q_layers = lasagne.layers.get_all_layers(network.lOut)
        w1 = q_layers[1].W.get_value()
        w2 = q_layers[2].W.get_value()
        #w3 = q_layers[3].W.get_value()
        #w4 = q_layers[4].W.get_value()

        f,axarr=plt.subplots(1,2)

        axarr[0].matshow(w1,  cmap=plt.cm.coolwarm)
        axarr[1].matshow(w2,  cmap=plt.cm.coolwarm) 
        #axarr[1,0].matshow(w3,  cmap=plt.cm.coolwarm)
        #axarr[1,1].matshow(w4,  cmap=plt.cm.coolwarm)

        plt.savefig('WeightPics/' + filename + '.png')
    else: 
        print('Picture ', filename+'.png', 'already in Folder.')
