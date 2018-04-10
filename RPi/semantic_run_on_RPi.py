#-*- coding:utf-8 -*-
import os
import sys
import time
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.training as T
import chainer.training.extensions as E
from chainer.datasets import tuple_dataset
from chainer import serializers

import net
import util

NPZ = "data/bin2train_data.npz"
model_folder = "model"
avr_time = 0
norm_scale = 1

show_scale = 3

    
try:
    print("loading")
    #find dataset (NPZ file)
    util.find_train_data(NPZ)

    #load dataset (NPZ file)
    ortrain, ortrain_label = util.load_train_data(NPZ)
    ortrain = ortrain.astype(np.float32)
    orlab = ortrain_label.astype(np.float32)

    print(ortrain.shape, ortrain_label.shape)

    input(">>")

    #load model
    model = L.Classifier(net.MLP())
    serializers.load_npz(model_folder + "/trained_model.npz",model)
    for j in range(10):
        for i in range(ortrain.shape[0]):
            inp = ortrain[i:i + 1,:,:,:]
            ans = orlab[i:i + 1,:,:,:]
            start = time.time()
            output = model.predictor(inp)
            end = time.time() - start
            avr_time += end
            
            print(j * ortrain.shape[0] + i ,avr_time / (j * ortrain.shape[0] + i + 1))         
            print(end)
            
                 
        
    
except:
    import traceback
    traceback.print_exc()
    input("press any key to continue...")

finally:
    input(">>")    

