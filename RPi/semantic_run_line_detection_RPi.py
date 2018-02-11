#-*- coding:utf -8-*-

import os
import cv2
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


threshold_1 = 60
threshold_2 = 150

norm_scale = 10
NPZ = "data/bin2train_data.npz"
model_folder = "model"
avr_time = 0


def find_train_data(npz):
    #find NPZ file
    if os.path.exists(npz) == False:
        print(npz + ' does not exist!')
        input(">>")
        sys.exit()

def create_folders(folder):
    #create a folder for model
    if os.path.exists(folder) == False:
        print("generate a folder")
        os.mkdir(folder)

def load_train_data(npz):
    print("loading dataset for training")
    #loading data from NPZ file
    with np.load(npz) as data:
        tmp_train = data["img"]
        tmp_train_label = data["img_bin"]
    
    print(tmp_train.shape)
    print(tmp_train_label.shape)

    return tmp_train, tmp_train_label


class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__(conv1=L.Convolution2D(1, 8, 5, stride=5),
            conv2=L.Convolution2D(None, 16, 4, stride=4),
            conv3=L.Convolution2D(None, 32, 2, stride=1),

            deconv3 = L.Deconvolution2D(None,16,2,stride=1),
            deconv2 = L.Deconvolution2D(None,8,4,stride=4),
            deconv1 = L.Deconvolution2D(None,1,5,stride=5))


    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv1(h))
        
        return h

    
try:
    print("loading")
    #find dataset (NPZ file)
    find_train_data(NPZ)

    #load dataset (NPZ file)
    ortrain, ortrain_label = load_train_data(NPZ)
    ortrain = ortrain.astype(np.float32)
    orlab = ortrain_label.astype(np.float32)

    print(ortrain.shape, ortrain_label.shape)

    input(">>")

    #load model
    model = L.Classifier(MLP())
    serializers.load_npz(model_folder + "/trained_model.npz",model)
   
    for j in range(10):
        for i in range(ortrain.shape[0]):
            inp = ortrain[i:i + 1,:,:,:]
            start = time.time()

            #detection
            output = model.predictor(inp * norm_scale)

            #filtering
            output.data[(output.data >= threshold_1 * norm_scale) & (output.data <= threshold_2 * norm_scale)] = 127 * norm_scale
            output.data[output.data > threshold_2 * norm_scale] = 255 * norm_scale
            output.data[output.data < threshold_1] = 0
         
            #convert to gray scale
            output = (output.data.reshape(ortrain.shape[2],ortrain.shape[3]) / norm_scale).astype(np.uint8)
            
            #calc moments
            moment_img = cv2.bitwise_not(output, False)
            moment_img[moment_img < threshold_2] = 0
            Moments = cv2.moments(moment_img)
            cx,cy = int(Moments["m10"] / Moments["m00"]),int(Moments["m01"] / Moments["m00"])
            cx,cy = int(1.5*(cx-80)+80), int(cy/3)
            #calc direction
            
            print(cx,cy)
            
            avr_time += (time.time() - start)
            print(j * ortrain.shape[0] + i ,avr_time / (j * ortrain.shape[0] + i + 1))
        
    
except:
    import traceback
    traceback.print_exc()
    input("press any key to continue...")

finally:
    input(">>")
    cv2.destroyAllWindows()    

