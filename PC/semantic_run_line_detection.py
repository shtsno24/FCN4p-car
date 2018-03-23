#-*- coding:utf-8 -*-
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

import net
import util

NPZ = "data/bin2train_data.npz"
model_folder = "model"
avr_time = 0


window_name = "input/output/teacher"
cv2.namedWindow(window_name)
show_scale = 10

   


    
try:
    print("loading")
    #find dataset (NPZ file)
    util.find_train_data(NPZ)

    #load dataset (NPZ file)
    ortrain, ortrain_label = util.load_train_data(NPZ,"img","img_test")
    ortrain = ortrain.astype(np.float32)
    orlab = ortrain_label.astype(np.float32)

    print(ortrain.shape, ortrain_label.shape)

    input(">>")

    #load model
    model = L.Classifier(net.MLP())
    serializers.load_npz(model_folder + "/trained_model.npz",model)
   
    for j in range(1):
        for i in range(ortrain.shape[0]):
            inp = ortrain[i:i + 1,:,:,:]
            ans = orlab[i:i + 1,:,:,:]
            start = time.time()

            #detection
            output = model.predictor(inp)

            avr_time += (time.time() - start)
            print(j * ortrain.shape[0] + i ,avr_time / (j * ortrain.shape[0] + i + 1))         

            inp = inp.reshape(1,ortrain.shape[2],ortrain.shape[3])
            inp = np.vstack((inp,inp,inp))
            inp = inp.transpose(1,2,0)
            
            ans = ans.reshape(3,ortrain.shape[2],ortrain.shape[3])
            ans = ans.transpose(1,2,0)
            
            output = util.label2img(output)
            output_buff = output
            output = output.transpose(1,2,0)
           
            #calc moments
            moment_img = output_buff[0]
            moment_img[moment_img <= 10] = 0
            Moments = cv2.moments(moment_img)
            cx,cy = int(Moments["m10"] / Moments["m00"]),int(Moments["m01"] / Moments["m00"])
            cx, cy = int(1.5 * (cx - moment_img.shape[1] / 2)), int(cy / 3)

            str_angle = np.arctan(float(cx) / float(-cy + moment_img.shape[0]))
            moment_img = cv2.cvtColor(moment_img,cv2.COLOR_GRAY2BGR)
            cv2.circle(moment_img,(int(cx + moment_img.shape[1] / 2), cy), 4, (127,255,127),-1,4)

            print(str_angle / np.pi * 180)

            show_img = np.vstack((inp, output, moment_img, ans))

            cv2.imshow(window_name, cv2.resize(show_img.astype(np.uint8),(show_img.shape[1] * show_scale, show_img.shape[0] * show_scale)))
            key = cv2.waitKey(1)
        
    
except:
    import traceback
    traceback.print_exc()
    input("press any key to continue...")

finally:
    input(">>")
    cv2.destroyAllWindows()    

