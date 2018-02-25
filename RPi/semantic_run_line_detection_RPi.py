#-*- coding:utf -8-*-
import os
import cv2
import sys
import time
import RPi.GPIO as GPIO
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.training as T
import chainer.training.extensions as E
from chainer.datasets import tuple_dataset
from chainer import serializers

import net

servo_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin,GPIO.OUT)
servo = GPIO.PWM(servo_pin,100)

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
    servo.start(0.0)
    #load model
    model = L.Classifier(net.MLP())
    serializers.load_npz(model_folder + "/trained_model.npz",model)
   
    for j in range(100):
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
            cx, cy = int(1.5 * (cx - moment_img.shape[1] / 2)), int(cy / 3)

            str_angle = np.arctan(float(cx) / float(-cy + moment_img.shape[0]))
            #str_angle = str_angle / np.pi * 180
            servo_duty = (np.pi / 2 + str_angle) * 9.5 + 2.5
            servo.ChangeDutyCycle(servo_duty)
            #calc direction
            
            
            
            avr_time += (time.time() - start)
            print(j * ortrain.shape[0] + i ,avr_time / (j * ortrain.shape[0] + i + 1))
        
    
except:
    import traceback
    traceback.print_exc()
    input("press any key to continue...")

finally:
    input(">>")
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()

