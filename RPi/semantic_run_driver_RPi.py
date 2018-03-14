#-*- coding:utf-8-*-
import os
import cv2
import sys
import time
import RPi.GPIO as GPIO
from picamera import PiCamera
from picamera import array
import numpy as np
import fast_capture

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

norm_scale = 1
NPZ = "data/bin2train_data.npz"
model_folder = "model"
avr_time = 0
i = 0

camera = fast_capture.fast_capture()



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
    find_train_data(NPZ)

    input(">>")
    servo.start(0.0)
    camera.start()

    #load model
    model = L.Classifier(net.MLP())
    serializers.load_npz(model_folder + "/trained_model.npz",model)
   
    while True:
        start = time.time()    
        inp = camera.read()
        #inp = inp.array[0:60, :, 0:1].reshape(160,60).astype(np.float32)
        inp = cv2.flip(inp[0:24,:,0:1].reshape(inp.shape[1],inp.shape[0]).astype(np.float32),0)
        inp = inp.reshape(1,1,inp.shape[0],inp.shape[1])

        #detection
        output = model.predictor(inp * norm_scale)
        output = output.data.reshape(3,output.shape[2],output.shape[3]) * norm_scale

        #calc moments
        moment_img = output[0]
        Moments = cv2.moments(moment_img)
        cx,cy = int(Moments["m10"] / Moments["m00"]),int(Moments["m01"] / Moments["m00"])
        cx, cy = int(1.5 * (cx - moment_img.shape[1] / 2)), int(cy / 3)

        str_angle = np.arctan(float(cx) / float(-cy + moment_img.shape[0]))
        servo_duty = (np.pi / 2 + str_angle) * 9.5 + 2.5
        servo.ChangeDutyCycle(servo_duty)
                    
        avr_time += (time.time() - start)
        print(avr_time / (i + 1))
        i+=1

except ZeroDivisionError:
    print("ZeroDivisionError")
except:
    import traceback
    traceback.print_exc()
    input("press any key to continue...")

finally:
    input(">>")
    camera.stop()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()

