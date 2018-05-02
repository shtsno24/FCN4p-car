#-*- coding:utf-8 -*-
import os
import cv2
import sys
import numpy as np

import chainer
import chainer.functions as F

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

def load_train_data(npz, input="img", teacher="img_test"):
    print("loading dataset for training")
    #loading data from NPZ file
    with np.load(npz) as data:
        tmp_train = data[input]
        tmp_train_label = data[teacher]
    
    print(tmp_train.shape)
    print(tmp_train_label.shape)

    return tmp_train, tmp_train_label

def label2img(label):
    """
    * 1 | road        | blue 
    * 2 | out of road | green
    * 3 | line        | red
    * 
    *
    """
    buff = F.argmax(label, axis = 1)
    buff = F.vstack((buff, buff, buff))

    buff.data[0][buff.data[0] == 0] = 255
    buff.data[1][buff.data[1] == 0] = 0
    buff.data[2][buff.data[2] == 0] = 0

    buff.data[0][buff.data[0] == 1] = 0
    buff.data[1][buff.data[1] == 1] = 255
    buff.data[2][buff.data[2] == 1] = 0

    buff.data[0][buff.data[0] == 2] = 0
    buff.data[1][buff.data[1] == 2] = 0
    buff.data[2][buff.data[2] == 2] = 255

    return buff.data.astype(np.uint8)