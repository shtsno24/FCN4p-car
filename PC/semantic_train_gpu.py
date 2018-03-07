#-*- coding:utf-8 -*-
import os
import sys
import time
#import cupy
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.training as T
import chainer.training.extensions as E
from chainer.datasets import tuple_dataset
from chainer import serializers

import net

epoch_num = 10000
norm_scale = 1
NPZ = "data/bin2train_data.npz"
model_folder = "model"


"""
using CPU : gpu_id = -1

using GPU : gpu_id = 0

"""
gpu_id = 0 
#gpu_id = -1


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
    #find dataset (NPZ file)
    find_train_data(NPZ)

    #load dataset (NPZ file)
    ortrain, ortrain_label = load_train_data(NPZ)
    #print(ortrain, ortrain_label)

    print("epoch : " + str(epoch_num))
    input(">>")

    #sprit dataset
    threshold = np.int32(ortrain.shape[0] * 0.50)
    ortrain = ortrain.astype(np.float32)
    orlab = ortrain_label.astype(np.float32)
    """
    train = tuple_dataset.TupleDataset(ortrain[0:threshold] / 255, orlab[0:threshold] / 255)
    test = tuple_dataset.TupleDataset(ortrain[0:threshold:] / 255,  orlab[0:threshold:] / 255)
    """
    train = tuple_dataset.TupleDataset(ortrain[0:threshold] * norm_scale, orlab[0:threshold] * norm_scale)
    test = tuple_dataset.TupleDataset(ortrain[0:threshold:] * norm_scale,  orlab[0:threshold:] * norm_scale)
    

    #load model
    model = L.Classifier(net.MLP(),lossfun=F.mean_squared_error)
    model.compute_accuracy = False
    
    #apply optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    #set iterator
    train_iter = chainer.iterators.SerialIterator(train, batch_size=30)
    test_iter = chainer.iterators.SerialIterator(test, batch_size=30, repeat=False, shuffle=False)

    #send model to gpu
    if gpu_id >= 0:
        model.to_gpu(gpu_id)


    updater = T.StandardUpdater(train_iter, optimizer, device = gpu_id)
    trainer = T.Trainer(updater, (epoch_num, 'epoch'), out='result')

    trainer.extend(E.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(E.LogReport())
    trainer.extend(E.ProgressBar())
    """
    trainer.extend(E.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy']))
     """
    trainer.extend(E.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.run()
    
except:
    import traceback
    traceback.print_exc()

finally:
    print('trained model is saved in "model" folder')
    create_folders(model_folder)
    model.to_cpu()
    serializers.save_npz(model_folder + "/trained_model.npz",model)
    input(">>")