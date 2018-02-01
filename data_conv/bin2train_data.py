#-*-coding:utf-8-*-

import numpy as np
import cv2
import os

i = 0 
ESC = 27
output_window_name = "output"
cv2.namedWindow(output_window_name)


NPZ = "array2bin.npz"
dir_bin_name = "img_bin"
dir_raw_name = "img_raw"

input(">>")


with np.load(NPZ) as data:
    print(data.files)
    img = data["img"]
    label = data["label"]
    print(img.shape)
    print(img.shape)

rot_img_bin = np.empty((1, 1, img.shape[2], img.shape[3]),np.uint8)
print(img)

for i in range(img.shape[0]):
    buff_bin = cv2.imread(dir_bin_name+"/"+str(i)+".png", cv2.IMREAD_GRAYSCALE)

    print(buff_bin)
    cv2.imshow(output_window_name,  cv2.resize(buff_bin.astype(np.uint8),(160*5,60*5)))
    buff_bin = buff_bin.reshape(1, 1, img.shape[2],img.shape[3])
    rot_img_bin = np.vstack((rot_img_bin, buff_bin))
    key = cv2.waitKey(1)

rot_img_bin = np.delete(rot_img_bin, 0, axis = 0)
print(rot_img_bin)


np.savez("bin2train_data.npz", img=img, img_bin = rot_img_bin, label=label)


while True:
        key = cv2.waitKey(10)
        if key == ESC:
            break
cv2.destroyAllWindows()
