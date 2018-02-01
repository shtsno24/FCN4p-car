#-*-coding:utf-8-*-

import numpy as np
import cv2
import os

i = 0 
ESC = 27
output_window_name = "output"
output_bin_window_name = "output_bin"
cv2.namedWindow(output_window_name)
cv2.namedWindow(output_bin_window_name)


NPZ = "data.npz"
dir_bin_name = "img_bin"
dir_raw_name = "img_raw"

if os.path.exists(dir_bin_name) == False:
    os.mkdir(dir_bin_name)

if os.path.exists(dir_raw_name) == False:
    os.mkdir(dir_raw_name)

input(">>")


with np.load(NPZ) as data:
    print(data.files)
    picture = data["train"]
    label = data["train_labels"]
    print(picture.shape)
    img = picture.reshape((picture.shape[0],60,160))
    print(img.shape)
    
rot_img = np.empty((1, 1, img.shape[1],img.shape[2]),np.uint8)
rot_img_bin = np.empty((1, 1, img.shape[1],img.shape[2]),np.uint8)
print(rot_img.shape)

for i in range(img.shape[0]):
    buff = cv2.flip(img[i],0)
    buff_bin = cv2.flip(img[i],0)
    buff_bin[buff > 150] = 255
    buff_bin[buff <= 150] = 0

    print(buff)
    cv2.imshow(output_window_name, cv2.resize(buff.astype(np.uint8),(160*5,60*5)))
    cv2.imshow(output_bin_window_name,  cv2.resize(buff_bin.astype(np.uint8),(160*5,60*5)))
    cv2.imwrite(os.path.join(dir_bin_name ,(str(i)+".png")), buff_bin)
    cv2.imwrite(os.path.join(dir_raw_name ,(str(i)+".png")), buff)
    buff = buff.reshape(1, 1, img.shape[1],img.shape[2])
    buff_bin = buff_bin.reshape(1, 1, img.shape[1],img.shape[2])
    rot_img = np.vstack((rot_img, buff))
    rot_img_bin = np.vstack((rot_img_bin, buff_bin))
    key = cv2.waitKey(1)

rot_img = np.delete(rot_img, 0, axis = 0)
rot_img_bin = np.delete(rot_img_bin, 0, axis = 0)
print(rot_img.shape)


np.savez("array2bin", img=rot_img, img_bin = rot_img_bin, label=label)


while True:
        key = cv2.waitKey(10)
        if key == ESC:
            break
cv2.destroyAllWindows()
