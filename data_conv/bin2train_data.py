#-*-coding:utf-8-*-
import numpy as np
import cv2
import os

i = 0 
ESC = 27
output_window_name = "output"
cv2.namedWindow(output_window_name)
img_scale = 0.4
norm_scale = 1

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

rot_img_bin = np.empty((1, 1, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale)),np.uint8)
rot_img = np.empty((1, 1, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale)),np.uint8)
rot_img_test = np.empty((1, 3, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale)),np.uint8)

print(rot_img_test.shape)



for i in range(img.shape[0]):
    buff_bin = cv2.imread(dir_bin_name + "/" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
    buff_img = img[i].reshape((img.shape[2],img.shape[3]))
    buff_test = np.stack((buff_bin,buff_bin,buff_bin),axis=0)
    
    buff_test[0][buff_test[0] != 0] = 255 * norm_scale
    buff_test[0][buff_test[0] == 0] = 127 * norm_scale
    buff_test[0][buff_test[0] != 127 * norm_scale] = 0

    buff_test[1][buff_test[1] != 127 * norm_scale] = 0

    buff_test[2][buff_test[2] != 255 * norm_scale] = 0
    buff_test[2][buff_test[2] == 255 * norm_scale] = 127
    buff_test = buff_test.transpose(1,2,0)

 
    
    rot_buff_bin = cv2.flip(buff_bin,1)
    rot_buff_img = cv2.flip(buff_img,1)
    rot_buff_test = cv2.flip(buff_test,1)

    cv2.imshow(output_window_name,  cv2.resize(rot_buff_test.astype(np.uint8),(img.shape[3] * 5,img.shape[2] * 5)))

    buff_img = cv2.resize(buff_img,(int(img.shape[3] * img_scale),int(img.shape[2] * img_scale)))
    buff_bin = cv2.resize(buff_bin,(int(img.shape[3] * img_scale),int(img.shape[2] * img_scale)))
    buff_test = cv2.resize(buff_test,(int(img.shape[3] * img_scale),int(img.shape[2] * img_scale)))
    rot_buff_img = cv2.resize(rot_buff_img,(int(img.shape[3] * img_scale),int(img.shape[2] * img_scale)))
    rot_buff_bin = cv2.resize(rot_buff_bin,(int(img.shape[3] * img_scale),int(img.shape[2] * img_scale)))
    rot_buff_test = cv2.resize(rot_buff_test,(int(img.shape[3] * img_scale),int(img.shape[2] * img_scale)))

    buff_test = buff_test.transpose(2,0,1)
    rot_buff_test = rot_buff_test.transpose(2,0,1)

    print(buff_test.shape)
    print(buff_img.shape)

    buff_img = buff_img.reshape(1, 1, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale))
    buff_bin = buff_bin.reshape(1, 1, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale))
    buff_test = buff_test.reshape(1, 3, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale))
    rot_buff_img = rot_buff_img.reshape(1, 1, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale))
    rot_buff_bin = rot_buff_bin.reshape(1, 1, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale))
    rot_buff_test = rot_buff_test.reshape(1, 3, int(img.shape[2] * img_scale),int(img.shape[3] * img_scale))

    rot_img_bin = np.vstack((rot_img_bin, buff_bin))
    rot_img_bin = np.vstack((rot_img_bin, rot_buff_bin))
    rot_img = np.vstack((rot_img, buff_img))
    rot_img = np.vstack((rot_img, rot_buff_img))
    rot_img_test = np.vstack((rot_img_test, buff_test))
    rot_img_test = np.vstack((rot_img_test, rot_buff_test))
    
    key = cv2.waitKey(1)

rot_img = np.delete(rot_img, 0, axis = 0)
rot_img_test = np.delete(rot_img_test, 0, axis = 0)
rot_img_bin = np.delete(rot_img_bin, 0, axis = 0)
print(rot_img_bin.shape)
print(rot_img.shape)
print(rot_img_test.shape)



np.savez("bin2train_data.npz", img=rot_img, img_bin = rot_img_bin, img_test = rot_img_test,label=label)


while True:
        key = cv2.waitKey(10)
        if key == ESC:
            break
cv2.destroyAllWindows()
