#-*- coding:utf-8 -*-
import cv2
import os
import sys

video = "test.mp4"
image_dir = "raw/"

def video2frame(video_file,img_dir,img_file="img_%s.png"):
    
    i = 0
    cam = cv2.VideoCapture(video_file)

    if os.path.exists(video_file) == False:
        print(video_file + ' does not exist!')
        sys.exit()

    if os.path.exists(img_dir) == False:
        print(img_dir + ' does not exist!')
        os.mkdir(img_dir)

    while(cam.isOpened()):
        flag,frame = cam.read()
        if flag == False:
            break
        cv2.imwrite(img_dir+img_file % str(i).zfill(6),frame)
        print("Save",img_dir+img_file % str(i).zfill(6))
        i += 1

    cam.release()

try:
    video2frame(video,image_dir)

except:
    import traceback
    traceback.print_exc()
    input("press any key to continue...")

finally:
    input("end >>")   
