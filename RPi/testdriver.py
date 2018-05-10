#-*-coding:utf-8-*-
import os
import sys
import picamera
import numpy as np
import serial

camera = picamera.PiCamera()
camera.resolution = (64,32)
Video = "video.h264"
COM = "/dev/ttyACM0"

if os.path.exists(Video)==True:
    os.remove(Video)

"""
if len(COM) <1:
    print("Nooooo COM port is selected!\nPlease select COM port")
    sys.exit()
elif len(COM)>1:
    print("Tooooo many COM ports are selected!\nPlease choose one COM port")
    sys.exit()
"""
try:
    camera.start_recording(Video)
    with serial.Serial(COM, 9600) as ser:
        while True:
            angle = input(">")
            if angle == "r":
                line = "-10,1,30,e"
            elif angle == "l":
                line = "10,1,30,e"
            else:
                line = "0,1,30,e"
            ser.write(bytes(line.encode()))

except:
    import traceback
    traceback.print_exc()
    input("press any key to continue...")

finally:
    camera.stop_recording()
    ser.close()
    camera.close()
    input("recoding stopeed >>")
