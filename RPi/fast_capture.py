# -*- coding: utf-8 -*-
# please read http://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

import picamera
import numpy as np
from threading import Thread
import picamera.array

class fast_capture:
    def __init__(self, resolution = (64,24), color_effects = (128, 128), framerate = 90):
        self.cam = picamera.PiCamera()
        self.cam.resolution = resolution
        self.cam.color_effects = color_effects
        self.capture = picamera.array.PiYUVArray(self.cam)
        self.stream = self.cam.capture_continuous(self.capture, format = 'yuv', use_video_port = True)
        self.frame = None
        self.stopped = False
        self.led = True
    
    def start(self):
        t = Thread(target = self.update, args = ())
        t.setDaemon(True)
        t.start()
        return self
    
    def update(self):
        for f in self.stream:
            self.frame = f.array
            self.capture.truncate(0)
        if self.stopped:
            self.stream.close()
            self.capture.close()
            self.cam.close()
            return
        
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
