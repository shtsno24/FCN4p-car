#-*-coding:utf-8-*-
import serial
import time

try:
    with serial.Serial('COM7', 9600) as ser:
        while True:
            line = "10,20,30,e"
            ser.write(bytes(line.encode()))
            time.sleep(0.5)
    
except:
    import traceback
    traceback.print_exc()
    input("press any key to continue...")

finally:
    ser.close()
    input("closed")
