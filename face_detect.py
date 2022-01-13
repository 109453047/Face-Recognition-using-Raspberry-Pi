import RPi.GPIO as GPIO
import numpy as np
import cv2
from picamera import PiCamera
from imutils.video import VideoStream
import imutils
import time
GPIO.setmode(GPIO.BCM)

led_green = 23

vs = VideoStream(usePiCamera=True, resolution=(640,480)).start()
time.sleep(2.0)

face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
body_cascade = cv2.CascadeClassifier('/home/pi/Desktop/opencv/data/haarcascades/haarcascade_fullbody.xml')


while True:
    frame = vs.read()
    frame = frame.astype('uint8')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Frame',frame)
    key= cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
(h,w,d)= frame.shape
print("width={}, height={},depth={}".format(w,h,d))

vs.stop()
cv2.destroyAllWindows()
print('finish')