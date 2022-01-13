import RPi.GPIO as GPIO
import numpy as np
import cv2
from imutils.video import VideoStream
import imutils
import time
GPIO.setmode(GPIO.BCM)

vs = VideoStream(usePiCamera=True, resolution=(640,480)).start()
time.sleep(2.0)

face_detector = cv2.CascadeClassifier('/home/pi/Desktop/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
user = input('\n Enter user name end press <Enter> ==>  ')
face_id = input('\n Enter user ID end press <Enter> ==>  ')
print("\n Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while(True):
    img = vs.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize = (30, 30))
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("/home/pi/Desktop/STUDY/dataset/" +str(user)+'.'+ str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
vs.stop()
cv2.destroyAllWindows()
print('finish')