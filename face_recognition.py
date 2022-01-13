import RPi.GPIO as GPIO
import cv2
import numpy as np
import os
from imutils.video import VideoStream
import imutils
import time
from datetime import datetime
#if can't recognize face,beep 0.5s
def warning():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(12, GPIO.OUT)
    GPIO.output(12,GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(12,GPIO.LOW)
    print("warning")

    GPIO.cleanup()
#if recognize face,beep do/re/me
def okpass():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(12, GPIO.OUT)
    p=GPIO.PWM(12,50)
    p.start(50)
    p.ChangeFrequency(523)
    time.sleep(0.2)
    p.ChangeFrequency(587)
    time.sleep(0.2)
    p.ChangeFrequency(659)
    time.sleep(0.2)
    print("PASS")
    p.stop()
    GPIO.cleanup()
    
os.chdir("/home/pi/Desktop/opencv/data/haarcascades")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/Desktop/STUDY/trainer.yml')
cascadePath = "/home/pi/Desktop/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0
i=0
j=0

# names related to ids: example ==> KUNAL: id=1,  etc
names = ['Jiani','chair','word','english']

# Initialize and start realtime video capture
vs = VideoStream(usePiCamera=True, resolution=(640,480)).start()
time.sleep(2.0)


while True:
    img =vs.read()
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (20, 20),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id,confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(id,confidence)
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (id<len(names) and confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            j=j+1
            if j>5:
                okpass()
                j=0
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            i=i+1
            if i>10:
                now = datetime.now()
                ptime = now.strftime("%Y_%m_%d_%H%M%S")
                cv2.imwrite('/home/pi/Desktop/STUDY/intruder/%s.jpg' % ptime,img) #if found instruder will take photo
                warning()
                i=0
    
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
vs.stop()
cv2.destroyAllWindows()