# Face-Recognition-using-Raspberry-Pi
IoT & Data Science homework
## 1. 關於專案
將鏡頭放在閘門入口，透過人臉辨識確認是否為已存取之人員，若確認為已存取人員將發出Do-re-mi暗示可以通過，若辨識非存取人員則發出警告聲響，並拍照存取入侵者頭像。

## 2. 專案緣由
機場捷運近兩年來不斷推陳出新，引領軌道業先驅新增信用卡及行動支付感應支付，作為國家門面希望推出更加智慧化的人臉辨識功能，帶領台灣進步。

## 3. 專案構想
人臉辨識因涉及隱私權，僅提供定期票購買旅客使用，可供旅客30、60、90、120日內搭乘使用，此方式可避免旅客定期票遺失、刷錯卡、一票多人共用等情形。
<br>支付定期費用的旅客使用相機拍照後，將照片存取在資料庫中，當旅客欲搭乘捷運於出入口感應人臉，透過人臉辨識確認是否為已存取人員，若為存取人員則透過蜂鳴器發出do-re-mi聲響，若非存取人員則發出告警聲響，並拍照存取入侵者頭像，可公布在車站避免旅客違規闖入。
* 擬於下方圖片箭頭處架設攝影機
![](pic/entrance.jpg)
## 4. 專案所需實體材料
* 一個Rasberry Pi 3
* 一個Raspberry Pi 樹莓派UPS 鋰電池擴充板USB 電源供應模組行動電源
* 一顆鏡頭
* 麵包板實驗板
* 兩條公對母杜邦線 (目的:連接Rasberry Pi與蜂鳴器)
* 蜂鳴器
* 筆電
## 5. 材料細節
* 鏡頭
![](pic/camera.jpg)
* 蜂鳴器
![](pic/buzzer1.jpg)
## 6. 線路設計,指令表與實體照片
![](pic/deployment.PNG)
## 7. 程式設計
### 第1步：安裝相機
### 第2步：安裝OpenCV
使用指令下達 ```$pip3 install opencv-python```
### 第3步：人臉檢測
原先的專案規劃為「丟菸蒂檢舉達人」，因此開始研究Haar Cascade 分類器，先拿網路上最多資訊的人臉偵測做練習，而後嘗試使用香菸偵測時，發現形狀太小辨識度不佳，難過的轉換專案題目，執行公司老闆要求的專案-人臉辨識。
<br> 起初在執行時遇到許多問題：
<br>1.多數的網站均使用VideoCapture ( 0 )作讀取，實際使用發現只能適用於 PiCam，藉由同學作業參考資料中，使用舵機雲台追蹤臉孔[[2]](https://github.com/ch-tseng/PanTilt/blob/master/main.py)，發現可以使用VideoStream，為此要執行指令下達```$pip3 install imutils```
<br>2.人臉偵測於Rasberry Pi 3未有即時偵測畫面，經網路搜尋發現是VNC viewer的問題，故調整Rasberry Pi 3上VNC viewer的設定
<br>3.問題1當中參考資料使用```vs = VideoStream(usePiCamera=1).start()```，執行後發現偵測畫面太大無法顯示，經搜尋後[[3]](https://www.twblogs.net/a/5db2cffebd9eee310d9fff12)調整為```vs = VideoStream(usePiCamera=True, resolution=(640,480)).start()```
```python
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
```
### 第4步：人臉數據蒐集
至此步驟起，參考github上使用樹梅派做人臉辨識[[4]](https://github.com/kunalyelne/Face-Recognition-using-Raspberry-Pi)，在此步驟中，可以讓用戶拍照，蒐集欲執行人臉辨識的照片。
<br>在人臉數據蒐集時也發現問題，原先的程式碼face_id只能為整數無法紀錄對應的人名，為了方便公司管理旅客名單，因此我在此步驟增加一欄```user = input('\n Enter user name end press <Enter> ==>  ')```，讓每張蒐集的照片名稱中，顯示face_id對應的用戶名稱。
```python
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
```
### 第5步：模型訓練
藉由已蒐集的數據集，訓練人臉辨識模型，並存取在```.yml```文件當中，供第6步驟人臉辨識使用
```python
import cv2
import numpy as np
from PIL import Image
import os
# Path for face image database
path = '/home/pi/Desktop/STUDY/dataset'

os.chdir("/home/pi/Desktop/opencv/data/haarcascades")
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('/home/pi/Desktop/opencv/data/haarcascades/haarcascade_frontalface_default.xml');
# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\nTraining faces. It will take few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('/home/pi/Desktop/STUDY/trainer.yml')
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
```
### 第6步：人臉辨識
最後執行人臉辨識，藉由前一步訓練的模型，辨識新偵測到的人臉，預測用戶名稱與顯示信任程度
<br>在此步驟中，新增了3個功能：
<br>1.為避免辨識成功或失敗只是偶然，故增加程式碼紀錄成功或失敗次數，當達到一定次數才判定為成功或失敗
<br>2.增加程式碼辨識失敗一定次數後，拍攝入侵者的照片，並記錄當下時間```cv2.imwrite('/home/pi/Desktop/STUDY/intruder/%s.jpg' % ptime,img) #if found instruder will take photo```
<br>3.增加蜂鳴器在辨識成功時發出Do-re-mi聲響[[5]](https://sites.google.com/site/zsgititit/home/raspberry-shu-mei-pai/raspberry-shi-yong-fengbuzzier)及辨識失敗時發出短鳴的程式碼
```python
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
```
## 8. 影片呈現連結
https://youtu.be/Sn16_KW4zAc
## 9. 可以改進或其他發想
* 攝影機架設與Raspberry-Pi藏匿的方式可以再優化。
* 因疫情因素公司車站管制，無法於車站執行現場測試，之後可至車站測試。
* 因旅客曾反映目前票卡過閘感應失敗時的叫聲太大，因此在本專案設計時以不擾民之聲響為主，但可能會使現場人員未發現侵入者，故之後可以再做討論調整。
* 依據規劃提供定期票旅客使用人臉辨識，每日都可能有當日新加入或需刪除名單，可思考自動更新訓練模型的方式
## 10.參考資料
[1]
<br>[2]https://github.com/ch-tseng/PanTilt/blob/master/main.py
<br>[3]https://github.com/kunalyelne/Face-Recognition-using-Raspberry-Pi
<br>[4]https://www.twblogs.net/a/5db2cffebd9eee310d9fff12
<br>[5]https://sites.google.com/site/zsgititit/home/raspberry-shu-mei-pai/raspberry-shi-yong-fengbuzzier
