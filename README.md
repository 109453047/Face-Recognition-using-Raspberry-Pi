# Face-Recognition-using-Raspberry-Pi
for homework
## 1. 關於專案
將鏡頭放在閘門入口，透過人臉辨識確認是否為已存取之人員，若確認為已存取人員將發出Do-re-me暗示可以通過，若辨識非存取人員則發出警告聲響，並拍照存取入侵者。

## 2. 專案緣由
機場捷運近兩年來不斷推陳出新，引領軌道業先驅新增信用卡及行動支付感應支付，作為國家門面希望推出更加智慧化的人臉辨識功能，帶領台灣進步。

## 3. 專案構想
人臉辨識因涉及隱私權，僅提供定期票購買旅客使用，可供旅客30、60、90、120日內搭乘使用，此方式可避免旅客定期票遺失、刷錯卡、一票多人共用等情形。
<br>支付定期費用的旅客使用相機拍照後，將照片存取在資料庫中，當旅客欲搭乘捷運於出入口感應人臉，透過人臉辨識確認是否為已存取人員，若為存取人員則透過蜂鳴器發出do-re-me聲響，若非存取人員則發出告警聲響，並拍照存取入侵者頭像，可公布在車站避免旅客違規闖入。

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
原先的專案規劃為丟菸蒂檢舉達人，因此而開始使用研究Haar Cascade 分類器做香菸偵測，而網路上大多的資源為人臉偵測，我也拿人臉偵測做練習，而後嘗試使用香菸偵測時，發現形狀太小辨識度不佳，故轉換專案題目，執行老闆要求的專案-人臉辨識。
在執行人臉偵測時，發現3個問題：
<br>1.多數的網站均使用VideoCapture ( 0 )作讀取，實際使用發現只能適用於 PiCam，藉由同學作業參考資料中，使用舵機雲台追蹤臉孔[[2]](https://github.com/ch-tseng/PanTilt/blob/master/main.py)，發現可以使用VideoStream，為此要執行指令下達```$pip3 install imutils```
<br>2.人臉偵測於Rasberry Pi 3未有即時偵測畫面，經網路搜尋發現是VNC的問題
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
### 第5步：人臉模型訓練
### 第6步：人臉辨識


## 8. 影片呈現連結
https://youtu.be/Sn16_KW4zAc
## 9. 可以改進或其他發想
## 10.參考資料
[1]https://github.com/kunalyelne/Face-Recognition-using-Raspberry-Pi
<br>[2]https://github.com/ch-tseng/PanTilt/blob/master/main.py
<br>[3]https://www.twblogs.net/a/5db2cffebd9eee310d9fff12
