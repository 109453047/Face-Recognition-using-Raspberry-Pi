import RPi.GPIO as GPIO
import time
from datetime import datetime
from picamera import PiCamera

pir = 18
led_green = 23
camera = PiCamera()

GPIO.setmode(GPIO.BCM)
GPIO.setup(pir, GPIO.IN)

while True:
    input_state = GPIO.input(pir)
    if input_state == True:
        print('Motion Detected')
        GPIO.setup(led_green, GPIO.OUT)
        now = datetime.now()
        ptime = now.strftime("%Y_%m_%d_%H%M%S")
        camera.start_preview(fullscreen=False,window=(100,20,640,480))
        camera.capture('/home/pi/Desktop/STUDY/image_%s.jpg' % ptime)
        print('A photo has been taken')
        time.sleep(10)
    else:
        GPIO.setup(led_green, GPIO.IN)
        camera.stop_preview()