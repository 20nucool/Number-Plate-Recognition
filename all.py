import RPi.GPIO as GPIO
import time
import cv2
import sqlite3
import pytesseract
from time import sleep
import numpy as np
from picamera import PiCamera

GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.IN)
GPIO.setup(15, GPIO.IN)
GPIO.setup(11, GPIO.OUT)
servo1 = GPIO.PWM(11, 50)

camera = PiCamera()
count = 0
count1 = 0

def create():
    connection = sqlite3.connect(":memory:")
    # print(connection.total_changes)
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS billing\
    (sn INTEGER, num_plate TEXT ,time_in REAL, time_out REAL)")
    cursor.execute("INSERT INTO billing VALUES('1', NULL , NULL, NULL)")
    # cursor.execute("SELECT sn, num_plate,time_in, time_out FROM billing").fetchall()
    # print(rows)


def camin(i):
    camera.start_preview()
    sleep(0.5)
    camera.capture('/home/pi/in%s.jpg' % i)
    sleep(0.5)
    camera.stop_preview()

def camout(i):
    camera.start_preview()
    sleep(0.5)
    camera.capture('/home/pi/out%s.jpg' % i)
    sleep(0.5)
    camera.stop_preview()

def crop(a):
    if a == 0:
        y = 250
        x = 200
        h = 400
        w = 700
        crp = img[y:y + h, x:x + w]
    elif a == 1:
        y = 150
        x = 200
        h = 400
        w = 700
        crp = img[y:y + h, x:x + w]

def plate_detect():
    gray = cv2.cvtColor(crp, cv2.COLOR_BGR2GRAY)
    fltr = cv2.bilateralFilter(gray, 13, 15, 15)
    # cv2.imshow('gray', fltr)

    edged = cv2.Canny(fltr, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screencnt = None

    for c in contours:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screencnt = approx
            break

    if screencnt is None:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(crp, [screencnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape, np.uint8)
    # new_image = cv2.drawContours(mask, [screencnt], 0, 255, -1, )
    # new_image = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('contour', crp)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
    # cv2.imshow('cropped', cropped)


def num_detect():
    re = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(re, (5, 5), 0)
    gray = cv2.medianBlur(blur, 3)
    # cv2.imshow('gray', gray)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #cv2.imshow("Otsu", thresh)
    # cv2.waitKey(0)
    kernel = np.ones((2, 2), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, kernel, iterations=1)

    img2 = dilation.copy()

    # print(pytesseract.image_to_string(img))
    # height, width, number of channels in image
    height = img2.shape[0]
    width = img2.shape[1]

    boxes = pytesseract.image_to_boxes(img2, lang='nep')
    #print(pytesseract.image_to_string(img2, lang='nep'))
    num = ""
    for b in boxes.splitlines():
        # print(b)
        b = b.split(' ')
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        # cv2.rectangle(img2, (x, height-y), (w, height-h), (0, 0, 255), 1)
        # if height / float(h) > 6: continue
        ratio = float((h - y) / (w - x))
        print(ratio)
        # if height to width ratio is less than 1.5 skip
        if 0.5 < ratio < 1.7:
            area = float((h - y) * (w - x))
            print(area)
            if 1000 < area < 11000:
                # print(b[0])
                cv2.rectangle(img2, (x, height - y), (w, height - h), (0, 0, 255), 1)
                num += b[0]
    return num


def add(i, plate_num):
    cursor.execute("INSERT INTO billing VALUES(?, ?, datetime('now'),NULL)", (i, plate_num))

def out(plate_num):
    moved_num_plate = plate_num
    cursor.execute("UPDATE billing SET time_out = datetime('now') WHERE num_plate = ?", (moved_num_plate,), )
    cursor.execute(
        "SELECT sn, num_plate, time_in, time_out FROM billing WHERE num_plate = ?",
        (moved_num_plate,),
    ).fetchall()

    diffinseconds = (cursor.execute("SELECT\
     (strftime('%s',time_out) - strftime('%s',time_in)) AS diff_in_sec FROM billing").fetchall())

    for row in diffinseconds:
        diff = row[0]
        if diff is not None and diff > 0:
            time_pe = diff
            return time_pe

def cost(tm):
    total_cost = tm * 2
    print(total_cost)

def delete(plate_num):
    released_num_plate = plate_num
    cursor.execute(
        "DELETE FROM billing WHERE plate_num = ?",
        (released_num_plate,)
    )

def servo():
    servo1.start(0)
    duty = 2
    while duty <= 8:
        servo1.ChangeDutyCycle(duty)
        time.sleep(0.5)
        duty = duty + 2
    time.sleep(10)
    duty = 8
    while duty >= 2:
        servo1.ChangeDutyCycle(duty)
        time.sleep(0.5)
        duty = duty - 2
    servo1.ChangeDutyCycle(0)


while True:
    val1 = GPIO.input(8)
    val2 = GPIO.input(15)
    print('value of ir in: ', val1)
    print('value of ir out: ', val2)
    create()
    if val1 == 0 or val2 == 0:
        print('object')
        sleep(2)
        x1 = GPIO.input(8)
        x2 = GPIO.input(15)
        if x1 == 0:
            print('OBJECT IN')
            camin(count)
            img = cv2.imread('/home/pi/in%s.jpg'%count)
            count += 1
            crop(0)
            plate_detect()
            plate_num = num_detect()
            add(count, plate_num)
            servo()
        elif x2 == 0:
            print('OBJECT OUT')
            camout(count1)
            img = cv2.imread('/home/pi/out%s.jpg'%count1)
            count1 += 1
            #print(count1)
            crop(1)
            plate_detect()
            plate_num = num_detect()
            time_total = out(plate_num)
            cost(time_total)
            delete(plate_num)
            servo()

