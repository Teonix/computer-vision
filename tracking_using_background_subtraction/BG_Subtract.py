import cv2
from random import randint
import numpy as np

cap = cv2.VideoCapture('pedestrians.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
#fgbg = cv2.BackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    if ret == True:
        # additional bluring to handle real life noise
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        fgmask = fgbg.apply(frame)

        # create bounding box
        mask = 255 - fgmask
        _, contours, _ = cv2.findContours(
        fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            area = cv2.contourArea(contour)
            #only show contours that match area criterea
            if area > 500 and area < 20000:
                rect = cv2.boundingRect(contour)
                x, y, w, h = rect
                cv2.rectangle(fgmask, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow('BG_Subtract',fgmask)
        cv2.imshow('original', frame)

        k = cv2.waitKey(20) & 0xff
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()