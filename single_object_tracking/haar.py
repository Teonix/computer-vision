import numpy as np
import cv2
import time

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_fullbody.xml
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

cap = cv2.VideoCapture("people.mp4")

# used to record the time when we processed last frame
prev_frame_time = time.time()

# used to record the time at which we processed current frame
new_frame_time = 0

total_fps = 0
total_pros = 0
counter = 0

while 1:
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

        # Time of frame
        new_frame_time = time.time()
        processing_time = new_frame_time - prev_frame_time
        total_pros += processing_time
        counter += 1

        # FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        total_fps += fps
        # converting the fps into integer
        fps = int(fps)

        prev_frame_time = new_frame_time

        for (x, y, w, h) in bodies:
          cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
          roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        cv2.imshow('Human Detection Haar', frame)
        k = cv2.waitKey(20) & 0xff
        if k == 27:
           break
    else:
        break

# Average time of frame
avg_pros = total_pros/counter
print("Average processing time per frame with Haar: {:.4f}".format(avg_pros))

# Average fps
avg_fps = total_fps/counter
print("Average fps with Haar: {:.0f}".format(avg_fps))

# Total time of video after haar
print("Time of video with Haar: {:.1f}".format(counter/avg_fps))

cap.release()
cv2.destroyAllWindows()