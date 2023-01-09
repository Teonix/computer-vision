# import the necessary packages
import numpy as np
import cv2
import time

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture('people.mp4')

# used to record the time when we processed last frame
prev_frame_time = time.time()

# used to record the time at which we processed current frame
new_frame_time = 0

total_fps = 0
total_pros = 0
counter = 0

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
       # using a greyscale picture, also for faster detection
       gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

       # detect people in the image
       # returns the bounding boxes for the detected objects
       boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

       boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

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

       for (xA, yA, xB, yB) in boxes:
           # display the detected boxes in the colour picture
           cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)

       # Display the resulting frame
       cv2.imshow('Human Detection Hog', frame)
       k = cv2.waitKey(20) & 0xff
       if k == 27:
          break
    else:
        break

# Average time of frame
avg_pros = total_pros/counter
print("Average processing time per frame with Hog: {:.4f}".format(avg_pros))

# Average fps
avg_fps = total_fps/counter
print("Average fps with Hog: {:.0f}".format(avg_fps))

# Total time of video after hog
print("Time of video with Hog: {:.1f}".format(counter/avg_fps))

# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)