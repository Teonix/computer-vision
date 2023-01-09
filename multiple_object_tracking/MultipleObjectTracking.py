from __future__ import print_function
import sys
import cv2
from random import randint
import time
import math
import pandas as pd

#https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

#Step 2: Read First Frame of a Video
# Set video to load
videoPath = 'chickens.mp4'

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

#Step 3: Locate Objects in the First Frame
## Select boxes
bboxes = []
colors = []

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner
  bbox = cv2.selectROI('MultiTracker', frame)
  bboxes.append(bbox)
  colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
  print("Press q to quit selecting boxes and start tracking")
  print("Press any other key to select next object")
  k = cv2.waitKey(0) & 0xFF
  if (k == 113):  # q is pressed
    break

print('Selected bounding boxes {}'.format(bboxes))

# Step 4: Initialize the MultiTracker
# Specify the tracker type
#trackerType = "CSRT"
#trackerType ='KCF'
trackerType ='BOOSTING'

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)

# used to record the time when we processed last frame
prev_frame_time = time.time()

# used to record the time at which we processed current frame
new_frame_time = 0

total_pros = 0
counter = 0

prev_centroidX = 0
total_centroidX = 0

prev_centroidY = 0
total_centroidY = 0

# Step 5: Update MultiTracker & Display Results
# Process video and track objects
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # Time of frame
    new_frame_time = time.time()
    processing_time = new_frame_time-prev_frame_time
    total_pros +=  processing_time
    counter += 1
    prev_frame_time = new_frame_time

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # Calculate x,y movement of centroid every frame
    centroidX = (int(newbox[0]) + int(newbox[2]))/2
    mov_centroidX = centroidX - prev_centroidX
    total_centroidX += mov_centroidX
    prev_centroidX = centroidX

    centroidY = (int(newbox[1]) + int(newbox[3])) / 2
    mov_centroidY = centroidY - prev_centroidY
    total_centroidY += mov_centroidY
    prev_centroidY = centroidY

    # show frame
    cv2.imshow('MultiTracker', frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

# Average time of frame
avg_pros = total_pros/counter
print("Average processing time per frame: {:.4f}".format(avg_pros))

# Average movement of centroid
avg_X = total_centroidX / counter
avg_Y = total_centroidY / counter
avg_dist = math.sqrt((avg_X*avg_X)+(avg_Y*avg_Y))
print("Average movement of centroid: {:.4f}".format(avg_dist))

# Save data to excel
data = {
    'Technique name': [trackerType],
    'Frame number': [counter],
    'Processing time': [avg_pros],
    'Centroid change box': [avg_dist],
}
df = pd.DataFrame(data, columns=['Technique name','Frame number',
                                   'Processing time','Centroid change box'])
df.to_excel("{}_technique.xls".format(trackerType))

cap.release()
cv2.destroyAllWindows()