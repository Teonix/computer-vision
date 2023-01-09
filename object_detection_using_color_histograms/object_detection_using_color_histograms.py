# ΑΠΑΙΤΟΥΜΕΝΑ ΠΑΚΕΤΑ
import cv2
import numpy as np
import random
import time
import skimage.metrics
import skimage.measure
import os

# ΔΗΛΩΣΗ ΤΟΥ ΒΑΣΙΚΟΥ ΜΟΝΟΠΑΤΙΟΥ 
path = r''

# ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
os.chdir(path) 

# ΜΕΘΟΔΟΣ CORRELATION

# ΔΙΑΒΑΣΜΑ ΤΗΣ ΕΙΚΟΝΑΣ
img = cv2.imread('card.jpg')

# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img_cpy = img.copy()
h_img, w_img = img.shape[:2]

# ΔΙΑΒΑΣΜΑ ΤΟΥ TEMPLATE
template = cv2.imread('template.jpg')
h_temp, w_temp = template.shape[:2]

# ΙΣΤΟΓΡΑΜΜΑ ΤΟΥ TEMPLATE
histTemp = cv2.calcHist([template], [0], None, [256], [0, 256])

score = np.zeros((h_img - h_temp, w_img - w_temp))
top_left = []
mask = np.ones((h_temp, w_temp, 3))
threshold = 0.95

# ΣΑΡΩΜΑ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΤΜΗΜΑΤΑ ΜΕΓΕΘΟΥΣ ΤΟΥ TEMPLATE
for i in range(0, h_img - h_temp):
    for j in range(0, w_img - w_temp):
        block = img[i: i + h_temp, j: j +w_temp]  
        histBlock = cv2.calcHist([block], [0], None, [256], [0, 256]) # ΙΣΤΟΓΡΑΜΜΑ ΤΟΥ ΜΠΛΟΚ
        score[i, j] = cv2.compareHist(histTemp, histBlock, cv2.HISTCMP_CORREL)   # ΣΥΓΚΡΙΣΗ ΤΩΝ ΔΥΟ ΙΣΤΟΓΡΑΜΜΑΤΩΝ
        if score[i, j] > threshold:  
           offset = np.array((i, j)) 
           img[offset[0]:offset[0] + mask.shape[0], offset[1]:offset[1] + mask.shape[1]] = mask
           (max_Y, max_X) = (i, j)
           top_left.append((max_X, max_Y)) 
           
# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for i in range(0,len(top_left)):
    loc = top_left[i]
    cv2.rectangle(img_cpy, loc, (loc[0] + w_temp, loc[1] + h_temp), (0, 255, 0), 3)

# ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
directory = path + '\Correlation_match'

# ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
os.chdir(directory)  

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with correlation method', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with correlation method',img_cpy) 
cv2.resizeWindow('Detection with correlation method', 819, 1024)
cv2.imwrite('detection_correlation.jpg', img_cpy)
cv2.waitKey()
cv2.destroyAllWindows() 


# ΜΕΘΟΔΟΣ HELLINGER

# ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
os.chdir(path) 

# ΔΙΑΒΑΣΜΑ ΤΗΣ ΕΙΚΟΝΑΣ
img = cv2.imread('card.jpg')

# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img_cpy = img.copy()
h_img, w_img = img.shape[:2]

score = np.zeros((h_img - h_temp, w_img - w_temp))
top_left = []
mask = np.ones((h_temp, w_temp, 3))
threshold = 0.2

# ΣΑΡΩΜΑ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΤΜΗΜΑΤΑ ΜΕΓΕΘΟΥΣ ΤΟΥ TEMPLATE
for i in range(0, h_img - h_temp):
    for j in range(0, w_img - w_temp):
        block = img[i: i + h_temp, j: j +w_temp]  
        histBlock = cv2.calcHist([block], [0], None, [256], [0, 256]) # ΙΣΤΟΓΡΑΜΜΑ ΤΟΥ ΜΠΛΟΚ
        score[i, j] = cv2.compareHist(histTemp, histBlock, cv2.HISTCMP_HELLINGER)   # ΣΥΓΚΡΙΣΗ ΤΩΝ ΔΥΟ ΙΣΤΟΓΡΑΜΜΑΤΩΝ
        if score[i, j] < threshold:  
           offset = np.array((i, j)) 
           img[offset[0]:offset[0] + mask.shape[0], offset[1]:offset[1] + mask.shape[1]] = mask
           (max_Y, max_X) = (i, j)
           top_left.append((max_X, max_Y)) 
           
# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for i in range(0,len(top_left)):
    loc = top_left[i]
    cv2.rectangle(img_cpy, loc, (loc[0] + w_temp, loc[1] + h_temp), (0, 255, 0), 3)

# ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
directory1 = path + '\Hellinger_match'

# ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
os.chdir(directory1) 

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with hellinger method', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with hellinger method',img_cpy) 
cv2.resizeWindow('Detection with hellinger method', 819, 1024)
cv2.imwrite('detection_hellinger.jpg', img_cpy)
cv2.waitKey()
cv2.destroyAllWindows() 