# ΑΠΑΙΤΟΥΜΕΝΑ ΠΑΚΕΤΑ
import cv2
import numpy as np
import random
import time
import skimage.metrics
import skimage.measure
import os

# ΚΑΤΑΣΚΕΥΗ ΣΥΝΑΡΤΗΣΗΣ ΘΟΡΥΒΟΥ
def noise(image,prob):
  rows,cols = image.shape
  output = np.zeros(image.shape,np.uint8)
  for i in range(rows):
      for j in range(cols):
         pixel = image[i,j]
         output[i][j] = pixel + (prob * random.random() * pixel) - (prob * random.random() * pixel)
  return output

# ΔΗΛΩΣΗ ΤΟΥ ΒΑΣΙΚΟΥ ΜΟΝΟΠΑΤΙΟΥ 
path = r''

# ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
os.chdir(path) 

# ΔΙΑΒΑΣΜΑ ΤΗΣ ΕΙΚΟΝΑΣ
image = cv2.imread('card.jpg')

# ΔΙΑΒΑΣΜΑ ΤΟΥ TEMPLATE
template = cv2.imread('template.jpg',0)
w, h = template.shape[::-1]

### ΕΡΩΤΗΜΑ A ###

# ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
directory = path + '\_Temp_match_noise'

# ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
os.chdir(directory)  

# NOISE 0%
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ΕΙΣΑΓΩΓΗ ΘΟΡΥΒΟΥ
img_gray_noise_0 = noise(img_gray,0)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_gray_noise_0,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 0% noise', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 0% noise',img)
cv2.resizeWindow('Detection with 0% noise', 819, 1024)
cv2.imwrite('0%_noise.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


# NOISE 5%
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ΕΙΣΑΓΩΓΗ ΘΟΡΥΒΟΥ
img_gray_noise_5 = noise(img_gray,0.05)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_gray_noise_5,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 5% noise', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 5% noise',img)
cv2.resizeWindow('Detection with 5% noise', 819, 1024)
cv2.imwrite('5%_noise.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


# NOISE 10%
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ΕΙΣΑΓΩΓΗ ΘΟΡΥΒΟΥ
img_gray_noise_10 = noise(img_gray,0.1)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_gray_noise_10,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 10% noise', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 10% noise',img)
cv2.resizeWindow('Detection with 10% noise', 819, 1024)
cv2.imwrite('10%_noise.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


# NOISE 15%
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ΕΙΣΑΓΩΓΗ ΘΟΡΥΒΟΥ
img_gray_noise_15 = noise(img_gray,0.15)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_gray_noise_15,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 15% noise', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 15% noise',img)
cv2.resizeWindow('Detection with 15% noise', 819, 1024)
cv2.imwrite('15%_noise.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


# NOISE 20%
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ΕΙΣΑΓΩΓΗ ΘΟΡΥΒΟΥ
img_gray_noise_20 = noise(img_gray,0.2)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_gray_noise_20,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 20% noise', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 20% noise',img)
cv2.resizeWindow('Detection with 20% noise', 819, 1024)
cv2.imwrite('20%_noise.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


### ΕΡΩΤΗΜΑ B ###

# ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
directory1 = path + '\_Temp_match_noise_gaussian' 

# ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
os.chdir(directory1) 

# NOISE 0% WITH GAUSSIAN FILTER
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΕΦΑΡΜΟΓΗ ΤΟΥ GAUSSIAN FILTER
img_noise_gauss_0 = cv2.GaussianBlur(img_gray_noise_0, (5,5), 0)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_noise_gauss_0,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 0% noise and gaussian filter', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 0% noise and gaussian filter',img)
cv2.resizeWindow('Detection with 0% noise and gaussian filter', 819, 1024)
cv2.imwrite('0%_noise_gaussian.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


# NOISE 5% WITH GAUSSIAN FILTER
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΕΦΑΡΜΟΓΗ ΤΟΥ GAUSSIAN FILTER
img_noise_gauss_5 = cv2.GaussianBlur(img_gray_noise_5, (5,5), 0)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_noise_gauss_5,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 5% noise and gaussian filter', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 5% noise and gaussian filter',img)
cv2.resizeWindow('Detection with 5% noise and gaussian filter', 819, 1024)
cv2.imwrite('5%_noise_gaussian.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


# NOISE 10% WITH GAUSSIAN FILTER
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΕΦΑΡΜΟΓΗ ΤΟΥ GAUSSIAN FILTER
img_noise_gauss_10 = cv2.GaussianBlur(img_gray_noise_10, (5,5), 0)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_noise_gauss_10,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 10% noise and gaussian filter', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 10% noise and gaussian filter',img)
cv2.resizeWindow('Detection with 10% noise and gaussian filter', 819, 1024)
cv2.imwrite('10%_noise_gaussian.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


# NOISE 15% WITH GAUSSIAN FILTER
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΕΦΑΡΜΟΓΗ ΤΟΥ GAUSSIAN FILTER
img_noise_gauss_15 = cv2.GaussianBlur(img_gray_noise_15, (5,5), 0)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_noise_gauss_15,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 15% noise and gaussian filter', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 15% noise and gaussian filter',img)
cv2.resizeWindow('Detection with 15% noise and gaussian filter', 819, 1024)
cv2.imwrite('15%_noise_gaussian.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()


# NOISE 20% WITH GAUSSIAN FILTER
# ΑΝΤΙΓΡΑΦΗ ΤΗΣ ΑΡΧΙΚΗΣ ΕΙΚΟΝΑΣ
img = image.copy()

# ΕΦΑΡΜΟΓΗ ΤΟΥ GAUSSIAN FILTER
img_noise_gauss_20 = cv2.GaussianBlur(img_gray_noise_20, (5,5), 0)

# ΤΡΕΞΙΜΟ ΤΟΥ TEMPLATE MATCHING
res = cv2.matchTemplate(img_noise_gauss_20,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.3
loc = np.where( res >= threshold)

# ΣΧΕΔΙΑΣΜΟΣ ΤΩΝ ΚΟΥΤΙΩΝ ΣΤΑ ΚΑΤΑΛΛΗΛΑ ΣΗΜΕΙΑ
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
cv2.namedWindow('Detection with 20% noise and gaussian filter', cv2.WINDOW_NORMAL) 
cv2.imshow('Detection with 20% noise and gaussian filter',img)
cv2.resizeWindow('Detection with 20% noise and gaussian filter', 819, 1024)
cv2.imwrite('20%_noise_gaussian.jpg', img) 
cv2.waitKey()
cv2.destroyAllWindows()