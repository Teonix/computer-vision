# ΑΠΑΙΤΟΥΜΕΝΑ ΠΑΚΕΤΑ
import cv2
import numpy as np
import random
import time
import skimage.metrics
import skimage.measure
import os
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering, MiniBatchKMeans
from skimage.color import label2rgb

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
img = cv2.imread('fire.jpg')

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ BINARY ANNOTATED
(thresh, img_bw) = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)

# ΕΜΦΑΝΙΣΗ ΤΩΝ ΕΙΚΟΝΩΝ
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.imshow("Original image", img)
cv2.resizeWindow("Original image", 512, 287)
    
cv2.namedWindow("Binary annotated image", cv2.WINDOW_NORMAL)
cv2.imshow("Binary annotated image",img_bw)
cv2.resizeWindow("Binary annotated image", 512, 287)
cv2.imwrite("binary_image.jpg", img_bw)
cv2.waitKey()
cv2.destroyAllWindows()

# ΕΠΑΝΑΛΗΠΤΙΚΗ ΤΟΠΟΘΕΤΗΣΗ ΘΟΡΥΒΟΥ
for i in range(0, 21, 5):
    prob = i/100;
    
    # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
    img_n_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ΚΑΛΕΣΜΑ ΤΗΣ ΣΥΝΑΡΤΗΣΗΣ ΘΟΡΥΒΟΥ
    img_n_gray = noise(img_gray,prob)
    
    # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΜΕ ΘΟΡΥΒΟ ΣΕ BINARY ANNOTATED
    (thresh, img_n_bw) = cv2.threshold(img_n_gray, 220, 255, cv2.THRESH_BINARY)
         
    img_n = cv2.cvtColor(img_n_gray, cv2.COLOR_GRAY2BGR)

    # ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
    directory = path + '\_Noise_images'

    # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
    os.chdir(directory)     

    # ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
    cv2.namedWindow("Original image with {}% noise".format(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Original image with {}% noise".format(i), img_n)
    cv2.resizeWindow("Original image with {}% noise".format(i), 512, 287)
    cv2.imwrite("{}%_noise_original_image.jpg".format(i), img_n)

    # ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
    directory1 = path + '\Binary_images'

    # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
    os.chdir(directory1)  
    
    # ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
    cv2.namedWindow("Binary annotated image with {}% noise".format(i), cv2.WINDOW_NORMAL)
    cv2.imshow("Binary annotated image with {}% noise".format(i),img_n_bw)
    cv2.resizeWindow("Binary annotated image with {}% noise".format(i), 512, 287)
    cv2.imwrite("{}%_noise_binary_image.jpg".format(i), img_n_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # ΣΧΗΜΑ ΤΗΣ ΕΙΚΟΝΑΣ
    originShape = img_n.shape
    
    # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΠΙΝΑΚΑ ΔΙΑΣΤΑΣΕΩΝ
    flatImg=np.reshape(img_n, [-1, 3])
    
    # ΑΛΓΟΡΙΘΜΟΣ MEANSHIFT
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
    
    # ΕΚΤΕΛΕΣΗ ΤΟΥ MEANSHIST ΣΤΗ flatImg
    print('Using MeanShift algorithm, it takes time!')
    ms.fit(flatImg)
    labels=ms.labels_
    
    # ΕΥΡΕΣΗ ΤΟΥ ΑΡΙΘΜΟΥ ΤΩΝ ΣΥΣΤΑΔΩΝ
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    
    # ΕΜΦΑΝΙΣΗ ΤΗΣ ΣΥΣΤΑΔΟΠΟΙΗΜΕΝΗΣ ΕΙΚΟΝΑΣ
    segmentedImg = np.reshape(labels, originShape[:2])
    segmentedImg = label2rgb(segmentedImg) * 255 
    
    # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
    img_ms_gray = cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    
    # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ BINARY ANNOTATED
    (thresh, img_ms_bw) = cv2.threshold(img_ms_gray, 70, 255, cv2.THRESH_BINARY)

    # ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
    directory2 = path + '\Clustering_images\MeanShift\Original_clustering'

    # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
    os.chdir(directory2) 
    
    # ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
    cv2.namedWindow("MeanShiftSegments with {}% noise".format(i), cv2.WINDOW_NORMAL)
    cv2.imshow("MeanShiftSegments with {}% noise".format(i),segmentedImg.astype(np.uint8))
    cv2.resizeWindow("MeanShiftSegments with {}% noise".format(i), 512, 287)
    cv2.imwrite("{}%_noise_meanShift_image.jpg".format(i), segmentedImg.astype(np.uint8))

    # ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
    directory3 = path + '\Clustering_images\MeanShift\Binary_clustering'

    # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
    os.chdir(directory3) 
        
    # ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
    cv2.namedWindow("MeanShift binary annotated image with {}% noise".format(i), cv2.WINDOW_NORMAL) 
    cv2.imshow("MeanShift binary annotated image with {}% noise".format(i),img_ms_bw) 
    cv2.resizeWindow("MeanShift binary annotated image with {}% noise".format(i), 512, 287)
    cv2.imwrite("{}%_noise_meanShift_binary_image.jpg".format(i), img_ms_bw)
    cv2.waitKey() 
    cv2.destroyAllWindows() 
    
    # ΑΛΓΟΡΙΘΜΟΣ Kmeans
    print('Using kmeans algorithm, it is faster!')
    km = MiniBatchKMeans(n_clusters = n_clusters_)
    km.fit(flatImg)
    labels = km.labels_
    
    # ΕΜΦΑΝΙΣΗ ΤΗΣ ΣΥΣΤΑΔΟΠΟΙΗΜΕΝΗΣ ΕΙΚΟΝΑΣ
    segmentedImg = np.reshape(labels, originShape[:2]) 
    segmentedImg = label2rgb(segmentedImg) * 255
    
    # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ ΓΚΡΙ
    img_km_gray = cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_BGR2GRAY) 
    
    # ΜΕΤΑΤΡΟΠΗ ΤΗΣ ΕΙΚΟΝΑΣ ΣΕ BINARY ANNOTATED
    (thresh, img_km_bw) = cv2.threshold(img_km_gray, 40, 255, cv2.THRESH_BINARY) 

    # ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
    directory4 = path + '\Clustering_images\Kmeans\Original_clustering'

    # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
    os.chdir(directory4) 
     
    cv2.namedWindow("KmeansSegments with {}% noise".format(i), cv2.WINDOW_NORMAL) 
    cv2.imshow("KmeansSegments with {}% noise".format(i),segmentedImg.astype(np.uint8)) 
    cv2.resizeWindow("KmeansSegments with {}% noise".format(i), 512, 287) 
    cv2.imwrite("{}%_noise_kmeans_image.jpg".format(i), segmentedImg.astype(np.uint8))

    # ΔΗΛΩΣΗ ΜΟΝΟΠΑΤΙΟΥ ΑΠΟΘΗΚΕΥΣΗΣ
    directory5 = path + '\Clustering_images\Kmeans\Binary_clustering'

    # ΕΠΙΛΟΓΗ ΤΟΥ ΜΟΝΟΠΑΤΙΟΥ ΠΟΥ ΦΤΙΑΞΑΜΕ ΠΙΟ ΠΑΝΩ
    os.chdir(directory5)
    
    # ΕΜΦΑΝΙΣΗ ΤΗΣ ΕΙΚΟΝΑΣ
    cv2.namedWindow("Kmeans binary annotated image with {}% noise".format(i), cv2.WINDOW_NORMAL) 
    cv2.imshow("Kmeans binary annotated image with {}% noise".format(i),img_km_bw) 
    cv2.imwrite("{}%_noise_kmeans_binary_image.jpg".format(i), img_km_bw)    
    cv2.waitKey() 
    cv2.destroyAllWindows()
    
    # SSIM SCORE ΤΟΥ MEANSHIFT
    print("\n\n")
    (score_ms, _) = skimage.measure.compare_ssim(img_bw, img_ms_bw, full=True) 
    print( " .. SSIM score of MeanShift clustering binary annotated image with {}% noise: {:.4f} \n".format(i,score_ms)) 
    
    # SSIM SCORE ΤΟΥ KMEANS
    (score_km, _) = skimage.measure.compare_ssim(img_bw, img_km_bw, full=True) 
    print( " .. SSIM score of Kmeans clustering binary annotated image with {}% noise: {:.4f} \n".format(i,score_km)) 
    print("\n\n")