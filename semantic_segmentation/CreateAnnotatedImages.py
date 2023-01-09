import cv2
import tensorflow as tf
import keras
import numpy as np
import os

#load the existing model
tumor_detector = keras.models.load_model('finalSegCNN.h5')

#print the structure
tumor_detector.summary()

#read the image
rgb_image_to_use = cv2.imread('')
original_shape = rgb_image_to_use.shape
rgb_image_to_use = cv2.resize(rgb_image_to_use, (240, 240))

#convert the m x n x 3 image to 1 x m x n x 3 tensor
tensor_input_image = np.expand_dims(rgb_image_to_use, axis=0)

#pass the image to the model
models_annotation = tumor_detector.predict(tensor_input_image)

#convert tensor output to m x n x 3 annotated image

#first eliminated reduntant dimension 1 x m x n x 2 to m x n x2
annotated_image = np.squeeze(models_annotation, axis=0)

#map the output to label, per pixel
annotated_image = np.argmax(annotated_image, axis=2).astype('uint8')

#resize upwards
annotated_image = cv2.resize(annotated_image, (original_shape[0], original_shape[1]))

#change values to be able to plot
annotated_image = 255 * annotated_image

#visualize the results
cv2.namedWindow("Original image", cv2.WINDOW_NORMAL) 
cv2.imshow("Original image", rgb_image_to_use)  
cv2.resizeWindow("Original image", 240, 240) 

cv2.namedWindow("Annotated image", cv2.WINDOW_NORMAL) 
cv2.imshow("Annotated image", annotated_image)  
cv2.resizeWindow("Annotated image", 240, 240)
path = r'C:\Users\User\Desktop\erg4\erwt2\Annotated_Images_UNET'
os.chdir(path)  
cv2.imwrite("annotated_brain_tumor_5.jpg", annotated_image)

cv2.waitKey() 
cv2.destroyAllWindows()