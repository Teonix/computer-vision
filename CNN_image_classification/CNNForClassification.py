import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd

#get the data
from erwt1_DataLoadClassif import X_train, Y_train, X_test, Y_test, X_val, Y_val, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS

batch_size = 100
num_classes = np.unique(Y_train).__len__()
epochs = 15

#defien some CNN parameters
baseNumOfFilters = 16

# the data, split between train and test sets
#(X_train, Y_train), (x_test, y_test) = mnist.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
Y_val = keras.utils.to_categorical(Y_val, num_classes)

# here we define and load the model

inputs = keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = keras.layers.Lambda(lambda x: x / 255)(inputs) #normalize the input
conv1 = keras.layers.Conv2D(filters=baseNumOfFilters, kernel_size=(13, 13))(s)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = keras.layers.Conv2D(filters=baseNumOfFilters * 2, kernel_size=(7, 7))(pool1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = keras.layers.Conv2D(filters=baseNumOfFilters * 4, kernel_size=(3, 3))(pool2)
drop3 = keras.layers.Dropout(0.25)(conv3)
flat1 = keras.layers.Flatten()(drop3)
dense1 = keras.layers.Dense(128, activation='relu')(flat1)
outputs = keras.layers.Dense(Y_train.shape[1], activation='softmax')(dense1)

CNNmodel = keras.Model(inputs=[inputs], outputs=[outputs])
CNNmodel.compile(optimizer='sgd', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
# print model summary
CNNmodel.summary()

# fit model parameters, given a set of training data
callbacksOptions = [
    keras.callbacks.EarlyStopping(patience=15, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
    keras.callbacks.ModelCheckpoint('tmpCNN.h5', verbose=1, save_best_only=True, save_weights_only=True)]

CNNmodel.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, epochs=epochs, verbose=1,
          callbacks=callbacksOptions, validation_data=(X_val, Y_val))

# calculate some common performance scores
score = CNNmodel.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# saving the trained model
model_name = 'finalCNN.h5'
CNNmodel.save(model_name)

# # loading a trained model & use it over test data
# loaded_model = keras.models.load_model(model_name)
#
# y_test_predictions_vectorized = loaded_model.predict(X_test)
# y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)

# CNN predictions
#now check for both train and test data, how well the model learned the patterns
y_pred_train = CNNmodel.predict(X_train)
y_pred_test = CNNmodel.predict(X_test)

y_pred_test = np.argmax(y_pred_test, axis=1)
y_pred_train = np.argmax(y_pred_train, axis=1)
Y_train = np.argmax(Y_train, axis=1)  
Y_test = np.argmax(Y_test, axis=1)

#calculate the scores
acc_train = accuracy_score(Y_train, y_pred_train)
acc_test = accuracy_score(Y_test, y_pred_test)
pre_train = precision_score(Y_train, y_pred_train, average='macro')
pre_test = precision_score(Y_test, y_pred_test, average='macro')
rec_train = recall_score(Y_train, y_pred_train, average='macro')
rec_test = recall_score(Y_test, y_pred_test, average='macro')
f1_train = f1_score(Y_train, y_pred_train, average='macro')
f1_test = f1_score(Y_test, y_pred_test, average='macro')

#print the scores
print('Accuracy scores of CNN classifier are:',
      'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
print('Precision scores of CNN classifier are:',
      'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
print('Recall scores of CNN classifier are:',
      'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
print('F1 scores of CNN classifier are:',
      'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
print('')

scores = {
    'Classifier Used': ['CNN'],
    'Train Data Ratio': ['{}%'.format(0.8 * 100)],
    'Accuracy (tr)': [acc_train],
    'Precision (tr)': [pre_train],
    'Recall (tr)': [rec_train],
    'F1 Score (tr)': [f1_train],
    'Accuracy (te)': [acc_test],
    'Precision (te)': [pre_test],
    'Recall (te)': [rec_test],
    'F1 Score (te)': [f1_test]
}
df = pd.DataFrame(scores, columns=['Classifier Used','Train Data Ratio', 
                                       'Accuracy (tr)', 'Precision (tr)', 'Recall (tr)', 'F1 Score (tr)',
                                       'Accuracy (te)', 'Precision (te)', 'Recall (te)', 'F1 Score (te)'])
df.to_excel('CNN_CLASSIFIER_(80%-20%).xls')