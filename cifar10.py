"""
Author: Griffin Noe '21 and Utkrist P. Thapa '21
CSCI-315: Artificial Intelligence
Implementing a CNN on the CIFAR-10 dataset
"""

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 100
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'CSCI_315_CIFAR_10'

# save the data into training and test sets
(x_train, y_train) , (x_test, y_test) = cifar10.load_data()

# convert label vectors (the ten classes) to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# normalize the dataset
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

# define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding= 'same', input_shape = x_train.shape[1:]),\
          activation = 'relu')
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation = 'softmax'))

# compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = \
              keras.optimizers.RMSprop(learning_rate = 0.0001, decay = 0.000001),
              metrics = ['accuracy'])

# train the model
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, \
          validation_data = (x_test, y_test), shuffle = True)

# save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("Saved trained model at %s" % model_path)

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose = 1)
print("Test Loss: ", scores[0])
print("Test accuracy: ", scores[1])

              
              






