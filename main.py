import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAvgPool2D
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Flatten
from tensorflow.keras import layers
import tensorflow as tf
import argparse
# use cv2 to read image
import cv2
import glob
import numpy as np

from matplotlib import pyplot as plt
from common import helper 
import os

current_path = os.path.dirname(os.path.realpath(__file__))


# MLP model
def MLP(input_shape, num_classes):
    model = Sequential()
    # Increase the number of neurons
    model.add(Dense(400, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model

# CNN Model
def CNN(input_shape, num_classes):
    model = Sequential([
        # 32 convolutional filters of size 3 x 3, 'relu activation', padding = same (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        # 2 X 2 max pooling layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Dropout with probability 20%. Useful to avoid overfitting. (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
        layers.Dropout(0.2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        # Flatten the last image features before liking to a FFN (https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D)
        layers.GlobalAvgPool2D(),
        # A simple fully connected layer with a 'relu' activation
        layers.Dense(64, activation='relu'),
        # A simple fully connected output layer with no activation
        layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model



# Transfer Learning Model
def TF_model(input_shape, num_classes):
    vgg_model = VGG19(include_top=False,input_shape=input_shape, weights='imagenet') # Load VGG model and weights
    # mark loaded layers as not trainable
    for layer in vgg_model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(vgg_model.output)
    fc1 = Dense(1024, activation='relu')(flat1)
    # activation = "sigmoid" -> multiple labels
    output = Dense(num_classes, activation='softmax')(fc1)
    classified_model = Model(inputs=vgg_model.input, outputs=output) 
    classified_model.summary()
    classified_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return classified_model

if __name__ == "__main__":
    model_choices = ["MLP", "CNN", "TF"]
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='select the model name', choices=model_choices)
    args = parser.parse_args()

    path = "Dataset/"
    model_name = args.model_name
    if model_name == "MLP":
        samples, letters = helper.MLP_load_data(path)
    elif model_name == "CNN":
        samples, letters = helper.CNN_load_data(path)
    elif model_name == "TF":
        samples, letters = helper.TL_load_data(path)
 
    num_classes = len(np.unique(letters))
    x_train, y_train, x_test, y_test, x_val, y_val = helper.split_dataset(samples, letters)

    if model_name == "MLP":
        # Normalize the data
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_val = x_val.astype('float32')
        x_train /= 255
        x_test /= 255
        x_val /= 255

    elif model_name == "CNN":
        # pass
        # Normalize the data
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_val = x_val.astype('float32')
        x_train /= 255
        x_test /= 255
        x_val /= 255
        x_train = tf.expand_dims(x_train, axis=-1)
        x_test = tf.expand_dims(x_test, axis=-1)
        x_val = tf.expand_dims(x_val, axis=-1)


    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_val.shape[0], 'val samples')   

    input_shape = x_train[0].shape

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    if model_name == "MLP":
        model = MLP(input_shape, num_classes)
    elif model_name == "TF":
        model = TF_model(input_shape, num_classes)
    elif model_name == "CNN":
        model = CNN(input_shape, num_classes)

    epochs = 10
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val)
                        )
    score = model.evaluate(x_val, y_val, verbose=0)

    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])


    helper.save_plot(history, current_path + "/images", model_name)

