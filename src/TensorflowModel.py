import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

def create():
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape = (28, 28, 1)) )
    model.add(Activation("relu")) #remove numbers less than 0
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add (Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(10)) #last dense must be equal to 10
    model.add(Activation('softmax'))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    return model
