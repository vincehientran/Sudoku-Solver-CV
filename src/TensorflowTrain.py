from TensorflowModel import create
import tensorflow as tf
import numpy as np

'''LOAD DATA'''
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize (x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

IMG_SIZE = 28
xtrainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

'''TRAIN MODEL'''
model = create()

model.fit(xtrainr, y_train, epochs=10, validation_split=0.3) # training the model

model.save('tensorflow_number_model')
