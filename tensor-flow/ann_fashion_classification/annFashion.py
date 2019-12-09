import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

#load the data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#normalize the image. Each pixel can be one of 255 colors
X_train = X_train / 255.0
X_test = X_test / 255.0

# Since each image's dimension is 28x28, we reshape the full dataset to [-1 (all elements), height * width]
X_train = X_train.reshape(-1, 28*28)

X_train.shape

X_test = X_test.reshape(-1, 28*28)

#define a sequential model

model = tf.keras.models.Sequential()

#adding hidden layer: 128 newrons, ReLu activation function, 784 input shape
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))

#adding second hidden layer with 0.2 dropout
model.add(tf.keras.layers.Dropout(0.2))


#adding output layer: 
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.summary()

#now train the model
model.fit(X_train, y_train, epochs=5)


test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("accuracy of model is: {}".format(test_accuracy))


#serialize JSON