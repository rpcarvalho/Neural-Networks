import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


import matplotlib.pyplot as plt
import cv2
import numpy as np
# Data loading and preprocessing
##import tflearn.datasets.mnist as mnist
##X, Y, testX, testY = mnist.load_data(one_hot=True)
##X = X.reshape([-1, 28, 28, 1])
##testX = testX.reshape([-1, 28, 28, 1])

#cv2.imshow('median',testX[0])

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')

network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)

network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)

network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
##model = tflearn.DNN(network, tensorboard_verbose=0)
##model.fit({'input': X}, {'target': Y}, n_epoch=3,
##           validation_set=({'input': testX}, {'target': testY}),
##           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
##
##model.save('quicktest.model')

def test(n = 1):
    prediction = np.round(model.predict([testX[n]])[0])
    i = np.where(prediction == max(prediction))
    print(i[0][0])
    cv2.imshow('IMG',testX[n])
    img = cv2.cvtColor(testX[n], cv2.COLOR_GRAY2BGR)
    cv2.imwrite("./paint.jpg",testX[n])

def testPic():
    img = cv2.imread("./paint.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('IMG',img)
    img = img.reshape([28, 28, 1])
    cv2.imshow('IMG',img)
    prediction = np.round(model.predict([img])[0])
    i = np.where(prediction == max(prediction))
    print(i[0][0])
    
    
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('quicktest.model')


testPic()
