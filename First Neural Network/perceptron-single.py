#importing libray keraas, which is a fantastic libray for building neural networks 'used extensively'
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

#setting up wandb for looking at results on websire
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

config.epochs = 10

# load data.... x = inputs and y = outputs
#X_train = 60,000 28 by 28 images or you can think like 60,000 28by28 array of integers from 0-255
#y_train = 60,000 labels which in this case are digits from 0-9
#y_test = 10,000 images we are gonna use to test our algorithm once we have trained it
#y_test = 10,000 more labels that correspond to the images in X_test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#for now we are only clasifyng 5s vs !5s , so we have to transform our ouput data into that
is_five_train = y_train == 5
is_five_test = y_test == 5
labels = ["Not Five", "Is Five"]

#getting width and height
#both are gonna be 28, because 28 by 28
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
#sequestial is the simplest and most important way to define a neural network in keras means our network is defines by a series of steps
model=Sequential()
#flattens our 2d 28 by 28 array into a single 784 lenght one-dimensional array
#we are telling the command the input is always going to be a 28 by 28 array
#important key in networks, the input size always has tobe the same.
model.add(Flatten(input_shape=(img_width,img_height)))
#adds a single perceptron to out network
#the layer is called dense because every input is connected to every output
#our model outputs 1 single number which is where the 1 coems from
model.add(Dense(1,activation='sigmoid'))
#optimeizer = how to change weights,
#adams gradiant decent funciton you dont have to specify the leraning rate, and it can really adapt to a wire radnge of cases
#metrics = settting it to outptu the accurarcy of our algortih as the algorithm learns
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
model.fit(X_train, is_five_train, epochs=config.epochs, validation_data=(X_test, is_five_test),
                    callbacks=[WandbCallback(labels=labels, data_type="image")])
