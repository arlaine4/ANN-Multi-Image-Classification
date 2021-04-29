import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#The goal of this program is to classify images
#	as one of the following classes defined below

#Each class correspond to one data type
class_names = ['0 Top/T-shirt',
				'1 Trouser',
				'2 Pullover',
				'3 Dress',
				'4 Coat',
				'5 Sandal',
				'6 Shirt',
				'7 Sneaker',
				'8 Bag',
				'9 Ankle boot']

def	normalize_and_flatten_data(train, test):
	#We normalize the data by dividing it by 255.0
	#	because we are talking about pictures
	#	we normalize the data with the images dimensions
	#	so we can just store the pixels activations
	#	values with a float ranging from 0.0 to 1.0
	#		0.0 -> not 'activated' at all
	#		1.0 -> full pixel 'activation'
	train = train / 255.0
	test = test / 255.0

	#Flattening dataset
	#	instead of having a 2 dimensions matrix
	#	we flip it to one dimensions so its easier to pass
	#	the data to the model
	train = train.reshape(-1, 28*28)
	test = test.reshape(-1, 28*28)
	return train, test

def	build_model():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
		metrics=['sparse_categorical_accuracy'])
	print("\nModel built and compiled, informations about its structure below..\n")
	model.summary()
	return model

def	plot_confusion_matrix(y_test, yhat):
	#Plotting confusion matrix
	#	creating it -> stored inside cm
	cm = confusion_matrix(y_test, yhat)
	df_cm = pd.DataFrame(cm, index = [i for i in class_names],
							columns = [i for i in class_names])
	plt.figure(figsize=(12,8))
	sn.heatmap(df_cm, annot=True)
	plt.show()

if __name__ == "__main__":
	#Loading data, splitting it into train and test set
	#x is the data given to the network while y corresponds to the labels
	#	associated with each x
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

	#Data preparation
	x_train, x_test = normalize_and_flatten_data(x_train, x_test)

	#Model building and compilation
	model = build_model()

	#Actual training process
	model.fit(x_train, y_train, epochs=30, verbose=2)

	#Evaluating the model
	print("\n\n")
	model.evaluate(x_test, y_test)

	#Model prediction
	yhat = np.argmax(model.predict(x_test), axis=-1)
	print("Prediction precision over {} elems : {}%".format(x_test.shape[0], 
		(np.sum(yhat == y_test) / y_test.shape[0]) * 100))

	#Confusion matrix
	plot_confusion_matrox(y_test, y_hat)
