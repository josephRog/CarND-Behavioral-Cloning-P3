# Title: 	model.py
# Author: 	Joseph Rogers
# Self Driving Car Project 3

import os
import csv
import cv2
import numpy as np
import sklearn

correction = 0.5 # steering correction for multiple cameras

# Read all lines from CSV file
print("Starting Project 3 training!")
samples = []

print("Opening CSV file...")
with open('../sim_recording_files/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	
	for line in reader:
		samples.append(line)

print("done!")


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Generator function for reading out batches of images
def generator(samples, batch_size):
	num_samples = len(samples)

	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			measurements = []

			for batch_sample in batch_samples:
				source_path = batch_sample[0]
				filename = source_path.split('/')[-1]
				current_path = '../sim_recording_files/IMG/' + filename

				# Read steering measurements
				measurement = float(batch_sample[3])
				measurement_left = measurement + correction
				measurement_right = measurement - correction

				# Read images
				img_center = cv2.imread(batch_sample[0])
				img_left = cv2.imread(batch_sample[1])
				img_right = cv2.imread(batch_sample[2])

				# Add all measurements and images to lists
				images.append(img_center)
				images.append(img_left)
				images.append(img_right)
				measurements.append(measurement)
				measurements.append(measurement_left)
				measurements.append(measurement_right)

				# Add augmented data by mirroring all images and taking inverse steering
				images.append(cv2.flip(img_center,1))
				images.append(cv2.flip(img_left,1))
				images.append(cv2.flip(img_right,1))
				measurements.append(measurement*(-1.0))
				measurements.append(measurement_left*(-1.0))
				measurements.append(measurement_right*(-1.0))

			X_train = np.array(images)
			y_train = np.array(measurements)
			yield (sklearn.utils.shuffle(X_train, y_train))


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# Start of Neural Network
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import matplotlib.pyplot as plt # For plotting loss

# NN Model Using a slightly modified verions of the NVIDIA Architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5)) # Dropout after last convolution
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

# print the keys contained in the history object
print(history_object.history.keys())

# Save the model
model.save('model.h5')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

