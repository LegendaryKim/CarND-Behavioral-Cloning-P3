import csv
import cv2
import numpy as np

### Hyperparameter
batch_size = 32
epoch = 30
streering_correction = 0.25
droppingprob = 0.5
lowsteering = 0.01
translation = 20


lines = []

# Udacity data
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)  # skip the headers
	for line in reader:
		lines.append(line)


### Randomly reducing low steering angle to remove bias
def randomly_del_low_steering_data(lines, prob):
	measurements = []
	for line in lines:
		measurement = float(line[3])
		measurements.append(measurement)
	indices = [index for index,value in enumerate(measurements) if abs(value) < lowsteering]

	rows = [index for index in indices if np.random.uniform() < prob]
	for index in rows[::-1]:
		del(lines[index])
	return lines

import sklearn
from sklearn.model_selection import train_test_split
lines = sklearn.utils.shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print('total:' + str(len(lines)))
print('train:' + str(len(train_samples)))

train_samples = randomly_del_low_steering_data(train_samples, droppingprob)
 
print('train (dropout):' + str(len(train_samples)))
print('validation:' + str(len(validation_samples)))

def AugmentationFlipping(images, measurements, image, measurement):
	return images.append(cv2.flip(image,1)), measurements.append(measurement*-1.0)


def AugmentationRandomTranslation(images, measurements, image, measurement, numtrans):
	for i in range(numtrans):
		rows, cols = image.shape[:2]
		uniftranslation = int(np.random.uniform(-translation, translation))
		translation_matrix = np.float32([ [1,0,uniftranslation], [0,1,0] ])
		translation_angle = uniftranslation/100.
		img_translation = cv2.warpAffine(image, translation_matrix, (cols, rows))
		images.append(img_translation), measurements.append(measurement + translation_angle)

	return images, measurements


### Applying Generator (memory efficient)
def generator(samples, batch_size=32):
	num_samples = len(samples)
	samples = sklearn.utils.shuffle(samples)	
	while 1: # Loop forever so the genrator never terminates
		for offset in range(0, num_samples, batch_size):
			images = []
			measurements = []
			batch_samples = samples[offset : offset + batch_size]
			for batch_sample in batch_samples:
				# Center Cameras
				source_path = batch_sample[0] 
				filename = source_path.split('/')[-1]				
				current_path = '../data/IMG/' + filename
				image = cv2.imread(current_path)
				images.append(image)
				measurement = float(batch_sample[3])
				measurements.append(measurement)

				# translation
				# AugmentationRandomTranslation(images, measurements, image, measurement, 3)

				# Augmentation
				AugmentationFlipping(images, measurements, image, measurement)

				correction = streering_correction # correction to steering measurements for side camera images
				# Left Cameras
				source_path = batch_sample[1] 
				filename = source_path.split('/')[-1]				
				current_path = '../data/IMG/' + filename
				image = cv2.imread(current_path)
				images.append(image)
				measurements.append(measurement + correction)

				# translation
				# AugmentationRandomTranslation(images, measurements, image, measurement + correction, 3)				

				# Augmentation
				AugmentationFlipping(images, measurements, image, measurement)							

				# Right Cameras
				source_path = batch_sample[2] 
				filename = source_path.split('/')[-1]				
				current_path = '../data/IMG/' + filename
				image = cv2.imread(current_path)
				images.append(image)
				measurements.append(measurement - correction)	

				# translation
				# AugmentationRandomTranslation(images, measurements, image, measurement - correction, 3)				

				# Augmentation
				AugmentationFlipping(images, measurements, image, measurement)				


			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(measurements)
			# yield sklearn.utils.shuffle(X_train, y_train)
			yield sklearn.utils.shuffle(X_train, y_train)


### compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

### parameter: number of training and validation samples per epoch
samples_per_epoch = (len(train_samples)*2*3//batch_size)*batch_size
nb_val_samples = (len(validation_samples)*2*3//batch_size)*batch_size

# sample 3 batches from the generator
for i in range(1):
    x_batch, y_batch = next(train_generator)
    print(x_batch.shape, y_batch.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# resize the image to 64x64
def resize_function(input):
	from keras.backend import tf as ktf
	resized = ktf.image.resize_images(input, (64, 64))
	return resized

# normalize the image	
def normalization_function(input):
	resized = input / 255.0 - 0.5
	return resized

# NVIDIA Model
model = Sequential()

model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3))) # ((from top_crop, from bottom_crop), (from left_crop, from right_crop))
model.add(Lambda(resize_function))
model.add(Lambda(normalization_function))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu')) # nb_filter, nb_stride_row, nb_stride_col
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu')) # nb_filter, nb_stride_row, nb_stride_col
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu')) # nb_filter, nb_stride_row, nb_stride_col
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation='relu')) # nb_filter, nb_stride_row, nb_stride_col
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation='relu')) # nb_filter, nb_stride_row, nb_stride_col
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))



model.compile(loss = 'mse', optimizer = 'adam')

history_object = model.fit_generator(train_generator, samples_per_epoch = samples_per_epoch, 
	validation_data = validation_generator, nb_val_samples=nb_val_samples, nb_epoch = epoch, verbose = 1)

### print the keys contained in the history object
print(history_object.history.keys())

model.save('model_gen_30_32_08_025_001_resize.h5')
print("Model Saved!!")

import matplotlib
matplotlib.use('Agg') # Generate images without having a window appear
import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('figure_new_30_32_05_025_001_resize.png')
# plt.show()

