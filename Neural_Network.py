# https://www.geeksforgeeks.org/python-image-classification-using-keras/

# Importing all necessary libraries
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import shutil
import os

# Directories of files
img_width, img_height = 224, 224
parent_fldr = os.path.dirname(os.path.realpath(__file__))       # Get folder where this script is
train_fldrs = os.listdir(parent_fldr + "/trainval")               # Get all of the folders containing images
test_fldrs = os.listdir(parent_fldr + "/test")               # Get all of the folders containing images

train_images = parent_fldr + "/train_images"
test_images = parent_fldr + "/test_images"

# Process images out of randomly named folders
idx = 0
# if not os.path.exists(train_images):
# 	os.makedirs(train_images)
# 	for folder in train_fldrs:
# 		idx += 1
# 		curr_dir = parent_fldr + "/trainval/" + folder
# 		files = os.listdir(curr_dir) # List of all images in that folder
# 		images = [s for s in files if "jpg" in s]
# 		bbox = [s for s in files if "bbox" in s]
# 		cloud = [s for s in files if "cloud" in s]
# 		proj = [s for s in files if "proj" in s]

# 		for image in images:
# 			old_dir = curr_dir + "/" + image
# 			new_dir = train_images + "/" + str(idx) + image
# 			shutil.copyfile(old_dir, new_dir)

#nb_train_samples = len(images)

idx = 0
if not os.path.exists(test_images):
	os.makedirs(test_images)
	for folder in test_fldrs:
		idx += 1
		curr_dir = parent_fldr + "/test/" + folder
		files = os.listdir(curr_dir) # List of all images in that folder
		images = [s for s in files if "jpg" in s]
		bbox = [s for s in files if "bbox" in s]
		cloud = [s for s in files if "cloud" in s]
		proj = [s for s in files if "proj" in s]

		for image in images:
			old_dir = curr_dir + "/" + image
			new_dir = test_images + "/" + folder + "_" + image[0:4] + ".jpg"
			shutil.copyfile(old_dir, new_dir)

nb_validation_samples = len(images)
# Parameters to tune for our project
epochs = 10
batch_size = 30

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)


# Make the NN
# Adjust these to attain performance requirement
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64)) # Makes this a fully connected model
model.add(Activation('relu'))
model.add(Dropout(0.5)) # avoids overfitting
model.add(Dense(1)) # output layer - single neuron 
model.add(Activation('sigmoid'))

# Might want model.fit instead
model.compile(loss='binary_crossentropy',
			optimizer='rmsprop',
			metrics=['accuracy'])

print("Fit model on training data")
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))


# Create the training set, testing set, and validation set
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
	train_images,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
	test_images,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='binary')

model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size)

model.save_weights('car_classification_model.h5')
