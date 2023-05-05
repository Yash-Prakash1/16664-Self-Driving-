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
