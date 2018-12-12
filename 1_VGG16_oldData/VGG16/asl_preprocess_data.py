import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import random
import numpy as np
from random import shuffle
from keras.utils import np_utils

class_dic = {"A":0,"B":1,"C":2,"D":3}

image_list = []
image_class = []

data_folder_path = "ASL_Data_sample"

files = os.listdir(data_folder_path)

for i in range(10):
	shuffle(files)

count = 0

for file_name in files:
	'''if count<10:
		print file_name
		count+=1'''
	#print(file_name)
	path = data_folder_path+'/'+file_name
	image = cv2.imread(path)
	resized_image = cv2.resize(image,(224,224))
	#print("*********")
	image_list.append(resized_image)
	image_class.append(class_dic[file_name[0]])
	
image_class = np_utils.to_categorical(image_class)

total_dataset_len = len(image_list)
	
#print len(image_list)
#print len(image_class)

#print image_class[:10]

#print "\n########\n"

train_len = int(total_dataset_len*0.7)
validation_len = train_len + int(total_dataset_len*0.2)
test_len = train_len + validation_len + int(total_dataset_len*0.1)

train_images = image_list[:train_len]
train_classes = image_class[:train_len]
#print len(train_images)

validation_images = image_list[train_len:validation_len]
validation_classes = image_class[train_len:validation_len]
#print len(validation_images)

test_images = image_list[validation_len:test_len]
test_classes = image_class[validation_len:test_len]
#print len(test_images)

np.save('train_set.npy',train_images)
np.save('train_classes.npy',train_classes)

np.save('validation_set.npy',validation_images)
np.save('validation_classes.npy',validation_classes)

np.save('test_set.npy',test_images)
np.save('test_classes.npy',test_classes)

print "Data pre-processing Success!"
