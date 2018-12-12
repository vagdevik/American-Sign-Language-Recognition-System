import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np
from keras.applications import VGG16
from keras.preprocessing import image
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
from keras.applications.vgg16 import preprocess_input

print "Imported Modules..."

with open('Model/model.json', 'r') as f:
	model = model_from_json(f.read())
	model.summary()
model.load_weights('Model/model_weights.h5')


scaling_factorx=1.5
scaling_factory=1.5
image_size = 224

# 0 if laptop cam, 1 if external cam
cap = cv2.VideoCapture(0)

while (cv2.waitKey(10) & 0xff)!=27:
	ret, frame = cap.read() # read and capture frames
	box = cv2.rectangle(frame, (100, 100), (400, 400),(0,255,0), 5)  #green box
	frame=cv2.resize(frame,None,fx=scaling_factorx,fy=scaling_factory,interpolation=cv2.INTER_AREA) #resize window
	frame = cv2.flip( frame, 1 ) #flip image
	cv2.imshow('video output',frame) #display image	
		
	frame = cv2.resize(frame, (image_size, image_size)) 
	img_data = image.img_to_array(frame)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)

	vgg16_category = model.predict(img_data)
	print vgg16_category
	category = np.argmax(vgg16_category,axis=1)
	print category
cap.release()
cv2.destroyAllWindows()

