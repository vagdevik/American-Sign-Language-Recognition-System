import numpy as np
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications import VGG16 
from keras.preprocessing import image
from keras.models import model_from_json
from keras import models,layers,optimizers
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input
from keras.layers import  AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge
from keras.layers import Input, Dense, Reshape, Activation

#Load the VGG model
image_size=224
vgg_base = VGG16(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))

#initiate a model
model = models.Sequential()

#Add the VGG base model
model.add(vgg_base)

#Add new layers
model.add(layers.Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

#summary of the model
Adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])

#get train and validation sets
X_train=np.load("train_set.npy")
Y_train=np.load("train_classes.npy")

X_valid=np.load("validation_set.npy")
Y_valid=np.load("validation_classes.npy")

model.fit(X_train/255.0, Y_train, epochs=12, batch_size=32,validation_data=(X_valid/255.0,Y_valid))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Saved model to disk")

#test the model
img_path = "C_396.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_category = model.predict(img_data)
category = np.argmax(vgg16_category,axis=1)
#print vgg16_category.shape
print category
