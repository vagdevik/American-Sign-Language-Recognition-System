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

model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

#summary of the model
#Adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint("Checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

#get train and validation sets
X_train=np.load("Numpy/train_set.npy")
Y_train=np.load("Numpy/train_classes.npy")

X_valid=np.load("Numpy/validation_set.npy")
Y_valid=np.load("Numpy/validation_classes.npy")

model.fit(X_train/255.0, Y_train, epochs=12, batch_size=32,validation_data=(X_valid/255.0,Y_valid), shuffle=True, callbacks=[checkpoint])

# serialize model to JSON
model_json = model.to_json()
with open("Model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Model/model_weights.h5")
print("Saved model to disk")
