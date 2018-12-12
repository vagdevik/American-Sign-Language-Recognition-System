import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score

# dimensions of our images
image_size = 224

with open('model.json', 'r') as f:
    model = model_from_json(f.read())
    model.summary()
model.load_weights('model_weights.h5')

img_path = "D_396.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_category = model.predict(img_data)
print vgg16_category
category = np.argmax(vgg16_category,axis=1)
print category

X_test=np.load("test_set.npy")
Y_test=np.load("test_classes.npy")

Y_predict = model.predict(X_test)

acc_score = accuracy_score(Y_test, Y_predict)
print("Accuracy: "+str(acc_score))
