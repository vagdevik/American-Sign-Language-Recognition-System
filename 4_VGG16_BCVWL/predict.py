import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
#from keras.applications import VGG16
#from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score

# dimensions of our images
image_size = 224

with open('Model/model.json', 'r') as f:
    model = model_from_json(f.read())
    model.summary()
model.load_weights('Model/model_weights.h5')
#model.load_weights('Weights/weights.15-0.00.hdf5')

X_test=np.load("Numpy/test_set.npy")
Y_test=np.load("Numpy/test_classes.npy")

Y_predict = model.predict(X_test)

#print(type(Y_test))
#print(type(Y_predict))

#print(Y_test.shape)
#print(Y_predict.shape)

#print(Y_predict)
#print(Y_test[0])

#print(X_test[6])

Y_predict = [np.argmax(r) for r in Y_predict]
Y_test = [np.argmax(r) for r in Y_test]

#print(Y_predict)
#print(Y_test)
print("##################")
acc_score = accuracy_score(Y_test, Y_predict)
print("Accuracy: "+str(acc_score))
print("##################")

