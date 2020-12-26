import cv2
from image_preprocessing import Preprocessing
import numpy as np
from model import get_model
from keras_preprocessing.image import ImageDataGenerator
import itertools


def show_img(image):
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow("preview", image)
    cv2.waitKey(0)


preprocessing = Preprocessing('./Backgrounds', './Data', './labels.csv')
images = preprocessing.get_all_data()
model = get_model()

image = images[0]
# label = labels[:,:1]



test = []
test.append([0.])
from tensorflow.keras.utils import to_categorical
for digit in range(1,6):
    test.append(to_categorical(0, num_classes=10))
test.append(to_categorical(-1, num_classes=11))

# test = np.array([test]).reshape(1, 6, 1)

# label = label.reshape(1,6,1)
# print(label.shape)
# print(label.T)
print(model.output_shape)
image = image.reshape(1,60,200,1)
res = model.predict(image)
test = np.array([((0.)),((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)),((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)),((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)),((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)),((0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.))])
# print(test)
# exit()
model.fit(image, test, epochs=3)


