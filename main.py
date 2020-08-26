import cv2
from image_preprocessing import Preprocessing
import numpy as np
from model import get_model
from keras.preprocessing.image import ImageDataGenerator
import itertools


def show_img(image):
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow("preview", image)
    cv2.waitKey(0)


preprocessing = Preprocessing('./Backgrounds')
images = preprocessing.load_and_prepare_data(f'./Data')
for i in images:
    show_img(i)


#img = np.array([img])
#print(img.shape)

#model = get_model()
#model.summary()
#print(model.predict(img))
