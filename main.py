import cv2
from image_preprocessing import Preprocessing
import pytesseract


def show_img(image):
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow("preview", image)
    cv2.waitKey(0)


preprocessing = Preprocessing('./Backgrounds')
for i in range(1, 11):
    img = preprocessing.prepare(f'./Data/{i}.png')
    show_img(img)
