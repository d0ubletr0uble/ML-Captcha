import cv2
from os import listdir
import numpy as np
from tensorflow.keras.utils import to_categorical


class Preprocessing:
    def __init__(self, background_dir, data_dir, labels):
        paths = map(lambda image: f'{background_dir}/{image}', listdir(background_dir))  # Get full paths to backgrounds
        self.backgrounds = list(map(lambda path: cv2.imread(path), paths))  # load all backgrounds
        self.labels = np.loadtxt(labels, dtype='str', delimiter=',')[:, 1]
        self.data_dir = data_dir

    def prepare(self, path_to_image):
        image = cv2.imread(path_to_image)
        image = self.remove_background(image)
        image = self.to_black_and_white(image)
        image = image.reshape(60, 200, 1)
        return image

    def remove_background(self, image):
        xor = map(lambda background: cv2.bitwise_xor(background, image), self.backgrounds)
        return min(xor, key=lambda x: x.sum())

    @staticmethod
    def to_black_and_white(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale first
        (_, black_white) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return black_white

    def load_and_prepare_data(self):
        paths = map(lambda image: f'{self.data_dir}/{image}', listdir(self.data_dir))  # Get full paths to images
        images = map(lambda path: self.prepare(path), paths)  # Remove background and turn to black/white
        return np.array(list(images))

    def get_label_data_prepared(self):
        #  result shape (1, 1, 1) and (4, 1, 10) and (1, 1, 11) x100 times
        #              (detected?)   (4 numbers)  (maybe 5th number)

        result = [[] for i in range(6)]
        results = []

        for label in self.labels:
            if label == 'N':
                result.append(0.)
                for digit in range(1,5):
                    result[digit].append(to_categorical(0, num_classes=10))
                result[5].append(to_categorical(-1, num_classes=11))
            else:
                result.append(1.)
                for digit in range(1,5):
                    result[digit].append(to_categorical(int(label[digit]), num_classes=10))
                if label[4] == 'a':
                    result[5].append(to_categorical(10, num_classes=11))
                else:
                    result[5].append(to_categorical(int(label[4]), num_classes=11))
            results.append(result)

        results = np.array(results)
        return results

    def get_all_data(self):
        return self.load_and_prepare_data()#, self.get_label_data_prepared()