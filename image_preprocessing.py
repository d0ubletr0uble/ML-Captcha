import itertools

import cv2
from os import listdir
import numpy as np

from keras_preprocessing.image import ImageDataGenerator


class Preprocessing:
    def __init__(self, directory):
        paths = map(lambda image: f'{directory}/{image}', listdir(directory))  # Get full paths to backgrounds
        self.backgrounds = list(map(lambda path: cv2.imread(path), paths))  # load all backgrounds

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

    @staticmethod
    def get_randomly_rotated_copies(image, count):
        image = image.reshape(1, 60, 200, 1)
        generator = ImageDataGenerator(rotation_range=15)
        images = generator.flow(image, batch_size=1)
        images = itertools.islice(images, 10)
        return np.array(map(lambda im: im.reshape(60, 200, 1), images))

    def load_and_prepare_data(self, directory):
        paths = map(lambda image: f'{directory}/{image}', listdir(directory))  # Get full paths to images
        images = map(lambda path: self.prepare(path), paths)  # Remove background and turn to black/white
        images = list(map(lambda img: self.get_randomly_rotated_copies(img, 10), images))
        images = np.stack(images)
        result = []
        for augmented in images:
            for image in augmented:
                result.append(image)

        return np.array(result)
