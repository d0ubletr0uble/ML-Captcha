import cv2
from os import listdir


class Preprocessing:
    def __init__(self, directory):
        paths = map(lambda image: f'{directory}/{image}', listdir(directory))  # Get full paths
        self.backgrounds = list(map(lambda path: cv2.imread(path), paths))  # load all backgrounds

    def prepare(self, path_to_image):
        image = cv2.imread(path_to_image)
        image = self.remove_background(image)
        image = self.to_black_and_white(image)
        return image

    def remove_background(self, image):
        xor = map(lambda background: cv2.bitwise_xor(background, image), self.backgrounds)
        return min(xor, key=lambda x: x.sum())

    @staticmethod
    def to_black_and_white(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale first
        (_, black_white) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return black_white
