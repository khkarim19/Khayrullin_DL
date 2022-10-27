from abc import ABC, abstractmethod
import numpy as np


class Transformations(ABC):
    @abstractmethod
    def __call__(self, images):
        pass


class Normalize(Transformations):
    def __init__(self, mean=128, var=255):
        """
            :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
            :param var (int): значение, на которое необходимо поделить.
                """
        self.__mean = mean
        self.__var = var

    def __call__(self, images):
        return (images - self.__mean) / self.__var


class View(Transformations):
    def __init__(self):
        pass

    def __call__(self, images):
        for image in images:
            image = np.array(image, dtype='uint8')
            image = image.reshape((28, 28))
        return images
