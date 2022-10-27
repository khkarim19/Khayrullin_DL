import os
import numpy as np
import idx2numpy
from enum import Enum
from typing import List, Callable, Iterable


class DatasetType(Enum):
    train = 'train'
    valid = 'valid'
    test = 'test'


class Dataset:
    def __init__(self, dataset_type: DatasetType, transforms: List[Callable], num_classes: int,
                 dataset_path='./mnist',
                 image_file_name="train-images.idx3-ubyte",
                 label_file_name="train-labels.idx1-ubyte"):
        self.__dataset_path = dataset_path
        self.__dataset_type = dataset_type
        self.__image_file_name = image_file_name
        self.__label_file_name = label_file_name
        self.__labels = []
        self.__images = []
        self.__transforms = transforms
        self.__num_classes = num_classes

    def read_data(self):
        """
                Считывание данных и вывод статистики.
        """
        self.__labels = idx2numpy.convert_from_file(os.path.join(self.__dataset_path, self.__labels_file))
        self.__images = idx2numpy.convert_from_file(os.path.join(self.__dataset_path, self.__images_file))

        self.show_statistics()

    def __len__(self):
        """
        :return: Размер выборки
        """
        return len(self.__images)

    def one_hot_labels(self, label):
        """
        для 10 классов метка 5-> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        res = []
        if (abs(label)> self.__num_classes):
            raise ValueError('В датасете нет такого класса')

    def __getitem__(self, idx):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        images = self.images[idx]
        labels = self.labels[idx]
        for transform in self.transforms:
            images = transform(images)
        return images, labels

    def show_statistics(self):
        unique, counts = np.unique(self.__labels, return_counts=True)
        print(
            f'Датасет - {self.__dataset_type.name} \nКоличество элементов в датасете: {self.__len__()} \nКоличество классов: {self.__nrof_classes} \n{dict(zip(unique, counts))}\n')
