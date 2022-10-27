import os
import json
from datasets.MNIST_dataset import MNIST_Dataset
from datasets.MNIST_dataset import DatasetType
from dataloader.dataloader import Dataloader
from utils.config import Config


def run():
    cfg = Config.from_file("configs/config.json")
    train_ds = MNIST_Dataset(dataset_type=DatasetType.train, transforms=[],
                             num_classes=cfg.data.num_classes)
    test_ds = MNIST_Dataset(dataset_type=DatasetType.test, transforms=[],
                            num_classes=cfg.data.num_classes)
    train_ds.read_data()
    test_ds.read_data()

    train_dataloader = Dataloader(train_ds, cfg.train.batch_size, None, cfg.train.epoch_num)
    test_generator = Dataloader(test_ds, cfg.train.batch_size, None, cfg.train.epoch_num)

    next(train_dataloader.batch_generator())
    train_dataloader.show_batch()


if __name__ == '__main__':
    run()
