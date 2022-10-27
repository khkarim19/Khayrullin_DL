from datasets.dataset import Dataset
from datasets.dataset import DatasetType


def run():
    # trainDS = Dataset('train',1,1)
    # trainDS.read_data('train')

    testDS = Dataset('test', 1, 1)
    testDS.read_data('test')



if __name__ == '__main__':
    run()