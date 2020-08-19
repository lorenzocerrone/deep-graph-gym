from torch_geometric.data import Data, DataLoader
import numpy as np


class Dataset2DataLoader:
    def __init__(self, config):
        self.loaders_type = config['loader_type']
        self.loader_collection = {'masked_data': self.masked_data,
                                  'create_split': self.create_split}
        self.config = config
        self.batch_size, self.split_ratios, self.random_seed = None, None, None

    def __call__(self, dataset):
        return self.loader_collection[self.loaders_type](dataset)

    @staticmethod
    def masked_data(_dataset):
        meta = {'num_features': _dataset.num_features,
                'num_classes': _dataset.num_classes}

        loader = {'type': 'masked_data',
                  'loader': _dataset,
                  'meta': meta}
        return loader

    def create_split(self, _dataset):
        self.batch_size = self.config['batch_size']
        self.random_seed = self.config['random_seed']
        self.split_ratios = self.config['split_ratios']

        meta = {'num_features': _dataset.num_features,
                'num_classes': _dataset.num_classes}

        mask = np.arange(0, len(_dataset))

        np.random.seed(self.random_seed)
        np.random.shuffle(mask)
        mask = list(mask)

        n_train, n_val, _ = [int(i * len(_dataset)) for i in self.split_ratios]

        train = _dataset[mask[:n_train]]
        val = _dataset[mask[n_train:n_train+n_val]]
        test = _dataset[mask[n_train+n_val:]]

        train_loader = DataLoader(train, batch_size=self.batch_size)
        val_loader = DataLoader(val, batch_size=self.batch_size)
        test_loader = DataLoader(test, batch_size=self.batch_size)

        loader = {'type': 'split_data',
                  'train_loader': train_loader,
                  'val_loader': val_loader,
                  'test_loader': test_loader,
                  'meta': meta}

        return loader
