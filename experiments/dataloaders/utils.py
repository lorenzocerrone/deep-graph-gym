from experiments import dataset_key, loader_key
from experiments.dataloaders.datasets import dataset_collection
from experiments.dataloaders.dataloader import Dataset2DataLoader


def load_dataset(config):
    dataset_config = config[dataset_key]
    dataset = dataset_collection[dataset_config['dataset_name']](dataset_config)
    return dataset


def load_data_loader(config):
    loader_config = config[loader_key]
    loader = Dataset2DataLoader(loader_config)
    return loader
