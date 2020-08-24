from torch_geometric.datasets import Planetoid, QM9, MNISTSuperpixels
from experiments.dataloaders.transformers import transforms_collection


def _planetoid(config):
    transform = transforms_collection[config['transform']]
    dataset = Planetoid(config['path'],
                        config['dataset_name'],
                        transform=transform())
    return dataset


def cora(config):
    return _planetoid(config)


def citeseer(config):
    return _planetoid(config)


def pubmed(config):
    return _planetoid(config)


def qm9(config):
    transform = transforms_collection[config['transform']]
    dataset = QM9(config['path'], transform=transform())
    return dataset


def mnist_superpixels(config):
    transform = transforms_collection[config['transform']]
    dataset = MNISTSuperpixels(config['path'], transform=transform())
    return dataset


dataset_collection = {'Cora': cora,
                      'Citeseer': citeseer,
                      'Pubmed': pubmed,
                      'QM9': qm9,
                      'MNISTSuperpixels': mnist_superpixels}
