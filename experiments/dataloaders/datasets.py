from torch_geometric.datasets import Planetoid, QM9, MNISTSuperpixels
from deepgraphgym.dataloaders.transformers import transforms_collection


def cora(config):
    transform = transforms_collection[config['transform']]
    dataset = Planetoid(config['path'],
                        config['dataset_name'],
                        transform=transform())
    return dataset


def qm9(config):
    transform = transforms_collection[config['transform']]
    dataset = QM9(config['path'], transform=transform())
    return dataset


def mnist_superpixels(config):
    transform = transforms_collection[config['transform']]
    dataset = MNISTSuperpixels(config['path'], transform=transform())
    return dataset


dataset_collection = {'Cora': cora, 'QM9': qm9, 'MNISTSuperpixels': mnist_superpixels}
