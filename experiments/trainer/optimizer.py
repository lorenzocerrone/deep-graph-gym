import torch


def adam(config, model):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config['lr']),
                                 weight_decay=float(config['weight_decay']))
    return optimizer


optimizer_collection = {'Adam': adam}


def load_optimizer(config, model):
    optimizer_config = config['optimizer']
    optimizer = optimizer_collection[optimizer_config['optimizer_name']](optimizer_config, model)
    return optimizer
