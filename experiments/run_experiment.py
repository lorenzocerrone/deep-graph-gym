import argparse

import yaml

from experiments.dataloaders.utils import load_dataset, load_data_loader
from experiments.models.utils import load_model
from experiments.trainer.utils import load_trainer


def arg_parser():
    parser = argparse.ArgumentParser(description='run an arbitrary experiment')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    return parser.parse_args()


def load_config(_path):
    with open(_path, 'r') as f:
        _config = yaml.safe_load(f)
    return _config


def main_run(_config):

    dataset = load_dataset(_config)
    loader = load_data_loader(_config)
    model = load_model(_config)
    trainer = load_trainer(_config)

    # start train
    trainer(model, dataset, loader)


if __name__ == '__main__':
    args = arg_parser()
    config = load_config(args.config)
    main_run(config)

