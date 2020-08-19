import argparse

import yaml

from deepgraphgym.run_experiment import main_run


def load_config(_path):
    with open(_path, 'r') as f:
        _config = yaml.safe_load(f)
    return _config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run an arbitrary experiment')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    main_run(config)


    print('ok')