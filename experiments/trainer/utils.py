from experiments import trainer_key
from experiments.trainer.trainer import trainer_collection


def load_trainer(config):
    trainer_config = config[trainer_key]
    trainer = trainer_collection[trainer_config['trainer_name']](trainer_config)
    return trainer
