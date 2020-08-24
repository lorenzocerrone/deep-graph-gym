from experiments import model_key
from experiments.models.models import models_collection


def load_model(config):
    model_config = config[model_key]

    model = models_collection[model_config['model_name']](model_config)
    return model
