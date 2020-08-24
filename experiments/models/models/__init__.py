from experiments.models.models.gcn import GCN, GCN4deep
from experiments.models.models.graphunet import GraphUNet
from experiments.models.models.gcnwalk import GCNWalk, GCN4deepWalk

models_collection = {"GCN": GCN,
                     "GCN4deep": GCN4deep,
                     "GraphUNet": GraphUNet,
                     "GCNWalk": GCNWalk,
                     "GCN4deepWalk": GCN4deepWalk}
