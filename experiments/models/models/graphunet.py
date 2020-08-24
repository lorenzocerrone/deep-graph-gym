from torch_geometric.nn import GraphUNet as _GraphUNet
from torch_geometric.utils import dropout_adj
import torch
import torch.nn.functional as F

class GraphUNet(torch.nn.Module):
    def __init__(self, config):
        super(GraphUNet, self).__init__()
        pool_ratios = [2000 / config['num_nodes'], 0.5]
        self.unet = _GraphUNet(config['num_input_features'], config['num_filters'], config['num_output_features'],
                              depth=3, pool_ratios=pool_ratios)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        edge_index, _ = dropout_adj(edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = F.dropout(x, p=0.92, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)