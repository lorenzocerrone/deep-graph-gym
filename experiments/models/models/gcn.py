import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv  # noqa


class GCN(torch.nn.Module):
    def __init__(self, config):
        """
        Standard GCN for node classification following implementation is
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config['num_input_features'], config['num_filters'], cached=True,
                             normalize=True)
        self.conv2 = GCNConv(config['num_filters'], config['num_output_features'], cached=True,
                             normalize=True)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def parameters(self, **kwargs):
        return [dict(params=self.reg_params, weight_decay=5e-4),
                dict(params=self.non_reg_params, weight_decay=0)]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN4deep(torch.nn.Module):
    def __init__(self, config):
        """
        Standard GCN for node classification following implementation is
        """
        super(GCN4deep, self).__init__()
        self.conv1 = GCNConv(config['num_input_features'], config['num_filters'], cached=True,
                             normalize=True)
        self.conv2 = GCNConv(config['num_filters'], int(2 * config['num_filters']), cached=True,
                             normalize=True)
        self.conv3 = GCNConv(int(2 * config['num_filters']), config['num_output_features'], cached=True,
                             normalize=True)

        self.reg_params1 = self.conv1.parameters()
        self.reg_params2 = self.conv2.parameters()

        self.non_reg_params = self.conv3.parameters()

    def parameters(self, **kwargs):
        return [dict(params=self.reg_params1, weight_decay=5e-4),
                dict(params=self.reg_params2, weight_decay=5e-4),
                dict(params=self.non_reg_params, weight_decay=0)]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
