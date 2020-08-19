import torch.nn.functional as F

def nll_loss():
    loss = F.nll_loss
    return loss

def l2_loss():
    loss = F.mse_loss
    return loss


loss_collection = {'nll': nll_loss, 'l2': l2_loss}


def load_loss(config):
    loss_config = config['loss']
    loss = loss_collection[loss_config['loss_name']]()
    return loss
