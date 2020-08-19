from abc import ABC

import torch

from deepgraphgym.trainer.loss import load_loss
from deepgraphgym.trainer.optimizer import load_optimizer


class GenericTrainer:
    def __init__(self, config):
        self.config = config

        self.number_epoch = self.config['number_epoch']
        self.val_frequency = config['val_frequency']
        self.optimizer = None
        self.loss = None
        self.model, self.dataset = None, None

    def __call__(self, model, dataset, loader):

        self.model, self.dataset = model, dataset
        self.loader = loader(dataset)
        self.loader_type = self.loader['type']

        self.init_optimizer()
        self.init_loss()

        for epoch in range(self.number_epoch):
            self.train_iteration(epoch)

    def init_optimizer(self):
        self.optimizer = load_optimizer(self.config, self.model)

    def init_loss(self):
        self.loss = load_loss(self.config)

    def train_iteration(self, epoch):
        raise NotImplementedError

    def val_iteration(self, epoch, iteration):
        raise NotImplementedError


class SimpleMask(GenericTrainer):

    def __init__(self, config):
        super(SimpleMask, self).__init__(config)

    def train_iteration(self, epoch):
        for it, data in enumerate(self.dataset):
            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

            if it % self.val_frequency == 0:
                self.val_iteration(epoch, it)

    @torch.no_grad()
    def val_iteration(self, epoch, it):
        self.model.eval()
        data = self.dataset[0]
        output = self.model(data)[data.val_mask]
        output_train = self.model(data)[data.train_mask]

        print("train", output_train.max(1)[1].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item())
        print("val", output.max(1)[1].eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item())


class Simple(GenericTrainer, ABC):
    def __init__(self, config):
        super(Simple, self).__init__(config)
        self.training_steps = {'split_data': self._train_split,
                               'masked_data': self._train_masked}

        self.val_steps = {'split_data': self._val_split,
                          'masked_data': self._val_masked}

    def train_iteration(self, epoch):
        self.training_steps[self.loader_type](epoch)

    def val_iteration(self, epoch, iteration):
        self.val_steps[self.loader_type](epoch, iteration)

    def _train_split(self, epoch):
        self.train_loader = self.loader['train_loader']
        for it, data in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(data)
            output = torch.mean(output, 0, keepdim=True)
            loss = self.loss(output, data.y)
            loss.backward()
            self.optimizer.step()

            if it % self.val_frequency == 0:
                self._val_split(epoch, it)

    def _train_masked(self, epoch):
        pass

    @torch.no_grad()
    def _val_split(self, epoch, iteration):
        self.model.eval()
        val_loader = self.loader['val_loader']
        mean_val = 0
        for it, data in enumerate(val_loader):
            output = self.model(data)
            output = torch.mean(output, 0, keepdim=True)
            mean_val += self.loss(output, data.y).item()

        print(f"Val at epoch {epoch}, it {iteration}, avg-loss {mean_val/it}")
        pass

    @torch.no_grad()
    def _val_masked(self, epoch, it):
        pass


trainer_collection = {'SimpleMask': SimpleMask, 'Simple': Simple}

