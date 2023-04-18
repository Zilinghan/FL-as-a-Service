import logging

from collections import OrderedDict
from .algorithm import BaseServer, BaseClient
from .server_federated import FedServer

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import copy

class AsyncFedServer(FedServer):
    def __init__(self, weights, model, loss_fn, num_clients, device, global_step = 0, **kwargs):
        super(AsyncFedServer, self).__init__(weights, model, loss_fn, num_clients, device, **kwargs)
        self.global_step = global_step

    def compute_pseudo_gradient(self, clinet_idx):
        for name, _ in self.model.named_parameters():
            self.pseudo_grad[name] = torch.zeros_like(self.model.state_dict()[name])
            self.pseudo_grad[name] += self.weights[clinet_idx] * (self.global_state[name]-self.primal_states[clinet_idx][name])

    def primal_residual_at_server(self, client_idx: int) -> float:
        primal_res = 0
        for name, _ in self.model.named_parameters():
            primal_res += torch.sum(torch.square(self.global_state[name]-self.primal_states[client_idx][name].to(self.device)))
        self.prim_res = torch.sqrt(primal_res).item()

    def update(self, local_states: OrderedDict, init_step: int, client_idx: int):  
        # Obtain the global and local states
        self.global_state = copy.deepcopy(self.model.state_dict())
        super(FedServer, self).primal_recover_from_local_states(local_states)
        # Calculate residual
        self.primal_residual_at_server(client_idx)
        # Change device
        for name, _ in self.model.named_parameters():
            self.primal_states[client_idx][name] = self.primal_states[client_idx][name].to(self.device)
        # Global state computation
        self.compute_step(init_step, client_idx)
        for name, _ in self.model.named_parameters():
            self.global_state[name] += self.step[name]
        # Model update
        self.model.load_state_dict(self.global_state)
        # Global step update
        self.global_step += 1