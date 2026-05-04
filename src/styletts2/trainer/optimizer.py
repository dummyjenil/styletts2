import torch
from torch.optim import AdamW


class MultiOptimizer:
    def __init__(self, optimizers=None, schedulers=None):
        self.optimizers = optimizers or {}
        self.schedulers = schedulers or {}
        self.keys = list(self.optimizers.keys())

    def state_dict(self):
        return [(key, self.optimizers[key].state_dict()) for key in self.keys]

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            if key in self.optimizers:
                self.optimizers[key].load_state_dict(val)

    def step(self, key=None):
        keys = [key] if key is not None else self.keys
        for k in keys:
            self.optimizers[k].step()
            if k in self.schedulers:
                self.schedulers[k].step()

    def zero_grad(self, key=None):
        keys = [key] if key is not None else self.keys
        for k in keys:
            self.optimizers[k].zero_grad()


def build_optimizer(parameters_dict, scheduler_params_dict, lr):
    optimizers = {}
    schedulers = {}

    for key, params in parameters_dict.items():
        opt = AdamW(params, lr=lr, weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9)
        optimizers[key] = opt

        s_params = scheduler_params_dict.get(key, {})
        schedulers[key] = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=s_params.get("max_lr", lr),
            epochs=s_params.get("epochs", 100),
            steps_per_epoch=s_params.get("steps_per_epoch", 1000),
            pct_start=s_params.get("pct_start", 0.0),
            div_factor=1,
            final_div_factor=1,
        )

    return MultiOptimizer(optimizers, schedulers)
