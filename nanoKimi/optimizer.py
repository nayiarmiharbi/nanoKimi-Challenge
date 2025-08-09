# Muon optimizer
# optimizer.py
import math
import torch
from torch.optim import Optimizer

class Muon(Optimizer):
    """A small experimental optimizer based on AdamW with per-parameter scaling.
       Not production — intended for experiments.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, scale_by_param_norm=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, scale_by_param_norm=scale_by_param_norm)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                update = exp_avg / denom
                if group['scale_by_param_norm']:
                    param_norm = p.data.norm().clamp(min=1e-6)
                    update = update / (param_norm)
                p.data.add_(update, alpha=-step_size)
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
        return loss
