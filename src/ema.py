"""Karras-style EMA with warmup.

Effective decay at step t is ``min(target, (1+t) / (10+t))`` so the shadow
tracks the live model quickly at the start of training and asymptotes to the
high target decay once enough steps have accumulated. Critical for short
training runs where the standard 0.9999 decay would leave the shadow
essentially equal to the initialization.
"""
import copy
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, target_decay: float = 0.9999):
        self.target = target_decay
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = min(self.target, (1 + self.step) / (10 + self.step))
        for ep, p in zip(self.shadow.parameters(), model.parameters()):
            ep.data.mul_(d).add_(p.data, alpha=1 - d)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)
