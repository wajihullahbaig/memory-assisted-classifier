import time
from dataclasses import replace

import torch

from .data import select
from .evaluate import accuracy, collect, fill_memory, sync
from .model import EarlyExitClassifier
from .train import train


def split_acc(pred, y, old):
    """Accuracy (%) on old classes, new classes and all classes."""
    is_old = torch.isin(y, torch.tensor(old, device=y.device))
    return {"old": accuracy(pred[is_old], y[is_old]), "new": accuracy(pred[~is_old], y[~is_old]), "all": accuracy(pred, y)}


def lookalikes(pred, y, new):
    """For each new class: the class its images are most often predicted as, and how often (%)."""
    result = {}
    for c in new:
        p = pred[y == c]
        top = p.bincount().argmax().item()
        result[c] = (top, (p == top).float().mean().item() * 100)
    return result


def train_base(name, cfg, data, old, log):
    """A fresh model trained on the old classes only."""
    torch.manual_seed(cfg.seed)
    model = EarlyExitClassifier(cfg).to(data["test"].x.device)
    train(model, {k: select(data[k], old) for k in ("train", "memory", "val")}, cfg, name, log)
    return model


def insert(model, data, old, new, shots, per_class):
    """Add `shots` images of each new class to a memory of old classes. No training; memory-only prediction."""
    fill_memory(model, select(data["memory"], old, per_class))
    device = data["test"].x.device
    sync(device)
    start = time.perf_counter()
    fill_memory(model, select(data["memory"], new, shots), reset=False)
    sync(device)
    seconds = time.perf_counter() - start
    out = collect(model, data["test"])
    return {"shots": shots, "steps": 0, "seconds": seconds, **split_acc(out["mem"], out["y"], old),
            "lookalikes": lookalikes(out["mem"], out["y"], new)}


def finetune(model, data, cfg, new, old, steps, lr, log):
    """Fine-tune the whole network on new-class images only (same loss and loop as training)."""
    new_data = {k: select(data[k], new, cfg.memory_size // 10) for k in ("train", "memory", "val")}
    ft_cfg = replace(cfg, max_steps=steps, eval_every=steps, lr=lr)
    device = data["test"].x.device
    sync(device)
    start = time.perf_counter()
    train(model, new_data, ft_cfg, "finetune", log)
    sync(device)
    seconds = time.perf_counter() - start
    out = collect(model, data["test"])
    return {"steps": steps, "seconds": seconds, **split_acc(out["full"], out["y"], old)}


def joint(cfg, data, old, ckpt, log):
    """Upper bound: a model trained on all classes (loaded from ckpt if given, else trained), both paths."""
    torch.manual_seed(cfg.seed)
    model = EarlyExitClassifier(cfg).to(data["test"].x.device)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=data["test"].x.device, weights_only=True))
    else:
        train(model, data, cfg, "joint", log)
    fill_memory(model, data["memory"])
    out = collect(model, data["test"])
    return split_acc(out["full"], out["y"], old), split_acc(out["mem"], out["y"], old)
