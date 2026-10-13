import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .evaluate import accuracy, autocast, calibrate, collect, fill_memory, score

STATE = ("step", "history", "best", "best_state", "bad", "done")  # training state kept in checkpoints


def forever(loader):
    while True:
        yield from loader


def validate(model, data, cfg):
    fill_memory(model, data["memory"])
    out = collect(model, data["val"])
    best = score(out, calibrate(out, cfg.max_drop))
    return {"val_acc": accuracy(out["full"], out["y"]), "val_memory_acc": accuracy(out["mem"], out["y"]), "val_exits": best["exits"]}


def train(model, data, cfg, name, log=print):
    """Step-based training with periodic validation, checkpointing and early stopping on val accuracy."""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.max_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    parts = {"model": model, "opt": opt, "sched": sched, "scaler": scaler}
    ckpt_path = Path(cfg.out_dir) / f"{name}_last.pt"

    step, history, best, best_state, bad, done = 0, [], -1.0, None, 0, False
    if cfg.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        for key, part in parts.items():
            part.load_state_dict(ckpt[key])
        step, history, best, best_state, bad, done = (ckpt[k] for k in STATE)
        log(f"[{name}] resumed from step {step}" + (" (already finished)" if done else ""))

    batches, start, running, seen = forever(data["train"]), time.time(), 0.0, 0
    bar = tqdm(range(step + 1, (step if done else cfg.max_steps) + 1), initial=step, total=cfg.max_steps, desc=name, ncols=100)
    for step in bar:
        model.train()
        x, y = next(batches)
        with autocast(device):
            keys, logits = model(x)
            loss = F.cross_entropy(logits, y)
            if cfg.mem_weight:
                loss = loss + cfg.mem_weight * model.memory.loss(keys, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        model.memory.add(keys.detach(), y)
        running, seen = running + loss.detach(), seen + 1

        if step % cfg.eval_every and step != cfg.max_steps:
            continue
        row = {"step": step, "loss": (running / seen).item(), **validate(model, data, cfg)}
        running, seen = 0.0, 0
        history.append(row)
        improved = row["val_acc"] > best
        if improved:
            best, bad = row["val_acc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        done = bad >= cfg.patience or step == cfg.max_steps
        torch.save({**{k: p.state_dict() for k, p in parts.items()}, **dict(zip(STATE, (step, history, best, best_state, bad, done)))}, ckpt_path)
        log(
            f"[{name}] step {step:>5} | loss {row['loss']:.3f} | val acc {row['val_acc']:.2f} "
            f"| memory acc {row['val_memory_acc']:.2f} | early exits {row['val_exits']:.1f}% "
            f"| {time.time() - start:.0f}s{'  *' if improved else ''}"
        )
        if bad >= cfg.patience:
            log(f"[{name}] early stop: no improvement for {cfg.patience} evals")
            break
    bar.close()

    model.load_state_dict(best_state)
    return history
