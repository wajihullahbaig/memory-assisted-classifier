import time

import torch

THRESHOLDS = [i / 100 for i in range(102)]  # 1.01 = never exit early


def autocast(device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def fill_memory(model, loader):
    """Rebuild the memory from clean (un-augmented) training images."""
    model.eval()
    model.memory.reset()
    for x, y in loader:
        with autocast(x.device):
            model.memory.add(model.keys(model.early(x)), y)


@torch.no_grad()
def collect(model, loader):
    """Run both paths on every image so any threshold can be scored offline."""
    model.eval()
    out = {"y": [], "full": [], "mem": [], "conf": []}
    for x, y in loader:
        with autocast(x.device):
            keys, logits = model(x)
        conf, mem = model.memory.predict(keys).max(1)
        for k, v in zip(out, (y, logits.argmax(1), mem, conf)):
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def accuracy(pred, y):
    return (pred == y).float().mean().item() * 100


def score(out, threshold):
    hit = out["conf"] >= threshold
    pred = torch.where(hit, out["mem"], out["full"])
    return {"threshold": threshold, "acc": accuracy(pred, out["y"]), "exits": hit.float().mean().item() * 100}


def sweep(out):
    return [score(out, t) for t in THRESHOLDS]


def calibrate(out, max_drop):
    """Lowest threshold (most early exits) that stays within max_drop of the full network."""
    full = accuracy(out["full"], out["y"])
    return min(r["threshold"] for r in sweep(out) if r["acc"] >= full - max_drop)


def summarize(out, threshold):
    return {
        "full_acc": accuracy(out["full"], out["y"]),
        "memory_acc": accuracy(out["mem"], out["y"]),
        **{("exit_acc" if k == "acc" else k): v for k, v in score(out, threshold).items()},
    }


@torch.no_grad()
def throughput(model, x, threshold, batch_size, reps=3):
    """Images per second through model.predict on the device x lives on."""
    model.eval()

    def run():
        with autocast(x.device):
            for xb in x.split(batch_size):
                model.predict(xb, threshold)

    run()  # warm-up
    sync(x.device)
    start = time.perf_counter()
    for _ in range(reps):
        run()
    sync(x.device)
    return reps * len(x) / (time.perf_counter() - start)


def benchmark(model, x, threshold, batch_size, reps=3):
    full = throughput(model, x, None, batch_size, reps)
    exit_ = throughput(model, x, threshold, batch_size, reps)
    return {"full_ips": full, "exit_ips": exit_, "speedup": exit_ / full}
