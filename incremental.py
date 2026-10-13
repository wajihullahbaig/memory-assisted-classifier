import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from tqdm import tqdm

from mac import plots
from mac.config import Config, add_flags
from mac.data import get_data
from mac.incremental import finetune, insert, joint, train_base

NUM_CLASSES = 10
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# base model name -> config overrides. "baseline" is trained without the memory loss.
BASES = {"memory": {}, "baseline": {"mem_weight": 0.0}}


def report(rows):
    """Final results as a text table."""
    head = f"{'method':<44}{'steps':>7}{'time':>9}{'old':>8}{'new':>8}{'all':>8}"
    lines = [head, "-" * len(head)]
    for name, r in rows.items():
        steps = "-" if r.get("steps") is None else str(r["steps"])
        secs = "-" if r.get("seconds") is None else f"{r['seconds']:.2f}s"
        lines.append(f"{name:<44}{steps:>7}{secs:>9}{r['old']:>8.2f}{r['new']:>8.2f}{r['all']:>8.2f}")
    return "\n".join(lines)


def main():
    """Train on the old classes, add the new ones by memory insert or fine-tuning, compare with joint training."""
    parser = argparse.ArgumentParser(description="Adding new classes to a memory-assisted classifier without retraining")
    add_flags(parser)
    parser.set_defaults(out_dir="results/incremental")
    parser.add_argument("--old-classes", type=int, default=5, help="classes 0..n-1 are old, the rest are new")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10, 50, 100, 1000], help="images added per new class")
    parser.add_argument("--finetune-steps", type=int, default=500)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--joint-ckpt", default="results/memory.pt", help="model trained on all classes by main.py ('' = train it here)")
    args = vars(parser.parse_args())
    extra = {k: args.pop(k) for k in ("old_classes", "shots", "finetune_steps", "finetune_lr", "joint_ckpt")}
    cfg = Config(**args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "incremental.log", "a" if cfg.resume else "w")

    def log(msg=""):
        """Print a line and append it to incremental.log."""
        tqdm.write(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"device: {device} | {cfg} | {extra}")
    old, new = list(range(extra["old_classes"])), list(range(extra["old_classes"], NUM_CLASSES))
    per_class = cfg.memory_size // NUM_CLASSES
    data = get_data(cfg, device)

    sweeps, rows = {}, {}
    for name, overrides in BASES.items():
        base_cfg = replace(cfg, **overrides)
        model = train_base(f"base_{name}", base_cfg, data, old, log)
        torch.save(model.state_dict(), out / f"base_{name}.pt")
        label = "memory insert" + (" (no memory loss)" if name == "baseline" else "")
        sweeps[label] = [insert(model, data, old, new, s, per_class) for s in extra["shots"]]
        rows[f"{label}, {extra['shots'][-1]}/class"] = sweeps[label][-1]
        for r in sweeps[label]:
            seen_as = ", ".join(f"{CLASSES[c]}->{CLASSES[t]} {p:.0f}%" for c, (t, p) in r["lookalikes"].items())
            log(f"[{label}] {r['shots']:>5} images/class | old {r['old']:.2f} | new {r['new']:.2f} | all {r['all']:.2f} | {seen_as}")
        if name == "memory":
            tuned = finetune(model, data, base_cfg, new, old, extra["finetune_steps"], extra["finetune_lr"], log)
    rows["fine-tune on new classes"] = tuned

    ckpt = extra["joint_ckpt"] if extra["joint_ckpt"] and Path(extra["joint_ckpt"]).exists() else ""
    full, mem = joint(cfg, data, old, ckpt, log)
    rows["joint training, full network"] = full
    rows["joint training, memory only"] = mem

    refs = {k: rows[k] for k in ("fine-tune on new classes", "joint training, memory only")}
    plots.shots(sweeps, refs, len(old), out / "shots.png")
    plots.forgetting(rows, len(old), out / "forgetting.png")
    (out / "results.json").write_text(json.dumps({"config": asdict(cfg), **extra, "rows": rows, "sweeps": sweeps}, indent=2))

    log(f"\nFinal results (CIFAR-10 test set, old = classes {old}, new = classes {new})\n")
    log(report(rows))
    log(f"\nPlots, results.json and incremental.log written to {out}/")


if __name__ == "__main__":
    main()
