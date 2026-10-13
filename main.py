import argparse
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from tqdm import tqdm

from mac import plots
from mac.config import Config, add_flags
from mac.data import get_data
from mac.evaluate import benchmark, calibrate, collect, fill_memory, summarize, sweep
from mac.model import EarlyExitClassifier
from mac.train import train

# name -> config overrides. "baseline" is the same network trained without the memory loss.
RUNS = {"memory": {}, "baseline": {"mem_weight": 0.0}}


def run(name, cfg, device, out, log):
    """Train, calibrate, evaluate and benchmark one run."""
    torch.manual_seed(cfg.seed)
    data = get_data(cfg, device)
    model = EarlyExitClassifier(cfg).to(device)
    history = train(model, data, cfg, name, log)

    fill_memory(model, data["memory"])
    threshold = calibrate(collect(model, data["val"]), cfg.max_drop)
    test = collect(model, data["test"])
    result = summarize(test, threshold)

    images = torch.cat([x for x, _ in data["test"]])
    gpu = benchmark(model, images, threshold, batch_size=cfg.batch_size)
    cpu = benchmark(copy.deepcopy(model).cpu(), images[:500].cpu(), threshold, batch_size=1, reps=1)
    result |= {f"gpu_{k}": v for k, v in gpu.items()} | {f"cpu_{k}": v for k, v in cpu.items()}

    torch.save(model.state_dict(), out / f"{name}.pt")
    return history, sweep(test), result


def report(results):
    """Final results as a text table."""
    head = f"{'run':<10}{'full acc':>10}{'exit acc':>10}{'exits':>8}{'thresh':>8}{'GPU':>8}{'CPU':>8}"
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        lines.append(f"{name:<10}{r['full_acc']:>10.2f}{r['exit_acc']:>10.2f}{r['exits']:>7.1f}%"
                     f"{r['threshold']:>8.2f}{r['gpu_speedup']:>7.2f}x{r['cpu_speedup']:>7.2f}x")
    return "\n".join(lines)


def main():
    """Parse flags, run every experiment, write plots, results and logs."""
    parser = argparse.ArgumentParser(description="Memory-assisted early-exit classifier on CIFAR-10")
    add_flags(parser)
    parser.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    args = vars(parser.parse_args())
    runs = args.pop("runs")
    cfg = Config(**args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(exist_ok=True)
    log_file = open(out / "train.log", "a" if cfg.resume else "w")

    def log(msg=""):
        """Print a line and append it to train.log."""
        tqdm.write(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"device: {device} | {cfg}")

    histories, sweeps, results = {}, {}, {}
    for name in runs:
        histories[name], sweeps[name], results[name] = run(name, replace(cfg, **RUNS[name]), device, out, log)

    plots.training(histories, out / "training.png")
    plots.tradeoff(sweeps, results, out / "tradeoff.png")
    plots.summary(results, out / "summary.png")
    (out / "results.json").write_text(json.dumps({"config": asdict(cfg), "results": results, "history": histories}, indent=2))

    log("\nFinal results (test set, CIFAR-10)\n")
    log(report(results))
    log(f"\nPlots, results.json and train.log written to {out}/")


if __name__ == "__main__":
    main()
