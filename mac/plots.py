import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order, one per run
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "lines.linewidth": 2, "legend.frameon": False, "axes.axisbelow": True, "ytick.left": False,
})


def _save(fig, path):
    """Lay out, save and close a figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def training(histories, path):
    """Loss, validation accuracy and share of early exits over training steps."""
    panels = [("loss", "Training loss (memory run includes contrastive loss)"), ("val_acc", "Validation accuracy (%)"), ("val_exits", "Early exits on validation (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (key, title) in zip(axes, panels):
        for color, (name, hist) in zip(COLORS, histories.items()):
            steps = [h["step"] for h in hist]
            ax.plot(steps, [h[key] for h in hist], color=color, label=name, marker="o", markersize=3.5)
        ax.set_title(title)
        ax.set_xlabel("step")
    axes[0].legend()
    _save(fig, path)


def tradeoff(sweeps, results, path):
    """Test accuracy vs share of images that exit early, one curve per run."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for color, (name, sweep) in zip(COLORS, sweeps.items()):
        r = results[name]
        ax.plot([s["exits"] for s in sweep], [s["acc"] for s in sweep], color=color, label=name)
        ax.axhline(r["full_acc"], color=color, linewidth=1, linestyle=":", alpha=0.8)
        ax.plot(r["exits"], r["exit_acc"], "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        right = r["exits"] > 60
        ax.annotate(f"{r['exits']:.0f}% exit early, {r['exit_acc']:.1f}% acc\n"
                    f"{r['gpu_speedup']:.2f}x GPU, {r['cpu_speedup']:.2f}x CPU",
                    (r["exits"], r["exit_acc"]), xytext=(-10 if right else 10, -30), textcoords="offset points",
                    ha="right" if right else "left", color=MUTED, fontsize=9)
    accs = [s["acc"] for sweep in sweeps.values() for s in sweep]
    ax.set_ylim(max(min(r["full_acc"] for r in results.values()) - 12, min(accs) - 1), max(accs) + 1.5)
    ax.set_xlim(-2, 102)
    ax.set_title("Accuracy vs early exits (test set)")
    ax.set_xlabel("images answered by memory (%)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower left")
    _save(fig, path)


def summary(results, path):
    """Final numbers: accuracy of each mode and measured speedup per device."""
    names = list(results)
    panels = [
        ("Test accuracy (%)", [("full network", "full_acc"), ("early exit", "exit_acc")], "{:.1f}"),
        ("Speedup over full network", [("GPU", "gpu_speedup"), ("CPU", "cpu_speedup")], "{:.2f}x"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.1 + 0.55 * len(names) * 2))
    for ax, (title, rows, fmt) in zip(axes, panels):
        labels, values, colors = [], [], []
        for color, name in zip(COLORS, names):
            for label, key in rows:
                labels.append(f"{name} - {label}")
                values.append(results[name][key])
                colors.append(color)
        y = range(len(values))[::-1]
        ax.barh(list(y), values, color=colors, height=0.5)
        for yi, v in zip(y, values):
            ax.text(v, yi, " " + fmt.format(v), va="center", color=INK, fontsize=9)
        ax.set_yticks(list(y), labels)
        ax.set_xlim(0, max(values) * 1.15)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    axes[1].axvline(1, color=MUTED, linewidth=1, linestyle=":")
    _save(fig, path)


def _splits(n_old, n=10):
    """(key, title) for the old, new and all-class accuracy panels."""
    return [("old", f"Old classes (0-{n_old - 1})"), ("new", f"New classes ({n_old}-{n - 1})"), ("all", f"All {n} classes")]


def shots(sweeps, refs, n_old, path):
    """Test accuracy vs images added per new class, with fixed reference methods as dotted lines."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (key, title) in zip(axes, _splits(n_old)):
        for color, (name, sweep) in zip(COLORS, sweeps.items()):
            ax.plot([s["shots"] for s in sweep], [s[key] for s in sweep], color=color, label=name, marker="o", markersize=5)
        for color, (name, r) in zip(COLORS[len(sweeps):], refs.items()):
            ax.axhline(r[key], color=color, linewidth=1.5, linestyle=":", label=name)
        ax.set_xscale("log")
        ticks = [s["shots"] for s in next(iter(sweeps.values()))]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(-3, 103)
        ax.set_title(title)
        ax.set_xlabel("images added per new class")
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    _save(fig, path)


def forgetting(rows, n_old, path):
    """Old, new and all-class test accuracy for each method as grouped horizontal bars."""
    names = list(rows)
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.75 * len(names)))
    height = 0.24
    for i, (color, (key, label)) in enumerate(zip(COLORS, _splits(n_old))):
        y = [len(names) - 1 - n + (1 - i) * (height + 0.03) for n in range(len(names))]
        values = [rows[n][key] for n in names]
        ax.barh(y, values, height=height, color=color, label=label)
        for yi, v in zip(y, values):
            ax.text(v, yi, f" {v:.1f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(names))[::-1], names)
    ax.set_xlim(0, 110)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("test accuracy (%)")
    ax.set_title(f"Adding classes {n_old}-9 to a model trained on classes 0-{n_old - 1}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(names)), ncol=3)
    _save(fig, path)
