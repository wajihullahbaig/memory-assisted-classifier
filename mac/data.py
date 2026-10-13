import torch
import torch.nn.functional as F
import torchvision

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def augment(x, pad=4):
    """Per-image random crop (zero padding) and horizontal flip, fully on the device."""
    b, c, h, w = x.shape
    x = F.pad(x, (pad,) * 4)
    i = torch.randint(0, 2 * pad + 1, (b, 1, 1, 1), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (b, 1, 1, 1), device=x.device)
    rows = i + torch.arange(h, device=x.device).view(1, 1, h, 1)
    cols = j + torch.arange(w, device=x.device).view(1, 1, 1, w)
    x = x[torch.arange(b, device=x.device).view(b, 1, 1, 1), torch.arange(c, device=x.device).view(1, c, 1, 1), rows, cols]
    flip = torch.rand(b, 1, 1, 1, device=x.device) < 0.5
    return torch.where(flip, x.flip(3), x)


class Loader:
    """Batches a uint8 image tensor that lives on the device. No workers, no copies."""

    def __init__(self, x, y, batch_size, train=False, img_size=32):
        """Wrap images [N, 3, H, W] (uint8) and labels [N] on one device."""
        self.x, self.y, self.batch_size, self.train, self.img_size = x, y, batch_size, train, img_size

    def __len__(self):
        """Number of batches."""
        return -(-len(self.x) // self.batch_size)

    def __iter__(self):
        """Yield (images, labels) batches, shuffled when training."""
        n, dev = len(self.x), self.x.device
        order = torch.randperm(n, device=dev) if self.train else torch.arange(n, device=dev)
        for idx in order.split(self.batch_size):
            yield self.prep(self.x[idx]), self.y[idx]

    def prep(self, x):
        """Normalise, augment (train only) and resize a uint8 batch."""
        x = (x.float() / 255 - MEAN.to(x.device)) / STD.to(x.device)
        if self.train:
            x = augment(x)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return x


def get_data(cfg, device, val_size=5000):
    """CIFAR-10 on the device: train / val split, a clean memory set, and test."""

    def load(train):
        """One CIFAR-10 split as device tensors."""
        ds = torchvision.datasets.CIFAR10("data", train=train, download=True)
        return torch.tensor(ds.data).permute(0, 3, 1, 2).to(device), torch.tensor(ds.targets).to(device)

    x, y = load(True)
    xt, yt = load(False)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(cfg.seed)).to(device)
    val, train, mem = perm[:val_size], perm[val_size:], perm[val_size:val_size + cfg.memory_size]

    def loader(idx=None, train=False, xs=x, ys=y):
        """Loader over a subset (idx) of a split."""
        xs, ys = (xs, ys) if idx is None else (xs[idx], ys[idx])
        return Loader(xs, ys, cfg.batch_size, train, cfg.img_size)

    return {
        "train": loader(train, train=True),
        "memory": loader(mem),
        "val": loader(val),
        "test": loader(xs=xt, ys=yt),
    }
