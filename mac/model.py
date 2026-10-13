import timm
import torch
import torch.nn as nn

from .memory import Memory


class EarlyExitClassifier(nn.Module):
    """A ResNet cut in two. The memory reads features after `exit_layer`;
    the late stages and the head only run for images the memory is unsure about."""

    def __init__(self, cfg, num_classes=10):
        """Build the split backbone, key projection, head and memory."""
        super().__init__()
        net = timm.create_model(cfg.backbone, pretrained=True, num_classes=0)
        stages = ["layer1", "layer2", "layer3", "layer4"]
        cut = stages.index(cfg.exit_layer) + 1
        stem = nn.Sequential(net.conv1, net.bn1, net.act1, net.maxpool)
        self.early = nn.Sequential(stem, *[getattr(net, s) for s in stages[:cut]])
        self.late = nn.Sequential(*[getattr(net, s) for s in stages[cut:]])
        channels = {f["module"]: f["num_chs"] for f in net.feature_info}
        self.proj = nn.Linear(channels[cfg.exit_layer], cfg.key_dim)
        self.head = nn.Linear(net.num_features, num_classes)
        self.memory = Memory(cfg.memory_size, cfg.key_dim, num_classes, cfg.k, cfg.temperature)

    def keys(self, h):
        """Memory key from early-stage features."""
        return self.proj(h.mean((2, 3)))

    def classify(self, h):
        """Full-network logits from early-stage features."""
        return self.head(self.late(h).mean((2, 3)))

    def forward(self, x):
        """Memory keys and full-network logits for a batch."""
        h = self.early(x)
        return self.keys(h), self.classify(h)

    @torch.no_grad()
    def predict(self, x, threshold=None):
        """Returns (predictions, exited_early). threshold=None runs the full network."""
        h = self.early(x)
        if threshold is None:
            return self.classify(h).argmax(1), torch.zeros(len(x), dtype=torch.bool, device=x.device)
        conf, pred = self.memory.predict(self.keys(h)).max(1)
        hit = conf >= threshold
        if not hit.all():
            pred[~hit] = self.classify(h[~hit]).argmax(1)
        return pred, hit
