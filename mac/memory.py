import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    """Ring buffer of (normalised key, label) pairs with a k-NN read-out.

    Everything is a buffer, so the memory is saved and loaded with the state_dict.
    """

    def __init__(self, size, dim, num_classes, k=10, temperature=0.07):
        super().__init__()
        self.num_classes, self.k, self.temperature = num_classes, k, temperature
        self.register_buffer("keys", torch.zeros(size, dim))
        self.register_buffer("labels", torch.zeros(size, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def reset(self):
        self.ptr.zero_()
        self.count.zero_()

    @torch.no_grad()
    def add(self, keys, labels):
        size = len(self.keys)
        idx = (self.ptr + torch.arange(len(keys), device=keys.device)) % size
        self.keys[idx] = F.normalize(keys.float(), dim=1)
        self.labels[idx] = labels
        self.ptr.copy_((self.ptr + len(keys)) % size)
        self.count.copy_((self.count + len(keys)).clamp(max=size))

    def similarity(self, keys):
        n = int(self.count)
        return F.normalize(keys, dim=1) @ self.keys[:n].to(keys.dtype).T, self.labels[:n]

    @torch.no_grad()
    def predict(self, keys):
        """Class probabilities from a similarity-weighted vote of the k nearest entries."""
        sim, labels = self.similarity(keys)
        top, idx = sim.topk(min(self.k, sim.shape[1]), dim=1)
        weights = (top.float() / self.temperature).softmax(dim=1)
        probs = torch.zeros(len(keys), self.num_classes, device=keys.device)
        return probs.scatter_add_(1, labels[idx], weights)

    def loss(self, keys, labels):
        """Supervised contrastive loss: pull each key towards same-class memory entries."""
        sim, mem_labels = self.similarity(keys)
        log_prob = (sim.float() / self.temperature).log_softmax(dim=1)
        pos = (labels[:, None] == mem_labels[None]).float()
        n_pos = pos.sum(1)
        per_sample = -(log_prob * pos).sum(1) / n_pos.clamp(min=1)
        has_pos = (n_pos > 0).float()
        return (per_sample * has_pos).sum() / has_pos.sum().clamp(min=1)
