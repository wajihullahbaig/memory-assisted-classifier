from dataclasses import dataclass


@dataclass
class Config:
    # model
    backbone: str = "resnet18"      # any timm ResNet
    exit_layer: str = "layer2"      # memory lookup happens after this stage
    img_size: int = 64              # CIFAR images are upscaled; pretrained ResNets like bigger inputs
    # memory
    memory_size: int = 10_000
    key_dim: int = 128
    k: int = 10
    temperature: float = 0.07
    mem_weight: float = 1.0         # weight of the memory contrastive loss (0 = plain network)
    # training
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.05
    max_steps: int = 6000
    eval_every: int = 300
    patience: int = 5               # evals without improvement before stopping
    resume: bool = False            # continue from results/<run>_last.pt if it exists
    # early exit
    max_drop: float = 0.5           # accuracy points we accept losing for speed
    seed: int = 0
    out_dir: str = "results"
