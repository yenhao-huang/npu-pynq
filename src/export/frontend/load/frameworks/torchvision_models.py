"""torchvision: a classification model by name, with weights from a file or by name.

    source: a state-dict file (.pth) or a torchvision weights name (IMAGENET1K_V1)
    model:  the torchvision model name (resnet18)
"""

from __future__ import annotations

from pathlib import Path

import torch


def load(source: str | Path, model: str = 'resnet18', **kwargs) -> torch.nn.Module:
    import torchvision

    path = Path(source)
    if path.suffix in ('.pth', '.pt') and path.is_file():
        net = torchvision.models.get_model(model, weights=None, **kwargs)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    else:
        net = torchvision.models.get_model(model, weights=str(source), **kwargs)
    return net.eval().requires_grad_(False)


def files(source: str | Path) -> dict[str, Path]:
    return {}
