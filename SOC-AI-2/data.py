"""MNIST data loader for the SOC-AI-2 study.

We deliberately avoid a torchvision dependency at runtime: the four
idx-format files are fetched from a GitHub-hosted mirror on first use and
cached under ``SOC-AI-2/data/``.  Parsing is done with ``gzip`` and ``numpy``
only, so the script is self-contained and reproducible inside the sandbox.

The 60 000 training and 10 000 test images are returned as float tensors
normalised to ``[0, 1]`` and flattened to 784-dim vectors, plus integer
class labels in {0, ..., 9}.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Public GitHub mirror with the four canonical idx files
MIRROR = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data"
FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


@dataclass
class MNISTConfig:
    flatten: bool = True       # 784-dim vectors (True) or 28×28 (False)
    normalise: bool = True     # divide pixel values by 255
    seed: int = 20260516


def _ensure_files() -> None:
    """Download the four idx files into ``DATA_DIR`` if absent."""
    for fname in FILES:
        path = DATA_DIR / fname
        if path.exists() and path.stat().st_size > 0:
            continue
        url = f"{MIRROR}/{fname}"
        print(f"[mnist] fetching {url}")
        with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad image magic {magic}"
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad label magic {magic}"
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


class MNIST:
    """In-memory MNIST corpus with mini-batch sampling."""

    def __init__(self, cfg: MNISTConfig = MNISTConfig()):
        self.cfg = cfg
        _ensure_files()
        train_x = _read_idx_images(DATA_DIR / FILES[0]).astype(np.float32)
        train_y = _read_idx_labels(DATA_DIR / FILES[1]).astype(np.int64)
        test_x = _read_idx_images(DATA_DIR / FILES[2]).astype(np.float32)
        test_y = _read_idx_labels(DATA_DIR / FILES[3]).astype(np.int64)
        if cfg.normalise:
            train_x = train_x / 255.0
            test_x = test_x / 255.0
        if cfg.flatten:
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)
        self.train_x = torch.from_numpy(train_x)
        self.train_y = torch.from_numpy(train_y)
        self.test_x = torch.from_numpy(test_x)
        self.test_y = torch.from_numpy(test_y)
        self.input_dim = self.train_x.shape[-1] if cfg.flatten else 28 * 28
        self.num_classes = 10

    # ------------------------------------------------------------------
    def batch(self, split: str, batch_size: int,
              rng: np.random.Generator
              ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.train_x if split == "train" else self.test_x
        y = self.train_y if split == "train" else self.test_y
        idx = rng.integers(0, x.shape[0], size=batch_size)
        idx_t = torch.from_numpy(idx)
        return x[idx_t], y[idx_t]


if __name__ == "__main__":
    ds = MNIST()
    print(f"train: {ds.train_x.shape}, {ds.train_y.shape}")
    print(f"test:  {ds.test_x.shape}, {ds.test_y.shape}")
    print(f"pixel range: [{ds.train_x.min():.3f}, {ds.train_x.max():.3f}]")
    print(f"label range: [{ds.train_y.min().item()}, "
          f"{ds.train_y.max().item()}]")
    # uniform baseline cross-entropy for 10 classes
    print(f"uniform NLL = {np.log(10):.4f} nats/sample")
