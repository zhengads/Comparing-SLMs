"""Artifact identity, deterministic setup and explicit device selection."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {path}. Choose a new output path.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run 'python -m core doctor'; CPU must be selected explicitly.")
    if name not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    return torch.device(name)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def environment(device="cpu"):
    packages = {}
    for name in ["torch", "numpy", "scipy", "scikit-learn", "matplotlib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        commit = rev.stdout.strip() if rev.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            gpu.append({"index": i, "name": prop.name, "total_bytes": prop.total_memory})
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": platform.processor(), "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "packages": packages,
            "torch_cuda_build": torch.version.cuda, "gpus": gpu, "device": str(device),
            "git_commit": commit,
            "source_hash": identity({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})}


def load_config(path):
    c = read_json(path)
    m, t = c["model"], c["train"]
    if m["hidden_dim"] % m["heads"] or min(m["embedding_dim"], m["context_length"], m["layers"]) < 1:
        raise ValueError("Invalid model dimensions or attention heads")
    if min(t["epochs"], t["batch_size"], t["learning_rate"], t["grad_clip"]) <= 0:
        raise ValueError("Training epochs, batch size, learning rate and clipping must be positive")
    if c["probe"]["max_tokens"] > m["context_length"]:
        raise ValueError("Probe max_tokens cannot exceed model context_length")
    if min(c["probe"]["max_tokens"], c["probe"]["batch_size"], c["probe"]["max_iter"]) < 1:
        raise ValueError("Probe lengths, batch size and max_iter must be positive")
    if not c["probe"]["c_values"] or min(c["probe"]["c_values"]) <= 0:
        raise ValueError("Classifier C grid must be nonempty and positive")
    if c["probe"]["pooling"] not in {"mean", "last"}:
        raise ValueError("Probe pooling must be mean or last")
    if not 0 <= m["dropout"] < 1 or t["num_workers"] < 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid dropout, worker count or weight decay")
    for key in ["max_train_batches", "max_val_batches"]:
        if t[key] is not None and t[key] < 1:
            raise ValueError(f"{key} must be null or positive")
    return c


def save_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)
