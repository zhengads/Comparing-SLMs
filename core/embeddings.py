"""Own Skip-gram negative sampling and streamed, provenance-tracked GloVe import."""
from __future__ import annotations

import io
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import environment, fresh_dir, read_json, seed_all, sha256, sync, write_json
from .data import load_data, read_rows


def random_table(size, dim, seed):
    table = np.random.default_rng(seed).normal(0, 0.02, (size, dim)).astype("float32")
    table[0] = 0
    return table


def save_table(output, table, manifest, vocab, metadata, covered):
    out = fresh_dir(output)
    np.save(out / "table.npy", table, allow_pickle=False)
    counts = np.asarray(vocab["counts"], dtype=np.int64)
    total = int(counts[4:].sum())
    metadata.update({"data_fingerprint": manifest["fingerprint"], "vocab_hash": manifest["vocab_hash"],
                     "synthetic": manifest["synthetic"], "dimension": table.shape[1],
                     "table_sha256": sha256(out / "table.npy"),
                     "vocab_coverage": sum(i in covered for i in range(4, len(counts))) / max(1, len(counts) - 4),
                     "training_token_coverage": sum(int(counts[i]) for i in covered) / max(total, 1),
                     "missing_initialization": "Normal(0, 0.02), fixed seed; PAD zero; entire table frozen in LM",
                     "covered_content_rows": len(covered), "vocab_content_rows": len(counts) - 4})
    write_json(out / "metadata.json", metadata)
    return metadata


def load_table(path, manifest, dimension):
    path = Path(path)
    meta = read_json(path / "metadata.json")
    if (meta["data_fingerprint"] != manifest["fingerprint"] or meta["vocab_hash"] != manifest["vocab_hash"]
            or meta["dimension"] != dimension or meta["table_sha256"] != sha256(path / "table.npy")):
        raise ValueError("Embedding provenance, dimension or file hash does not match prepared data")
    table = np.load(path / "table.npy", allow_pickle=False)
    if not np.isfinite(table).all():
        raise ValueError("Embedding contains nonfinite values")
    return table, meta


def import_glove(data, source, output, config):
    manifest, vocab = load_data(data)
    dim = config["model"]["embedding_dim"]
    table = random_table(len(vocab["tokens"]), dim, config["seed"])
    lookup = {w: i for i, w in enumerate(vocab["tokens"]) if i >= 4}
    covered = set()
    start = time.perf_counter()
    source = Path(source)
    member = None
    archive = None
    if source.suffix == ".zip":
        archive = zipfile.ZipFile(source)
        candidates = [n for n in archive.namelist() if n.endswith(f".{dim}d.txt")]
        if len(candidates) != 1:
            archive.close()
            raise ValueError(f"Expected exactly one .{dim}d.txt member in GloVe ZIP")
        member = candidates[0]
        stream = io.TextIOWrapper(archive.open(member), encoding="utf-8")
    else:
        stream = source.open(encoding="utf-8")
    try:
        with stream:
            for line in stream:
                parts = line.rstrip().split()
                if not parts or parts[0] not in lookup:
                    continue
                if len(parts) != dim + 1:
                    raise ValueError(f"GloVe dimension mismatch for token {parts[0]}")
                values = np.asarray(parts[1:], dtype="float32")
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite GloVe vector")
                idx = lookup[parts[0]]
                table[idx] = values
                covered.add(idx)
    finally:
        if archive:
            archive.close()
    if not covered:
        raise ValueError("No vocabulary overlap with supplied vectors")
    return save_table(output, table, manifest, vocab,
                      {"method": "public_glove", "source": str(source), "member": member,
                       "source_sha256": sha256(source), "import_seconds": time.perf_counter() - start,
                       "seed": config["seed"], "external_training_data": True}, covered)


def pair_batches(rows, window, batch_size, rng, keep):
    centers, contexts = [], []
    for index in rng.sample(range(len(rows)), len(rows)):
        ids = [i for i in rows[index]["ids"] if i >= 4 and rng.random() < keep[i]]
        for pos, token in enumerate(ids):
            for j in range(max(0, pos - window), min(len(ids), pos + window + 1)):
                if j != pos:
                    centers.append(token)
                    contexts.append(ids[j])
                    if len(centers) == batch_size:
                        yield centers, contexts
                        centers, contexts = [], []
    if centers:
        yield centers, contexts


def train_own(data, output, config, device):
    if Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError(output)
    manifest, vocab = load_data(data)
    rows = read_rows(Path(data) / "lm_train.jsonl")
    seed_all(config["seed"])
    c = config["embeddings"]
    dim, size = config["model"]["embedding_dim"], len(vocab["tokens"])
    if min(c["epochs"], c["window"], c["negatives"], c["batch_size"], c["learning_rate"]) <= 0:
        raise ValueError("Skip-gram settings must be positive")
    counts = np.asarray(vocab["counts"], dtype="float64")
    freq = counts / max(1, counts.sum())
    keep = np.ones(size)
    if c["subsample"] > 0:
        positive = freq > 0
        ratio = c["subsample"] / freq[positive]
        keep[positive] = np.minimum(1, np.sqrt(ratio) + ratio)
    noise = torch.as_tensor(counts ** 0.75, dtype=torch.float32, device=device)
    if noise.sum() <= 0:
        raise ValueError("No content tokens available for Skip-gram")
    noise /= noise.sum()
    table = random_table(size, dim, config["seed"])
    inp = nn.Embedding.from_pretrained(torch.tensor(table), freeze=False).to(device)
    out = nn.Embedding(size, dim).to(device)
    nn.init.zeros_(out.weight)
    opt = torch.optim.Adam(list(inp.parameters()) + list(out.parameters()), lr=c["learning_rate"])
    history, covered = [], set()
    sync(device)
    started = time.perf_counter()
    for epoch in range(c["epochs"]):
        total_loss, pairs = 0.0, 0
        for a, b in pair_batches(rows, c["window"], c["batch_size"], random.Random(config["seed"] + epoch), keep):
            covered.update(a)
            centers = torch.tensor(a, device=device)
            targets = torch.tensor(b, device=device)
            negatives = torch.multinomial(noise, len(a) * c["negatives"], replacement=True).view(len(a), -1)
            iv = inp(centers)
            pos = (iv * out(targets)).sum(-1)
            neg = torch.bmm(out(negatives), iv.unsqueeze(-1)).squeeze(-1)
            loss = -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Skip-gram loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(a)
            pairs += len(a)
        if pairs == 0:
            raise ValueError("No Skip-gram pairs; reduce subsampling or use more training data")
        history.append({"epoch": epoch + 1, "loss_per_pair": total_loss / pairs, "pairs": pairs})
        print(f"Skip-gram epoch {epoch + 1}: {pairs} pairs, loss={total_loss / pairs:.4f}", flush=True)
    sync(device)
    seconds = time.perf_counter() - started
    return save_table(output, inp.weight.detach().cpu().numpy(), manifest, vocab,
                      {"method": "own_skipgram_negative_sampling", "config": c, "seed": config["seed"],
                       "training_seconds": seconds, "history": history, "environment": environment(device),
                       "external_training_data": False}, covered)
