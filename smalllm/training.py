"""Token-weighted training, validation-only selection, epoch-boundary resume."""
from __future__ import annotations

import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import environment, fresh_dir, read_json, save_torch, seed_all, sync, write_json
from .data import BlockDataset, PAD, collate_blocks, load_data, read_rows
from .embeddings import load_table
from .models import NeuralLM, NGram


def loader(rows, config, shuffle=False, seed=0):
    return DataLoader(BlockDataset(rows, config["model"]["context_length"]),
                      batch_size=config["train"]["batch_size"], shuffle=shuffle,
                      collate_fn=collate_blocks, num_workers=config["train"]["num_workers"],
                      generator=torch.Generator().manual_seed(seed))


def loss_sum(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=PAD, reduction="sum")


def metrics(nll, tokens):
    if tokens <= 0 or not math.isfinite(nll):
        raise ValueError("No valid evaluation tokens or nonfinite negative log likelihood")
    ce = nll / tokens
    return {"nll_sum": nll, "tokens": tokens, "cross_entropy": ce,
            "perplexity": math.exp(ce) if ce < 700 else None}


@torch.inference_mode()
def neural_score(model, batches, device, max_batches=None):
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        total += float(loss_sum(model(x), y))
        count += int(y.ne(PAD).sum())
    return metrics(total, count)


def ngram_score(model, blocks):
    total, count = 0.0, 0
    for x, y in blocks:
        x, y = x.tolist(), y.tolist()
        for i, target in enumerate(y):
            if target != PAD:
                total -= math.log(model.prob(x[max(0, i + 2 - model.order):i + 1], target))
                count += 1
    return metrics(total, count)


def train(data, run, kind, config, device, embedding_path=None, resume=False):
    started = time.perf_counter()
    manifest, vocab = load_data(data)
    if config["data"] != manifest["data_config"]:
        raise ValueError("Data configuration differs from prepared data; use its original configuration")
    if kind == "ngram" and embedding_path:
        raise ValueError("n-gram does not use input embeddings")
    if embedding_path and kind != "lstm":
        raise ValueError("This experiment fixes the Part II architecture to LSTM")
    table, embedding_meta = (None, None) if embedding_path is None else load_table(embedding_path, manifest, config["model"]["embedding_dim"])
    variant = "random" if table is None else ("own" if embedding_meta["method"].startswith("own_") else "glove")
    signature = {"kind": kind, "variant": variant, "config": config,
                 "data_fingerprint": manifest["fingerprint"], "embedding": embedding_meta}
    run = Path(run)
    if resume:
        old = read_json(run / "run.json")
        if old["signature"] != signature:
            raise ValueError("Resume rejected: data, configuration or embedding identity changed")
        if kind == "ngram":
            raise ValueError("n-gram fit is atomic; choose a new run directory instead")
    else:
        fresh_dir(run)
    seed_all(config["seed"])
    effective_device = torch.device("cpu") if kind == "ngram" else device
    meta = {"signature": signature, "kind": kind, "variant": variant,
            "synthetic": manifest["synthetic"], "data_fingerprint": manifest["fingerprint"],
            "environment": environment(effective_device), "status": "running",
            "budget_limited": any(config["train"][k] is not None for k in ["max_train_batches", "max_val_batches"])}
    if kind == "ngram" and meta["budget_limited"]:
        raise ValueError("Batch-limited preflight is neural-only")
    write_json(run / "run.json", meta)
    train_rows, val_rows = [read_rows(Path(data) / f"lm_{s}.jsonl") for s in ["train", "val"]]
    base = {"kind": kind, "config": config, "variant": variant,
            "data_fingerprint": manifest["fingerprint"], "vocab_size": len(vocab["tokens"])}
    if kind == "ngram":
        model = NGram(len(vocab["tokens"]), **config["ngram"])
        t = time.perf_counter()
        model.fit(BlockDataset(train_rows, config["model"]["context_length"]))
        train_seconds = time.perf_counter() - t
        t = time.perf_counter()
        score = ngram_score(model, BlockDataset(val_rows, config["model"]["context_length"]))
        val_seconds = time.perf_counter() - t
        save_torch(run / "best.pt", {**base, "state": model.state(), "epoch": 1})
        meta.update(status="complete", best_epoch=1, best_val_ce=score["cross_entropy"],
                    training_seconds=train_seconds, validation_seconds=val_seconds,
                    wall_seconds=time.perf_counter() - started,
                    stored_count_entries=sum(len(v) for level in model.counts for v in level.values()),
                    checkpoint_bytes=(run / "best.pt").stat().st_size)
        write_json(run / "run.json", meta)
        return meta

    model = NeuralLM(kind, len(vocab["tokens"]), config["model"], table, frozen=table is not None).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    history, best_ce, best_epoch, best_state, start_epoch = [], float("inf"), 0, None, 0
    previous_wall = 0.0
    if resume:
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(last["state"])
        opt.load_state_dict(last["optimizer"])
        history, best_ce, best_epoch = last["history"], last["best_ce"], last["best_epoch"]
        best_state, start_epoch = last["best_state"], last["epoch"]
        previous_wall = last["wall_seconds"]
        save_torch(run / "best.pt", {**base, "state": best_state, "epoch": best_epoch})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    val_batches = loader(val_rows, config)
    for epoch in range(start_epoch, config["train"]["epochs"]):
        seed_all(config["seed"] + epoch)  # Makes epoch-boundary resume deterministic.
        batches = loader(train_rows, config, shuffle=True, seed=config["seed"] + epoch)
        model.train()
        sync(device)
        t = time.perf_counter()
        total, count, steps, clipped, grad_max = 0.0, 0, 0, 0, 0.0
        for i, (x, y) in enumerate(batches):
            limit = config["train"]["max_train_batches"]
            if limit is not None and i >= limit:
                break
            x, y = x.to(device), y.to(device)
            tokens = int(y.ne(PAD).sum())
            nll = loss_sum(model(x), y)
            loss = nll / tokens
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, batch {i}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"], error_if_nonfinite=True))
            opt.step()
            grad_max = max(grad_max, norm)
            clipped += int(norm > config["train"]["grad_clip"])
            total += float(nll.detach())
            count += tokens
            steps += 1
        sync(device)
        train_seconds = time.perf_counter() - t
        train_metrics = metrics(total, count)
        t = time.perf_counter()
        val = neural_score(model, val_batches, device, config["train"]["max_val_batches"])
        sync(device)
        val_seconds = time.perf_counter() - t
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val,
                  "training_seconds": train_seconds, "validation_seconds": val_seconds,
                  "optimizer_steps": steps, "gradient_norm_max_before_clip": grad_max,
                  "clipped_step_fraction": clipped / steps}
        history.append(record)
        if val["cross_entropy"] < best_ce:
            best_ce, best_epoch = val["cross_entropy"], epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # last.pt is the atomic source of truth after interruption; best/history are regenerated on resume.
        save_torch(run / "last.pt", {**base, "state": model.state_dict(), "optimizer": opt.state_dict(),
                                     "epoch": epoch + 1, "history": history, "best_ce": best_ce,
                                     "best_epoch": best_epoch, "best_state": best_state,
                                     "wall_seconds": previous_wall + time.perf_counter() - started})
        save_torch(run / "best.pt", {**base, "state": best_state, "epoch": best_epoch})
        write_json(run / "history.json", history)
        print(f"{run.name} epoch {epoch + 1}: train CE={train_metrics['cross_entropy']:.4f}, val CE={val['cross_entropy']:.4f}, train={train_seconds:.1f}s", flush=True)
    write_json(run / "history.json", history)
    meta.update(status="complete", best_epoch=best_epoch, best_val_ce=best_ce,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                training_seconds=sum(h["training_seconds"] for h in history),
                validation_seconds=sum(h["validation_seconds"] for h in history),
                wall_seconds=previous_wall + time.perf_counter() - started,
                checkpoint_bytes=(run / "best.pt").stat().st_size,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                peak_memory_scope="current process invocation; not cumulative across resume")
    write_json(run / "run.json", meta)
    return meta


def load_model(run, manifest, device):
    run = Path(run)
    meta = read_json(run / "run.json")
    if meta["status"] != "complete":
        raise ValueError("Training run is not complete; finish/resume before evaluating")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["data_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint belongs to a different prepared dataset")
    if checkpoint["kind"] == "ngram":
        model = NGram.from_state(checkpoint["state"])
    else:
        model = NeuralLM(checkpoint["kind"], checkpoint["vocab_size"], checkpoint["config"]["model"])
        model.load_state_dict(checkpoint["state"])
        model.to(device).eval()
    return model, checkpoint, meta
