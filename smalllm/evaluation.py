"""Explicit held-out scoring and a shared generation workload."""
from pathlib import Path
import time

import numpy as np
import torch

from .common import environment, fresh_dir, seed_all, sha256, sync, write_json
from .data import BlockDataset, BOS, EOS, PAD, UNK, load_data, read_rows, tokenize
from .models import NGram
from .training import load_model, loader, neural_score, ngram_score


@torch.inference_mode()
def generate_ids(model, prompt_ids, settings, context_length, device, seed, greedy=False, fixed_length=False, count=None):
    ids = list(prompt_ids) or [BOS]
    result = []
    size = settings["max_new_tokens"] if count is None else count
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(size):
        context = ids[-context_length:]
        if isinstance(model, NGram):
            logits = np.log(model.distribution(context))
            logits[[PAD, BOS, UNK] + ([EOS] if fixed_length else [])] = -np.inf
            if greedy:
                token = int(np.argmax(logits))
            else:
                logits /= settings["temperature"]
                k = min(settings["top_k"], len(logits))
                if k > 0:
                    threshold = np.partition(logits, -k)[-k]
                    logits[logits < threshold] = -np.inf
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                token = int(rng.choice(len(probs), p=probs))
        else:
            logits = model(torch.tensor([context], device=device))[0, -1].clone()
            logits[[PAD, BOS, UNK] + ([EOS] if fixed_length else [])] = -float("inf")
            if greedy:
                token = int(logits.argmax())
            else:
                logits /= settings["temperature"]
                k = min(settings["top_k"], logits.numel())
                if k > 0:
                    threshold = logits.topk(k).values[-1]
                    logits[logits < threshold] = -float("inf")
                token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        ids.append(token)
        result.append(token)
        if token == EOS and not fixed_length:
            break
    return result


def benchmark_and_samples(model, vocab, config, device):
    settings = config["evaluation"]
    if (settings["temperature"] <= 0 or settings["max_new_tokens"] < 1 or settings["benchmark_tokens"] < 1
            or settings["benchmark_repeats"] < 1 or not settings["prompts"]):
        raise ValueError("Invalid decoding or benchmark settings")
    lookup = {w: i for i, w in enumerate(vocab["tokens"])}
    encoded = [[BOS] + [lookup.get(w, UNK) for w in tokenize(p)] for p in settings["prompts"]]
    context_length = config["model"]["context_length"]
    samples = []
    for i, (prompt, ids) in enumerate(zip(settings["prompts"], encoded)):
        generated = generate_ids(model, ids, settings, context_length, device, config["seed"] + i, fixed_length=True)
        samples.append({"prompt": prompt, "token_ids": generated,
                        "continuation": " ".join(vocab["tokens"][t] for t in generated if t != EOS),
                        "generated_tokens": len(generated), "fixed_length": True,
                        "special_tokens_suppressed": ["PAD", "BOS", "UNK", "EOS"]})
    generate_ids(model, encoded[0], settings, context_length, device, config["seed"],
                 greedy=True, fixed_length=True, count=settings["benchmark_tokens"])
    times = []
    tokens_per_repeat = len(encoded) * settings["benchmark_tokens"]
    for _ in range(settings["benchmark_repeats"]):
        sync(device)
        t = time.perf_counter()
        for ids in encoded:
            generate_ids(model, ids, settings, context_length, device, config["seed"],
                         greedy=True, fixed_length=True, count=settings["benchmark_tokens"])
        sync(device)
        times.append(time.perf_counter() - t)
    benchmark = {"seconds_per_repeat": times, "median_seconds": float(np.median(times)),
                 "tokens_per_repeat": tokens_per_repeat,
                 "tokens_per_second": tokens_per_repeat / float(np.median(times)),
                 "batch_size": 1, "warmup_workload": "one prompt, full benchmark token count",
                 "decoding": "greedy, fixed token count; PAD/BOS/UNK/EOS suppressed",
                 "context_length": context_length, "kv_cache": False,
                 "scope": "autoregressive generation including tensor creation and token selection; excludes initial tokenization/model loading",
                 "hardware_note": "n-gram CPU, neural model selected device; implementation comparison"}
    return benchmark, samples


def evaluate(run, data, split, output, device):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    manifest, vocab = load_data(data)
    model, checkpoint, meta = load_model(run, manifest, device)
    if meta["budget_limited"] and split == "test":
        raise ValueError("Preflight/batch-limited runs must not produce test results")
    out = fresh_dir(output)
    config = checkpoint["config"]
    seed_all(config["seed"])
    rows = read_rows(Path(data) / f"lm_{split}.jsonl")
    effective_device = torch.device("cpu") if isinstance(model, NGram) else device
    sync(effective_device)
    t = time.perf_counter()
    score = (ngram_score(model, BlockDataset(rows, config["model"]["context_length"]))
             if isinstance(model, NGram) else neural_score(model, loader(rows, config), device))
    sync(effective_device)
    score_seconds = time.perf_counter() - t
    benchmark, samples = benchmark_and_samples(model, vocab, config, effective_device)
    result = {"run_name": Path(run).name, "run": str(run), "split": split,
              "kind": checkpoint["kind"], "variant": checkpoint["variant"],
              "data_fingerprint": manifest["fingerprint"], "synthetic": manifest["synthetic"],
              "checkpoint_sha256": sha256(Path(run) / "best.pt"), "checkpoint_epoch": checkpoint["epoch"],
              "metrics": score, "scoring_seconds": score_seconds, "benchmark": benchmark,
              "decoding": config["evaluation"], "environment": environment(effective_device),
              "training": meta, "split_stats": manifest["stats"][f"lm_{split}"]}
    write_json(out / "lm_metrics.json", result)
    write_json(out / "generations.json", samples)
    return result
