"""Explicit entry points: no automatic downloads, training or test evaluation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import urllib.request

import torch

from .common import environment, get_device, load_config, sha256, write_json


ASSETS = {
    "imdb": ("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", "aclImdb_v1.tar.gz"),
    "glove": ("https://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip"),
}


def download(asset, destination):
    url, name = ASSETS[asset]
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    if path.exists() or path.with_suffix(path.suffix + ".part").exists():
        raise FileExistsError(f"Download target or partial file exists: {path}; choose another destination")
    partial = path.with_suffix(path.suffix + ".part")
    received = 0
    print(f"Explicit download: {url} -> {path}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("xb") as output:
        expected = response.headers.get("Content-Length")
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            output.write(chunk)
            received += len(chunk)
        if expected and received != int(expected):
            raise IOError(f"Incomplete download: {received}/{expected} bytes; partial file retained")
    partial.replace(path)
    info = {"url": url, "path": str(path), "bytes": received, "sha256": sha256(path),
            "hash_note": "Locally measured content hash; not an independently verified publisher checksum"}
    write_json(path.with_suffix(path.suffix + ".source.json"), info)
    return info


def parser():
    p = argparse.ArgumentParser(description="SLM Core: explicit, reproducible experiment stages")
    p.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads (default: 4)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect local environment without training")
    d = sub.add_parser("download", help="Explicit network download; never called by other commands")
    d.add_argument("--asset", choices=ASSETS, required=True)
    d.add_argument("--dest", default="data/raw")
    prepare = sub.add_parser("prepare", help="Prepare an existing local IMDb tar.gz")
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--synthetic", action="store_true", help="Mark software-test fixture, not experiment evidence")
    verify = sub.add_parser("verify-splits", help="Verify exact ordered reviews against the published experiment")
    verify.add_argument("--data", required=True)
    verify.add_argument("--reference", default="dataset_splits/manifest.json")
    verify.add_argument("--archive", help="Optionally verify the raw IMDb archive bytes as well")
    own = sub.add_parser("embeddings", help="Train own Skip-gram or import existing GloVe file/ZIP")
    own.add_argument("--method", choices=["own", "glove"], required=True)
    own.add_argument("--source", help="Existing GloVe text file or ZIP; no download")
    tr = sub.add_parser("train", help="Train one language model, using validation only")
    tr.add_argument("--kind", choices=["ngram", "rnn", "lstm", "transformer"], required=True)
    tr.add_argument("--embedding", help="Prepared frozen embedding directory (LSTM only)")
    tr.add_argument("--run", required=True)
    tr.add_argument("--resume", action="store_true", help="Resume same configuration at last complete epoch")
    ev = sub.add_parser("evaluate", help="Explicit validation/test scoring, generation and benchmark")
    ev.add_argument("--run", required=True)
    ev.add_argument("--split", choices=["val", "test"], required=True)
    pr = sub.add_parser("probe", help="Frozen-representation classifiers; validation only unless --test")
    pr.add_argument("--runs", nargs="+", required=True)
    pr.add_argument("--test", action="store_true", help="Evaluate held-out test after validation C selection")
    pr.add_argument("--tfidf", action="store_true", help="Optional additional traditional baseline")
    pf = sub.add_parser("preflight", help="Explicit bounded training/timing pilot, never evaluates test")
    pf.add_argument("--kind", choices=["rnn", "lstm", "transformer"], default="transformer")
    pf.add_argument("--batches", type=int, default=20)
    pf.add_argument("--run", required=True)
    rep = sub.add_parser("report", help="Export tables and figures from saved results")
    rep.add_argument("--root", required=True)
    rep.add_argument("--output", required=True)
    rep.add_argument("--split", choices=["val", "test"], default="test")
    for command in [prepare, own, tr, pr, pf]:
        command.add_argument("--config", default="configs/server.json")
    for command in [own, tr, ev, pr, pf]:
        command.add_argument("--data", required=True)
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    for command in [own, ev, pr]:
        command.add_argument("--output", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "doctor":
        result = environment()
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10)
            result["nvidia_smi"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_smi"] = "not available"
    elif args.command == "download":
        result = download(args.asset, args.dest)
    elif args.command == "verify-splits":
        from .splits import verify_splits
        result = verify_splits(args.data, args.reference, args.archive)
    elif args.command == "prepare":
        from .data import prepare
        result = prepare(args.archive, args.output, load_config(args.config), args.synthetic)
    elif args.command == "embeddings":
        from .embeddings import import_glove, train_own
        config = load_config(args.config)
        if args.method == "glove":
            if not args.source:
                raise ValueError("--source is required for public vectors")
            result = import_glove(args.data, args.source, args.output, config)
        else:
            result = train_own(args.data, args.output, config, get_device(args.device))
    elif args.command == "train":
        from .training import train
        result = train(args.data, args.run, args.kind, load_config(args.config), get_device(args.device), args.embedding, args.resume)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.run, args.data, args.split, args.output, get_device(args.device))
    elif args.command == "probe":
        from .probe import probe
        result = probe(args.runs, args.data, args.output, load_config(args.config), get_device(args.device), args.test, args.tfidf)
    elif args.command == "preflight":
        from .data import read_rows
        from .training import train
        config = load_config(args.config)
        if args.batches < 1:
            raise ValueError("Preflight batches must be positive")
        short = copy.deepcopy(config)
        short["train"].update(epochs=1, max_train_batches=args.batches, max_val_batches=5)
        result = train(args.data, args.run, args.kind, short, get_device(args.device))
        from .common import read_json
        history = read_json(Path(args.run) / "history.json")
        tokens = history[0]["train"]["tokens"]
        total = sum(len(r["ids"]) - 1 for r in read_rows(Path(args.data) / "lm_train.jsonl"))
        estimate = {"kind": args.kind, "observed_tokens": tokens,
                    "observed_train_seconds": result["training_seconds"],
                    "estimated_training_only_seconds_for_this_model": total / tokens * result["training_seconds"] * config["train"]["epochs"],
                    "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
                    "caveat": "Rough extrapolation from short pilot, includes startup effects; excludes validation/I/O/embeddings/other models; not a cost quote"}
        write_json(Path(args.run) / "estimate.json", estimate)
        result = estimate
    else:
        from .report import report
        result = report(args.root, args.output, args.split)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result
