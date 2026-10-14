"""Train-only vocabulary and auditable IMDb split construction (no downloads)."""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
import random
import re
import tarfile

import torch
from torch.utils.data import Dataset

from .common import fresh_dir, identity, read_json, sha256, write_json

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
TOKENIZER_VERSION = "html-lower-word-apostrophe-digit-punctuation-v1"


def tokenize(text):
    clean = re.sub(r"<[^>]+>", " ", html.unescape(text)).lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)*|[0-9]+|[^\w\s]", clean)


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_vocab(rows, size, min_count):
    if size <= len(SPECIAL) or min_count < 1:
        raise ValueError("Vocabulary size must exceed four specials; min_count must be positive")
    counts = Counter(token for row in rows for token in row["tokens"])
    words = sorted((w for w in counts if counts[w] >= min_count and w not in SPECIAL),
                   key=lambda w: (-counts[w], w))[:size - len(SPECIAL)]
    if not words:
        raise ValueError("No content words survive vocabulary filtering; reduce min_count")
    return {"tokens": SPECIAL + words, "counts": [0] * 4 + [counts[w] for w in words],
            "tokenizer": TOKENIZER_VERSION}


def read_archive(archive):
    groups = {"unsup": [], "train": [], "test": []}
    pattern = re.compile(r"(?:^|/)aclImdb/(train|test)/(unsup|pos|neg)/[^/]+\.txt$")
    # Read member bytes directly; do not extract any paths or links from an archive.
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            match = pattern.search(member.name)
            if not member.isfile() or not match:
                continue
            split, label = match.groups()
            if label == "unsup" and split != "train":
                continue
            if member.size > 10_000_000:
                raise ValueError(f"Unexpectedly large text member: {member.name}")
            content = stream.extractfile(member).read().decode("utf-8")
            tokens = tokenize(content)
            if not tokens:
                continue
            group = "unsup" if label == "unsup" else split
            groups[group].append({"id": member.name, "tokens": tokens,
                                  "text_hash": identity(tokens),
                                  "label": None if label == "unsup" else int(label == "pos")})
    for rows in groups.values():
        rows.sort(key=lambda r: r["id"])
    return groups


def unique_rows(rows, excluded):
    out, removed = [], 0
    for row in rows:
        if row["text_hash"] in excluded:
            removed += 1
        else:
            excluded.add(row["text_hash"])
            out.append(row)
    return out, removed


def stratified(rows, sizes, rng):
    if any(n <= 0 or n % 2 for n in sizes):
        raise ValueError("Classification split sizes must be positive even numbers")
    results = [[] for _ in sizes]
    for label in [0, 1]:
        candidates = [r for r in rows if r["label"] == label]
        rng.shuffle(candidates)
        required = sum(sizes) // 2
        if len(candidates) < required:
            raise ValueError(f"Need {required} unique label-{label} records, found {len(candidates)}")
        start = 0
        for result, count in zip(results, sizes):
            result.extend(candidates[start:start + count // 2])
            start += count // 2
    for result in results:
        rng.shuffle(result)
    return results


def prepare(archive, output, config, synthetic=False):
    c = config["data"]
    groups = read_archive(archive)
    seen = set()
    test, dropped_test = unique_rows(groups["test"], seen)
    train, dropped_train = unique_rows(groups["train"], seen)
    # All labeled records reserve their normalized text, not just the selected subset.
    unsup, dropped_unsup = unique_rows(groups["unsup"], seen)
    rng = random.Random(config["seed"])
    rng.shuffle(unsup)
    counts = [c[k] for k in ["lm_train", "lm_val", "lm_test"]]
    if min(counts) < 1 or c["max_lm_tokens"] < 1 or len(unsup) < sum(counts):
        raise ValueError(f"Requested {sum(counts)} LM documents; found {len(unsup)} after deduplication")
    rows = {}
    start = 0
    for split, count in zip(["train", "val", "test"], counts):
        rows[f"lm_{split}"] = [{**r, "tokens": r["tokens"][:c["max_lm_tokens"]]} for r in unsup[start:start + count]]
        start += count
    rows["cls_train"], rows["cls_val"] = stratified(train, [c["cls_train"], c["cls_val"]], rng)
    rows["cls_test"], = stratified(test, [c["cls_test"]], rng)
    vocab = build_vocab(rows["lm_train"], c["vocab_size"], c["min_count"])
    lookup = {w: i for i, w in enumerate(vocab["tokens"])}
    stats = {}
    for split, records in rows.items():
        for row in records:
            row["ids"] = [BOS] + [lookup.get(w, UNK) for w in row["tokens"]] + [EOS]
        token_count = sum(len(r["tokens"]) for r in records)
        unk_count = sum(r["ids"].count(UNK) for r in records)
        stats[split] = {"documents": len(records), "content_tokens": token_count,
                        "unk_tokens": unk_count, "unk_rate": unk_count / token_count}
    path = fresh_dir(output)
    write_json(path / "vocab.json", vocab)
    for split, records in rows.items():
        write_rows(path / f"{split}.jsonl", records)
    files = {p.name: sha256(p) for p in sorted(path.iterdir()) if p.is_file()}
    manifest = {"format_version": 1, "synthetic": synthetic,
                "source": str(archive), "source_sha256": sha256(archive),
                "seed": config["seed"], "data_config": c, "tokenizer": TOKENIZER_VERSION,
                "dedup_dropped": {"test": dropped_test, "train": dropped_train, "unsup": dropped_unsup},
                "dedup_policy": "normalized full text; labeled test > labeled train > unsup",
                "stats": stats, "files": files, "vocab_hash": identity(vocab)}
    manifest["fingerprint"] = identity({"files": files, "synthetic": synthetic, "data_config": c})
    write_json(path / "manifest.json", manifest)
    return manifest


def load_data(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    required = {"vocab.json"} | {f"{task}_{split}.jsonl" for task in ["lm", "cls"] for split in ["train", "val", "test"]}
    if set(manifest["files"]) != required:
        raise ValueError("Incomplete or unexpected prepared data file set")
    expected = identity({"files": manifest["files"], "synthetic": manifest["synthetic"], "data_config": manifest["data_config"]})
    if expected != manifest["fingerprint"]:
        raise ValueError("Invalid data manifest fingerprint")
    for name, digest in manifest["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Prepared data changed: {name}. Prepare a new dataset instead.")
    vocab = read_json(path / "vocab.json")
    if identity(vocab) != manifest["vocab_hash"]:
        raise ValueError("Vocabulary identity mismatch")
    return manifest, vocab


class BlockDataset(Dataset):
    def __init__(self, rows, length):
        if length < 1:
            raise ValueError("Block length must be positive")
        self.rows, self.length = rows, length
        self.blocks = [(i, start) for i, row in enumerate(rows) for start in range(0, len(row["ids"]) - 1, length)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        i, start = self.blocks[index]
        ids = self.rows[i]["ids"][start:start + self.length + 1]
        return torch.tensor(ids[:-1]), torch.tensor(ids[1:])


def collate_blocks(batch):
    xs, ys = zip(*batch)
    return (torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=PAD),
            torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=PAD))
