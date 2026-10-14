"""Verify prepared reviews against the published, ordered experiment split lists."""
import csv
import hashlib
import json
from pathlib import Path

from .common import identity, read_json, sha256
from .data import load_data

SPLITS = ("lm_train", "lm_val", "lm_test", "cls_train", "cls_val", "cls_test")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def verify_splits(data, reference, archive=None):
    """Fail on differences in membership, order, labels, tokens, IDs, or vocabulary."""
    data, reference = Path(data), Path(reference)
    expected = read_json(reference)
    actual, vocab = load_data(data)
    if expected.get("format_version") != 1 or set(expected["splits"]) != set(SPLITS):
        raise ValueError("Unsupported or incomplete reference split manifest")
    if actual["synthetic"]:
        raise ValueError("Synthetic data cannot match the recorded experiment")
    for key in ("seed", "tokenizer", "data_config", "vocab_hash"):
        if actual[key] != expected[key]:
            raise ValueError(f"Prepared data differs from reference: {key}")
    if identity(vocab) != expected["vocab_hash"]:
        raise ValueError("Vocabulary content differs from reference")
    if actual["source_sha256"] != expected["source_archive_sha256"]:
        raise ValueError("Source archive identity differs from reference")
    if archive is not None and sha256(archive) != expected["source_archive_sha256"]:
        raise ValueError("Supplied archive SHA-256 differs from reference")
    result, seen = {}, set()
    for split in SPLITS:
        spec = expected["splits"][split]
        listing = (reference.parent / spec["id_list"]).resolve()
        if listing.parent != reference.parent.resolve():
            raise ValueError("ID list must be next to its reference manifest")
        if sha256(listing) != spec["id_list_sha256"]:
            raise ValueError(f"Reference ID list checksum mismatch: {split}")
        with listing.open(encoding="utf-8", newline="") as stream:
            entries = list(csv.DictReader(stream))
        with (data / f"{split}.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(entries) != spec["count"] or len(records) != spec["count"]:
            raise ValueError(f"Review count differs from reference: {split}")
        digest = hashlib.sha256()
        for index, (entry, record) in enumerate(zip(entries, records)):
            label = "" if record["label"] is None else str(record["label"])
            values = {"index": str(index), "review_id": record["id"], "label": label,
                      "normalized_full_text_sha256": record["text_hash"],
                      "prepared_record_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()}
            if entry != values:
                raise ValueError(f"Review content, label, ID, or order mismatch: {split}, index {index}")
            if record["id"] in seen:
                raise ValueError(f"Duplicate review across splits: {record['id']}")
            seen.add(record["id"])
            digest.update(canonical_bytes(record) + b"\n")
        if digest.hexdigest() != spec["canonical_records_sha256"]:
            raise ValueError(f"Canonical split checksum mismatch: {split}")
        result[split] = {"reviews": len(records), "canonical_match": True,
                         "original_file_bytes_match": sha256(data / f"{split}.jsonl") == spec["original_prepared_file_sha256"]}
    return {"verified": True, "total_reviews": len(seen), "splits": result,
            "original_data_fingerprint": expected["original_data_fingerprint"],
            "prepared_data_fingerprint": actual["fingerprint"],
            "note": "Canonical checks accept equivalent JSON and newline formatting; original byte hashes are reported separately."}
