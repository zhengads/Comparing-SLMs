"""Frozen encoder classification with validation-only regularization selection."""
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from .common import environment, fresh_dir, seed_all, sha256, sync, write_json
from .data import PAD, UNK, load_data, read_rows, write_rows
from .models import NGram, pool_states
from .training import load_model


def classification_metrics(labels, predictions):
    return {"samples": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


@torch.inference_mode()
def extract_features(model, rows, config, device):
    model.eval()
    model.requires_grad_(False)
    result = []
    for start in range(0, len(rows), config["batch_size"]):
        sequences = [torch.tensor(r["ids"][1:-1][:config["max_tokens"]]) for r in rows[start:start + config["batch_size"]]]
        ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD).to(device)
        vectors = pool_states(model.states(ids), ids, config["pooling"])
        result.append(vectors.cpu().numpy())
    return np.concatenate(result)


def identity_tokens(tokens):
    return tokens


def fit_classifier(x_train, y_train, x_val, y_val, config, seed, dense):
    trials, selected, best = [], None, -1.0
    with threadpool_limits(limits=4):
        for c in config["c_values"]:
            classifier = LogisticRegression(C=c, max_iter=config["max_iter"], random_state=seed)
            pipeline = make_pipeline(StandardScaler(), classifier) if dense else classifier
            with warnings.catch_warnings(record=True) as found:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(x_train, y_train)
            if any(issubclass(w.category, ConvergenceWarning) for w in found):
                raise RuntimeError("Logistic regression failed to converge; increase probe.max_iter before test evaluation")
            score = classification_metrics(y_val, pipeline.predict(x_val))
            trials.append({"C": c, **score})
            if score["macro_f1"] > best:
                best, selected = score["macro_f1"], pipeline
                best_c = c
    if selected is None:
        raise ValueError("At least one positive classifier C is required")
    return selected, best_c, trials


def prediction_rows(rows, predicted, probabilities, max_tokens):
    results = []
    for row, pred, probability in zip(rows, predicted, probabilities):
        ids = row["ids"][1:-1][:max_tokens]
        results.append({"id": row["id"], "label": row["label"], "prediction": int(pred),
                        "positive_probability": float(probability), "correct": int(pred) == row["label"],
                        "full_length": len(row["tokens"]), "used_length": len(ids),
                        "truncated": len(row["tokens"]) > max_tokens,
                        "unk_rate": ids.count(UNK) / len(ids),
                        "contains_negation_marker": any(w in {"not", "never", "no", "isn't", "wasn't", "don't"} for w in row["tokens"][:max_tokens]),
                        "text_used": " ".join(row["tokens"][:max_tokens])})
    return results


def slices(predictions):
    groups = {
        "length_1_64": lambda r: r["full_length"] <= 64,
        "length_65_128": lambda r: 64 < r["full_length"] <= 128,
        "length_over_128": lambda r: r["full_length"] > 128,
        "unk_at_most_5_percent": lambda r: r["unk_rate"] <= 0.05,
        "unk_over_5_percent": lambda r: r["unk_rate"] > 0.05,
        "contains_negation_marker_not_a_semantic_test": lambda r: r["contains_negation_marker"],
    }
    out = {}
    for name, predicate in groups.items():
        subset = [r for r in predictions if predicate(r)]
        out[name] = (classification_metrics([r["label"] for r in subset], [r["prediction"] for r in subset])
                     if subset else {"samples": 0, "accuracy": None, "macro_f1": None, "confusion_matrix": [[0, 0], [0, 0]]})
    return out


def probe(runs, data, output, config, device, test=False, tfidf=False):
    if len(runs) < 2:
        raise ValueError("Provide at least two neural runs for the required representation comparison")
    names = [Path(r).name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError("Run names must be unique")
    manifest, vocab = load_data(data)
    # Fail before starting expensive extraction if any comparison is incompatible.
    for run in runs:
        from .common import read_json
        info = read_json(Path(run) / "run.json")
        if info["kind"] == "ngram" or info["data_fingerprint"] != manifest["fingerprint"] or info["status"] != "complete":
            raise ValueError("Probe requires completed neural runs on this exact prepared dataset")
        if info["budget_limited"]:
            raise ValueError("Preflight runs are not downstream experiment inputs")
        if config["probe"]["max_tokens"] > info["signature"]["config"]["model"]["context_length"]:
            raise ValueError("Probe input exceeds an encoder's context length")
    out = fresh_dir(output)
    seed_all(config["seed"])
    split_names = ["train", "val"] + (["test"] if test else [])
    rows = {s: read_rows(Path(data) / f"cls_{s}.jsonl") for s in split_names}
    labels = {s: np.array([r["label"] for r in rows[s]]) for s in split_names}
    outputs = []
    options = list(zip(names, runs)) + ([("tfidf", None)] if tfidf else [])
    for name, run in options:
        path = fresh_dir(out / name)
        t = time.perf_counter()
        feature_times = {}
        if run is not None:
            model, _, _ = load_model(run, manifest, device)
            if isinstance(model, NGram):
                raise ValueError("n-gram has no neural hidden-state representation")
            features = {}
            for split in split_names:
                sync(device)
                t_split = time.perf_counter()
                features[split] = extract_features(model, rows[split], config["probe"], device)
                sync(device)
                feature_times[split] = time.perf_counter() - t_split
            checkpoint_hash = sha256(Path(run) / "best.pt")
            del model
        else:
            vectorizer = TfidfVectorizer(tokenizer=identity_tokens, preprocessor=None,
                                         token_pattern=None, lowercase=False, ngram_range=(1, 2),
                                         max_features=30000, sublinear_tf=True)
            texts = {s: [r["tokens"][:config["probe"]["max_tokens"]] for r in rows[s]] for s in split_names}
            features = {"train": vectorizer.fit_transform(texts["train"])}
            features.update({s: vectorizer.transform(texts[s]) for s in split_names if s != "train"})
            joblib.dump(vectorizer, path / "vectorizer.joblib")
            checkpoint_hash = None
        feature_seconds = time.perf_counter() - t
        t = time.perf_counter()
        classifier, c, trials = fit_classifier(features["train"], labels["train"], features["val"], labels["val"],
                                               config["probe"], config["seed"], dense=run is not None)
        fit_seconds = time.perf_counter() - t
        joblib.dump(classifier, path / "classifier.joblib")
        split_metrics, slice_metrics = {}, {}
        for split in split_names[1:]:
            predicted = classifier.predict(features[split])
            probs = classifier.predict_proba(features[split])[:, 1]
            records = prediction_rows(rows[split], predicted, probs, config["probe"]["max_tokens"])
            write_rows(path / f"{split}_predictions.jsonl", records)
            write_json(path / f"{split}_errors.json", [r for r in records if not r["correct"]][:30])
            split_metrics[split] = classification_metrics(labels[split], predicted)
            slice_metrics[split] = slices(records)
        result = {"name": name, "synthetic": manifest["synthetic"], "data_fingerprint": manifest["fingerprint"],
                  "checkpoint_sha256": checkpoint_hash, "encoder_run": str(run) if run else None,
                  "frozen_encoder": run is not None, "config": config["probe"], "seed": config["seed"],
                  "feature_dimension": features["train"].shape[1], "selected_C": c, "validation_trials": trials,
                  "feature_seconds_including_model_load": feature_seconds, "extraction_seconds_by_split": feature_times,
                  "classifier_search_seconds": fit_seconds, "metrics": split_metrics, "slices": slice_metrics,
                  "environment": environment(device if run else "cpu"),
                  "train_samples": len(rows["train"]), "label_order": ["negative", "positive"]}
        write_json(path / "probe_metrics.json", result)
        outputs.append(result)
        print(f"Probe {name}: C={c}, val macro-F1={split_metrics['val']['macro_f1']:.4f}", flush=True)
    write_json(out / "comparison.json", outputs)
    return outputs
