"""Export tables and figures from recorded evidence, never invented results."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import fresh_dir, read_json, write_json


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root, output, split="test"):
    root = Path(root)
    lm = [read_json(p) for p in sorted(root.rglob("lm_metrics.json"))]
    lm = [r for r in lm if r["split"] == split]
    probes = [read_json(p) for p in sorted(root.rglob("probe_metrics.json"))]
    probes = [r for r in probes if split in r["metrics"]]
    if not lm and not probes:
        raise ValueError(f"No saved {split} metrics found under {root}")
    fingerprints = {r["data_fingerprint"] for r in lm + probes}
    if len(fingerprints) > 1:
        raise ValueError("Refusing to aggregate different dataset fingerprints; select one experiment root")
    if len({r["run_name"] for r in lm}) != len(lm) or len({r["name"] for r in probes}) != len(probes):
        raise ValueError("Duplicate run names in results; choose a single evaluation/probe root per comparison")
    out = fresh_dir(output)
    synthetic = any(r["synthetic"] for r in lm + probes)
    prefix = "SYNTHETIC SOFTWARE CHECK - " if synthetic else ""
    table = []
    for r in lm:
        tr = r["training"]
        table.append({"run": r["run_name"], "split": split, "cross_entropy": r["metrics"]["cross_entropy"],
                      "perplexity": r["metrics"]["perplexity"], "evaluated_tokens": r["metrics"]["tokens"],
                      "training_seconds": tr["training_seconds"], "validation_seconds": tr["validation_seconds"],
                      "parameters": tr.get("parameters"), "trainable_parameters": tr.get("trainable_parameters"),
                      "stored_count_entries": tr.get("stored_count_entries"), "checkpoint_bytes": tr["checkpoint_bytes"],
                      "tokens_per_second": r["benchmark"]["tokens_per_second"],
                      "device": r["environment"]["device"], "synthetic": synthetic})
    write_csv(out / "language_models.csv", table)
    if table:
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in table:
            ax.scatter(r["tokens_per_second"], r["cross_entropy"], s=75, label=r["run"])
        ax.set(xlabel="Autoregressive tokens / second (log scale)", ylabel=f"{split} cross-entropy (lower is better)",
               title=prefix + "Prediction quality and inference cost", xscale="log")
        ax.margins(x=0.3, y=0.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "quality_cost.png", dpi=160)
        plt.close(fig)
    histories = sorted(root.rglob("history.json"))
    if histories:
        fig, ax = plt.subplots(figsize=(9, 5))
        for path in histories:
            history = read_json(path)
            if not history:
                continue
            epochs = [h["epoch"] for h in history]
            line, = ax.plot(epochs, [h["val"]["cross_entropy"] for h in history], label=path.parent.name + " val")
            ax.plot(epochs, [h["train"]["cross_entropy"] for h in history], "--", color=line.get_color(), alpha=0.55)
        ax.set(xlabel="Epoch", ylabel="Token-weighted cross-entropy", title=prefix + "Learning curves (dashed: training)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "learning_curves.png", dpi=160)
        plt.close(fig)
    probe_table = [{"run": r["name"], "split": split, "accuracy": r["metrics"][split]["accuracy"],
                    "macro_f1": r["metrics"][split]["macro_f1"], "samples": r["metrics"][split]["samples"],
                    "selected_C": r["selected_C"], "feature_dimension": r["feature_dimension"],
                    "classifier_search_seconds": r["classifier_search_seconds"], "synthetic": synthetic} for r in probes]
    write_csv(out / "downstream.csv", probe_table)
    paired = []
    for p in probes:
        matches = [r for r in lm if r["checkpoint_sha256"] == p["checkpoint_sha256"]]
        for match in matches:
            paired.append({"run": p["name"], "lm_cross_entropy": match["metrics"]["cross_entropy"],
                           "classification_macro_f1": p["metrics"][split]["macro_f1"]})
    write_csv(out / "lm_vs_downstream.csv", paired)
    if paired:
        fig, ax = plt.subplots(figsize=(7, 5))
        for row in paired:
            ax.scatter(row["lm_cross_entropy"], row["classification_macro_f1"], s=75, label=row["run"])
        ax.set(xlabel="LM cross-entropy (lower is better)", ylabel="Classification macro-F1 (higher is better)", title=prefix + "Prediction vs representation")
        ax.margins(x=0.3, y=0.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "lm_vs_downstream.png", dpi=160)
        plt.close(fig)
    write_json(out / "summary.json", {"synthetic": synthetic, "split": split,
                                      "language_models": table, "downstream": probe_table, "paired": paired})
    return {"output": str(out), "language_models": len(table), "probes": len(probe_table), "synthetic": synthetic}
