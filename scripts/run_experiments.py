"""Run one explicit stage of the real experiment. No downloads or implicit next stage."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from core.common import get_device, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["train", "evaluate", "probe", "report"], required=True)
    p.add_argument("--config", default="configs/server.json")
    p.add_argument("--data", default="data/prepared/imdb-small")
    p.add_argument("--root", default="artifacts/experiment-v1")
    p.add_argument("--embeddings", default="artifacts/embeddings")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--test", action="store_true", help="Explicitly use test data for evaluate/probe; default validation")
    p.add_argument("--tfidf", action="store_true", help="Optional additional classifier baseline")
    args = p.parse_args()
    torch.set_num_threads(4)
    config, root = load_config(args.config), Path(args.root)
    device = get_device(args.device) if args.stage != "report" else torch.device("cpu")
    specs = [("ngram", "ngram", None), ("rnn_random", "rnn", None),
             ("lstm_random", "lstm", None), ("transformer_random", "transformer", None),
             ("lstm_own", "lstm", Path(args.embeddings) / "own"),
             ("lstm_glove", "lstm", Path(args.embeddings) / "glove")]
    if args.stage == "train":
        from core.data import load_data
        from core.embeddings import load_table
        from core.training import train
        manifest, _ = load_data(args.data)
        for name, _, table in specs:
            if (root / "runs" / name).exists():
                raise FileExistsError("Suite requires fresh run paths. Resume interrupted runs individually; see README.md.")
            if table:
                load_table(table, manifest, config["model"]["embedding_dim"])
        for name, kind, table in specs:
            train(args.data, root / "runs" / name, kind, config, device, table)
    elif args.stage == "evaluate":
        from core.evaluation import evaluate
        split = "test" if args.test else "val"
        for name, _, _ in specs:
            evaluate(root / "runs" / name, args.data, split, root / "evaluation" / split / name, device)
    elif args.stage == "probe":
        from core.probe import probe
        probe([root / "runs" / "lstm_random", root / "runs" / "transformer_random"], args.data,
              root / ("probes-test" if args.test else "probes-val"), config, device, args.test, args.tfidf)
    else:
        from core.report import report
        split = "test" if args.test else "val"
        # Collect only one probe phase to avoid double-counting validation-only and test probes.
        report(root, root / ("report-test" if args.test else "report-val"), split)


if __name__ == "__main__":
    main()
