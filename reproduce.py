"""Reproduce the fixed DSA4213 Assignment 1 experiment from official data."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def build_commands(args):
    """Build the complete sequence without importing ML libraries or running it."""
    output = Path(args.output).resolve()
    raw = Path(args.raw).resolve()
    config = Path(args.config).resolve()
    data, embeddings, experiment = output / "prepared", output / "embeddings", output / "experiment"
    python = sys.executable
    cli = [python, "-m", "core"]
    suite = [python, str(ROOT / "scripts/run_experiments.py"), "--config", str(config),
             "--data", str(data), "--embeddings", str(embeddings), "--root", str(experiment),
             "--device", args.device]
    commands = []
    for asset, filename in [("imdb", "aclImdb_v1.tar.gz"), ("glove", "glove.6B.zip")]:
        if not (raw / filename).is_file():
            if not args.download:
                raise ValueError(f"Missing {raw / filename}. Supply official archives or use --download.")
            commands.append(cli + ["download", "--asset", asset, "--dest", str(raw)])
    commands.append(cli + ["prepare", "--archive", str(raw / "aclImdb_v1.tar.gz"),
                           "--output", str(data), "--config", str(config)])
    commands.append(cli + ["verify-splits", "--data", str(data),
                           "--reference", str(ROOT / "dataset_splits/manifest.json"),
                           "--archive", str(raw / "aclImdb_v1.tar.gz")])
    commands.append(cli + ["embeddings", "--method", "own", "--data", str(data),
                           "--config", str(config), "--device", args.device, "--output", str(embeddings / "own")])
    commands.append(cli + ["embeddings", "--method", "glove", "--source", str(raw / "glove.6B.zip"),
                           "--data", str(data), "--config", str(config), "--output", str(embeddings / "glove")])
    commands.append(suite + ["--stage", "train"])
    for final_test in [False, True]:
        for stage in ["evaluate", "probe", "report"]:
            commands.append(suite + ["--stage", stage] + (["--test"] if final_test else []))
    return output, commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--download", action="store_true", help="Download missing official IMDb/GloVe archives")
    parser.add_argument("--raw", default=str(ROOT / "data/raw"), help="Archive cache directory")
    parser.add_argument("--output", default=str(ROOT / "artifacts/reproduction"), help="New, empty output directory")
    parser.add_argument("--config", default=str(ROOT / "configs/server.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; no downloads or training")
    args = parser.parse_args(argv)
    output, commands = build_commands(args)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}. Select a new --output; existing results are never overwritten.")
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        print("Dry run only. No files downloaded and no experiments executed.")
        return
    # Validate the installed environment and requested device before large downloads.
    from core.common import get_device, load_config
    load_config(args.config)
    get_device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    status = output / "pipeline-status.json"
    for index, command in enumerate(commands, 1):
        status.write_text(json.dumps({"stage": index, "total": len(commands), "command": command,
                                      "status": "running"}, indent=2), encoding="utf-8")
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        except subprocess.CalledProcessError as error:
            status.write_text(json.dumps({"stage": index, "command": command, "status": "failed",
                                          "exit_code": error.returncode}, indent=2), encoding="utf-8")
            raise
    status.write_text(json.dumps({"status": "complete", "stages": len(commands)}, indent=2), encoding="utf-8")
    print(f"Complete. Test tables and figures: {output / 'experiment/report-test'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
