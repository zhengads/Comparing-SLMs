# Comapring SLMs

**DSA4213 Assignment 1** — Comparing small language models: prediction, embeddings, and sentiment transfer.

Author: **ZHENG MINGLIANG**.

This repository contains the experimental implementation, fixed configuration, and reproduction entry point for the assignment. It compares an interpolated trigram, RNN, LSTM, and causal Transformer on IMDb, then compares LSTM input embeddings and frozen representations for sentiment classification.

## Quick start

Use Python **3.12** (the recorded experiment used 3.12.3). Run these commands in a terminal:

```bash
git clone https://github.com/zhengads/Comapring-SLMs.git
cd Comapring-SLMs
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the recorded PyTorch release and dependencies. For an NVIDIA GPU with a driver supporting CUDA 13.0:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m core doctor
```

For a CPU-only Linux/Windows environment, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

On macOS, use `python -m pip install torch==2.12.1` and select `--device cpu`. See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2121) for platform-specific wheels. Do not install torchvision or torchaudio; this project does not need them.

### One-command reproduction

After installation, run:

```bash
python reproduce.py --download --device cuda
```

This explicitly downloads missing official archives, prepares and verifies the exact published data splits, trains both embedding tables and all six language-model settings, evaluates validation results, fits the frozen-encoder classifiers, and evaluates the fixed design on the held-out test sets. It then exports CSV tables and diagnostic figures. GloVe vectors are **imported**, not retrained from their original external corpus.

To run entirely on CPU:

```bash
python reproduce.py --download --device cpu
```

CPU execution can be substantially slower. CUDA execution fails clearly if CUDA is unavailable; it does not silently fall back to CPU. The downloads total approximately 0.9 GB; allow several GB of free disk space for prepared data and checkpoints. The recorded model-training loops plus own-vector training took about 12 minutes on the recorded GPU environment, excluding downloads, preprocessing, evaluation, and I/O. This is a historical measurement, not a runtime guarantee for a different machine.

Preview the commands without downloads, training, or filesystem changes:

```bash
python reproduce.py --download --device cuda --dry-run
```

Already have the official archives? Put `aclImdb_v1.tar.gz` and `glove.6B.zip` in `data/raw/`, then omit `--download`. A different cache can be supplied with `--raw /path/to/archives`.

Outputs default to `artifacts/reproduction/`. Existing outputs are never overwritten. For another run:

```bash
python reproduce.py --device cuda --output artifacts/reproduction-2
```

If interrupted, inspect `pipeline-status.json`. The full launcher requires a fresh output directory; advanced users can continue individual stages using the commands printed by the launcher. Individual model training supports `python -m core train ... --resume` with the same configuration and run directory. A download ending in `.part` is incomplete; use a clean archive cache or remove only that incomplete file before retrying.

## Experiment design

| Part | Comparison |
| --- | --- |
| I | Interpolated add-0.1 trigram, tanh RNN, LSTM, causal Transformer |
| II | LSTM with random trainable embeddings, corpus-trained frozen Skip-gram, or frozen GloVe 6B 100d |
| III | Frozen Part I LSTM and Transformer, mean pooling, standardized features, logistic regression |

The random LSTM is reused across Parts I and II, giving **six LM settings**, not seven. The two pretrained LSTMs start their remaining parameters from scratch. The classifier uses the original random-embedding LSTM and Transformer; it does not use the Part II models.

The full fixed recipe is in [`configs/server.json`](configs/server.json):

- LM train/validation/test: 10,000 / 1,000 / 1,000 reviews from IMDb's unsupervised pool.
- Sentiment train/validation/test: 8,000 / 2,000 / 5,000 labeled reviews, balanced by class. Test reviews come from the official test pool.
- Exact normalized-text deduplication; all labeled texts excluded from the LM pool. This does not establish semantic or external-pretraining independence.
- Vocabulary: 12,000 entries, built only on LM training text. At most 512 content tokens per LM review; within-review prediction blocks of length 128.
- Neural models: 100-dimensional embeddings, two layers of width 256, five epochs, seed 4213, batch size 32, dropout 0.2, AdamW. Validation CE selects checkpoints.
- Own embeddings: three epochs of Skip-gram with negative sampling, trained only on LM training text.
- Sentiment: first 128 content tokens, masked mean pooling, frozen encoders. Standardization is fitted on training features; validation macro-F1 selects regularization C. Validation is not merged into training afterward.
- Test evaluation occurs after training and validation selection, with no test-driven parameter search in the launcher.

## Output files

Within `artifacts/reproduction/`:

```text
prepared/manifest.json                  Data fingerprint, split hashes, tokenizer, counts
embeddings/own/                         Learned in-domain vectors and metadata
embeddings/glove/                       Imported vectors and coverage metadata
experiment/runs/<model>/history.json    Training/validation CE and timing per epoch
experiment/runs/<model>/best.pt         Validation-selected checkpoint
experiment/evaluation/test/<model>/     LM metrics, generations, generation benchmarks
experiment/probes-test/<encoder>/       Probe metrics, predictions, errors, slices
experiment/report-test/                 Summary CSV tables and diagnostic PNG figures
pipeline-status.json                   Launcher stage and completion/failure state
```

`report-test/language_models.csv` and `downstream.csv` contain the main numerical comparisons. `generations.json` contains the three shared prompts and saved 64-token continuations. The generated PNGs are diagnostic plots; the manuscript uses separately typeset figures from the same recorded values.

## Recorded results and reproducibility

[`reference_results/`](reference_results/) contains the original report's compact CSV summaries and environment/data identity, so a reproduction can be compared with the recorded run without distributing model checkpoints or datasets.

| Model | Recorded test PPL | Recorded sentiment accuracy |
| --- | ---: | ---: |
| Trigram | 415.38 | Not evaluated |
| RNN | 147.63 | Not evaluated |
| LSTM, random trainable | 182.17 | 65.80% |
| Transformer | 133.24 | 68.44% |
| LSTM, own frozen | 162.67 | Not evaluated |
| LSTM, GloVe frozen | 185.28 | Not evaluated |

The recorded environment used Python 3.12.3, PyTorch 2.12.1+cu130, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2, Matplotlib 3.9.2, and an RTX 4080 SUPER device. The required non-PyTorch packages are pinned in `requirements.txt`. The model, training, and scoring logic comes from the executed source snapshot. The Python package has been renamed to `core`; the release also adds split verification and orchestration utilities without changing the experimental recipe.

The original Linux prepared-data fingerprint is `f8c2391b65a73cb56e39d892f73eb55ebc75aab9a97ca142544d79d6cd01b42b`. The verification command below checks exact review order and canonical record content across platforms; different newline formatting may change raw file hashes without changing the data. Identical data and seeds do not guarantee bitwise-identical learned weights or timings across hardware, drivers, and library versions. The results use one LM seed and five epochs, with convergence not established.

CE sums negative log likelihood over all non-PAD targets, including UNK and EOS, then divides by their token count; PPL is exp(CE). Generation speed uses batch 1, no KV cache or recurrent-state reuse, and a median over three synchronized repetitions. Trigram runs on CPU while neural models use the selected device. These are implementation-specific timings. Sentiment accuracy covers all 5,000 test reviews; macro-F1 equally weights the two class-level F1 scores. Lower PPL does not directly measure generation coherence or sentiment understanding.

## Exact published data splits

[`dataset_splits/`](dataset_splits/) contains ordered review-ID lists exported directly from the prepared data used in the reported experiment. These are the actual selected reviews, not newly sampled lists. No review text is redistributed.

| File | Task | Split | Reviews |
| --- | --- | --- | ---: |
| `lm_train.csv` | Language modeling | Training | 10,000 |
| `lm_val.csv` | Language modeling | Validation | 1,000 |
| `lm_test.csv` | Language modeling | Test | 1,000 |
| `cls_train.csv` | Sentiment classification | Training | 8,000 |
| `cls_val.csv` | Sentiment classification | Validation | 2,000 |
| `cls_test.csv` | Sentiment classification | Test | 5,000 |

Each CSV preserves the original prepared-record order. `review_id` is the member path inside the official IMDb archive (for example, `aclImdb/train/unsup/31897_0.txt`). `index` is zero-based. Labels are `0` for negative, `1` for positive, and empty for unsupervised LM data. The CSVs also provide:

- `normalized_full_text_sha256`: the original checksum of the full normalized token list, computed before LM truncation. It is not a checksum of raw review bytes.
- `prepared_record_sha256`: the checksum of the entire prepared JSON record, including ID, label, normalized-text checksum, retained tokens, and vocabulary IDs. This detects changes to truncation, tokenization, and encoding as well as split membership.

[`dataset_splits/manifest.json`](dataset_splits/manifest.json) records the IMDb archive SHA-256, tokenizer and configuration, vocabulary identity, each CSV checksum, original prepared-file checksums, and canonical ordered-record checksums. Canonical records use Python `json.dumps(record, sort_keys=True, ensure_ascii=False)` encoded as UTF-8 with its default separators; the split checksum concatenates each canonical record followed by a single LF byte. Original byte hashes and the original dataset fingerprint are retained separately.

### Recreate and verify without training

After installing dependencies, the following commands download IMDb only, regenerate the fixed splits with the original recipe, and verify every published ID and checksum:

```bash
python -m core download --asset imdb --dest data/raw
python -m core prepare --archive data/raw/aclImdb_v1.tar.gz --output artifacts/data-only/prepared --config configs/server.json
python -m core verify-splits --data artifacts/data-only/prepared --reference dataset_splits/manifest.json --archive data/raw/aclImdb_v1.tar.gz
```

Skip the first command if the archive already exists. Preparation requires a fresh output directory. No GloVe download or model training is needed for these commands.

To verify a full reproduction's prepared data later:

```bash
python -m core verify-splits --data artifacts/reproduction/prepared --reference dataset_splits/manifest.json
```

A successful check prints `"verified": true`, `"total_reviews": 27000`, and `"canonical_match": true` for every split. `original_file_bytes_match` may be false for equivalent newline/JSON formatting; this does not bypass record-content checks. Wrong IDs, order, labels, record contents, vocabulary, configuration, or source identity cause a nonzero exit. The optional `--archive` checks the raw archive itself, rather than only its recorded identity.

The one-command launcher automatically runs this verification immediately after preparation and before embedding/model training. Its published reference is tied to the assignment configuration; custom data settings will intentionally fail verification. Use the individual CLI commands for a separate experiment instead of treating altered splits as a reproduction.

## Repository contents

```text
core/                Data preparation, models, embeddings, training, evaluation, probes, reporting
configs/server.json     The fixed assignment configuration
scripts/run_experiments.py  Individual experiment-stage runner
reproduce.py            Cross-platform end-to-end launcher
requirements.txt        Pinned non-PyTorch dependencies
pyproject.toml          Package metadata and CLI entry point
reference_results/      Compact original results and environment/data identity
dataset_splits/         Actual ordered review IDs, per-record hashes, and reference manifest
```

Unit tests, synthetic smoke-test fixtures, development notes, local paths, caches, credentials, raw datasets, and checkpoints are intentionally excluded from this submission repository. No paid API or cloud service is required by the code.

## Data and method sources

- IMDb: Maas et al. (2011), [Learning Word Vectors for Sentiment Analysis](https://aclanthology.org/P11-1015/); [official dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
- Transformer: Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Skip-gram: Mikolov et al. (2013), [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).
- GloVe: Pennington et al. (2014), [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/); [official vectors and terms](https://nlp.stanford.edu/projects/glove/).

Data and pretrained vectors are downloaded from their official sources and are not redistributed here. Respect the respective dataset and vector terms.

## AI assistance

OpenAI Codex assisted with implementation, debugging, experimental workflow, inspection of saved outputs, documentation, and report revision. The author directed the work, reviewed the AI-assisted content, and retains responsibility for the submitted implementation, evidence, and conclusions.
