"""Causal, untied-output neural LMs and an interpolated add-k n-gram LM."""
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

from .data import PAD, BOS, EOS


class NeuralLM(nn.Module):
    def __init__(self, kind, vocab_size, config, embedding=None, frozen=False):
        super().__init__()
        self.kind, self.config = kind, config
        d, h, layers = config["embedding_dim"], config["hidden_dim"], config["layers"]
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=PAD)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()
            if embedding is not None:
                table = torch.as_tensor(embedding, dtype=torch.float32)
                if table.shape != self.embedding.weight.shape or not torch.isfinite(table).all():
                    raise ValueError("Invalid input embedding table")
                self.embedding.weight.copy_(table)
        if frozen and embedding is None:
            raise ValueError("Frozen mode requires an explicit embedding table")
        self.embedding.weight.requires_grad_(not frozen)
        self.dropout = nn.Dropout(config["dropout"])
        if kind in {"rnn", "lstm"}:
            layer = nn.RNN if kind == "rnn" else nn.LSTM
            self.core = layer(d, h, layers, batch_first=True,
                              dropout=config["dropout"] if layers > 1 else 0)
        elif kind == "transformer":
            self.projection = nn.Linear(d, h)
            self.position = nn.Embedding(config["context_length"], h)
            layer = nn.TransformerEncoderLayer(h, config["heads"], config["ff_dim"],
                                               config["dropout"], batch_first=True, norm_first=True)
            self.core = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(h), enable_nested_tensor=False)
            # TransformerEncoder clones layers; initialize each clone independently.
            for block in self.core.layers:
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
                nn.init.xavier_uniform_(block.linear1.weight)
                nn.init.xavier_uniform_(block.linear2.weight)
        else:
            raise ValueError(f"Unknown neural architecture: {kind}")
        self.output = nn.Linear(h, vocab_size)  # Deliberately no embedding weight tying.

    def states(self, ids):
        if ids.size(1) > self.config["context_length"]:
            raise ValueError("Sequence exceeds configured context length")
        x = self.dropout(self.embedding(ids))
        if self.kind == "transformer":
            positions = torch.arange(ids.size(1), device=ids.device)
            x = self.projection(x) + self.position(positions)[None]
            mask = torch.triu(torch.ones(ids.size(1), ids.size(1), dtype=torch.bool, device=ids.device), diagonal=1)
            return self.core(x, mask=mask, src_key_padding_mask=ids.eq(PAD))
        return self.core(x)[0]

    def forward(self, ids):
        return self.output(self.dropout(self.states(ids)))


def pool_states(states, ids, mode="mean"):
    mask = ids.ne(PAD) & ids.ne(BOS) & ids.ne(EOS)
    if not mask.any(dim=1).all():
        raise ValueError("Cannot pool a sequence without content tokens")
    if mode == "mean":
        return (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    if mode == "last":
        positions = torch.arange(ids.size(1), device=ids.device).expand_as(ids)
        index = positions.masked_fill(~mask, -1).max(1).values
        return states[torch.arange(ids.size(0), device=ids.device), index]
    raise ValueError("Pooling must be mean or last")


class NGram:
    def __init__(self, vocab_size, order=3, alpha=0.1, interpolation=0.8):
        if order < 1 or alpha <= 0 or not 0 <= interpolation <= 1:
            raise ValueError("Invalid n-gram smoothing parameters")
        self.vocab_size, self.order = vocab_size, order
        self.alpha, self.interpolation = alpha, interpolation
        self.counts = [defaultdict(Counter) for _ in range(order)]
        self.totals = [Counter() for _ in range(order)]

    def fit(self, blocks):
        for xs, ys in blocks:
            xs, ys = list(map(int, xs)), list(map(int, ys))
            for i, target in enumerate(ys):
                if target == PAD:
                    continue
                context = xs[max(0, i + 2 - self.order):i + 1]
                for n in range(min(len(context), self.order - 1) + 1):
                    key = tuple(context[-n:]) if n else ()
                    self.counts[n][key][target] += 1
                    self.totals[n][key] += 1

    def prob(self, context, target):
        p = (self.counts[0].get((), {}).get(target, 0) + self.alpha) / (self.totals[0][()] + self.alpha * self.vocab_size)
        for n in range(1, min(len(context), self.order - 1) + 1):
            key = tuple(context[-n:])
            total = self.totals[n].get(key, 0)
            if total:
                local = (self.counts[n][key].get(target, 0) + self.alpha) / (total + self.alpha * self.vocab_size)
                p = self.interpolation * local + (1 - self.interpolation) * p
        return p

    def distribution(self, context):
        out = np.full(self.vocab_size, self.alpha, dtype=np.float64)
        for token, count in self.counts[0].get((), {}).items():
            out[token] += count
        out /= self.totals[0][()] + self.alpha * self.vocab_size
        for n in range(1, min(len(context), self.order - 1) + 1):
            key = tuple(context[-n:])
            total = self.totals[n].get(key, 0)
            if total:
                local = np.full(self.vocab_size, self.alpha, dtype=np.float64)
                for token, count in self.counts[n][key].items():
                    local[token] += count
                local /= total + self.alpha * self.vocab_size
                out = self.interpolation * local + (1 - self.interpolation) * out
        return out

    def state(self):
        return {"vocab_size": self.vocab_size, "order": self.order, "alpha": self.alpha,
                "interpolation": self.interpolation,
                "counts": [{k: dict(v) for k, v in level.items()} for level in self.counts],
                "totals": [dict(t) for t in self.totals]}

    @classmethod
    def from_state(cls, state):
        model = cls(state["vocab_size"], state["order"], state["alpha"], state["interpolation"])
        model.counts = [defaultdict(Counter, {k: Counter(v) for k, v in level.items()}) for level in state["counts"]]
        model.totals = [Counter(t) for t in state["totals"]]
        return model
