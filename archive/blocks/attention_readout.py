"""Attention pooling: a few learnable probes ask every galaxy a question.

WHY THIS BLOCK EXISTS
---------------------
Mean pooling throws away almost everything. It squashes thousands of galaxy
embeddings into their average, so a cloud with a few very dense knots and a
cloud that is uniformly mediocre can produce the same summary. That is one
plausible reason our mean-pooled GNN scores 0.6600 on CAMELS Omega_m while a
49-parameter linear fit on the two-point correlation function scores 0.8034:
the network's readout is a much blunter instrument than the hand-designed
statistic it is competing against.

Pooling by attention replaces the fixed average with k learnable queries called
seeds. Each seed is a question ("how much mass sits in tight groups?"), it scores
every point in the cloud, and the answer is the score-weighted average of that
cloud's point vectors. Because the k seeds are learned by gradient descent, the
readout can specialise: one seed can lock onto the densest neighbourhoods while
another averages the voids. The output is k vectors instead of one, so the head
that follows sees k views of the cloud rather than a single blurred one.

This is Pooling by Multihead Attention (PMA) from the Set Transformer,
arXiv:1810.00825, with the count-blindness switch and segment-wise batching
added. The framing of readout choice as the thing that decides how much of a
set survives to the predictor follows the adaptive readouts work,
arXiv:2211.04952.

DOES THE OUTPUT DEPEND ON THE NUMBER OF POINTS N
------------------------------------------------
BY DEFAULT, NO. The attention weights inside one cloud are a softmax, so they
sum to exactly one no matter how many points that cloud has. The output is
therefore a weighted AVERAGE, in the same count-blind family as mean pooling,
and it does NOT hand the model the galaxy count. Duplicating a cloud halves
every weight and leaves the summary unchanged, exactly, in real arithmetic.

Measured 2026-08-24, 32 clouds of 2000 points, feature dim 64, MPS: listing
every point twice moved the output by 7.2e-07 to 9.5e-07 in absolute terms
across four runs (three seeds, plus a repeat of seed 0), which is the same
figure relative to one output standard deviation. That is the float32 rounding
floor for a sum over 4000 terms, not a real dependence, and it is not bitwise
repeatable because index_add_ on MPS accumulates in a nondeterministic order.
Controls in the same run: mean pooling drifted 2.6e-08 to 3.7e-08, and sum
pooling drifted by exactly 1.0000 relative, that is to say it doubled, as it
must.

That default matters here because the galaxy count per cloud is a resolution
artifact that leaks Omega_m in CAMELS (correlation 0.73, since a halo is only
recorded above about 20 particles and the particle mass depends on Omega_m).
Measured 2026-08-18 to 2026-08-20: GNN sum pooling 0.8020 against mean pooling
0.6600 on CAMELS, but 0.5170 against 0.5196 on CAMELS-SAM where every cloud has
exactly 5000 galaxies. Sum pooling is therefore reading the leak, not the
physics.

The count dependence is a constructor flag, not a buried detail. Set
`count_aware=True` and one extra column, log(N / count_reference), is appended
to the summary. That is the honest way to run the leaky variant: the count
arrives as a single named feature that can be ablated by setting one flag,
rather than smeared through every channel the way sum pooling does it.

BATCHING WITHOUT PADDING
------------------------
Clouds hold 588 to 5000 points, so a dense (n_clouds, max_points, dim) tensor
would be mostly padding and would cost about 5000 / 2500 = 2x the memory for
CAMELS and far worse for a mixed batch. Instead this block takes the same flat
layout the rest of the repo uses: all points concatenated, plus an index saying
which cloud each point belongs to. The softmax is computed segment-wise over
that index with index_reduce_ (per-cloud maximum, for numerical stability) and
index_add_ (per-cloud sum). Peak extra memory is one (n_points, heads, seeds)
weight tensor plus one (n_points, dim) buffer per seed, never n_points times k
times dim at once.

MPS AND WHAT IT COSTS
---------------------
Every operation used here runs on MPS with torch 2.12.1: einsum, index_add_,
index_reduce_ with the amax reduction, LayerNorm and softmax arithmetic. Verified
2026-08-24 on an M5 Pro. index_reduce_ is flagged beta by torch and emits a
UserWarning on first use; it is correct, and the CPU and CUDA paths use the same
code, so no fallback branch is needed.

The readout on its own is roughly 13x the cost of mean pooling: 10.7 +/- 0.2 ms
against 0.8 +/- 0.0 ms per forward plus backward on 64000 points at dim 64
(n = 10). That ratio is misleading on its own, because the readout runs once per
batch while message passing runs once per layer per edge. Inside the actual
MessagePassingNet on real CAMELS graphs (32 clouds, 86583 galaxies, 415902
edges at cutoff 0.015 x box, hidden 64, 3 layers), one training step went from
101.6 ms to 115.5 ms, a factor of 1.14, and the parameter count from 66,562 to
108,482 (medians of 20 interleaved steps after 5 warmups, n = 20, spread 1.7 and
1.9 ms). Projected over 200 epochs and 600 clouds that is about 6.4 minutes
against 7.3 minutes. An earlier version of that same measurement reported 0.87x,
which was an artifact of timing the two models in separate loops with a single
warmup step.
"""

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn

from common.metrics import resolve_device, seed_all


def segment_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over the points of each cloud, one column at a time.

    logits is (n_points, n_columns) and index says which of the n clouds each
    point belongs to. Every column of the result sums to one within each cloud.

    The per-cloud maximum is subtracted before the exponential. Without it a
    logit of 90 overflows float32, and attention logits grow with the feature
    dimension, so this is a real failure mode rather than a formality. Clouds
    with no points get a zero weight column instead of a NaN.
    """
    largest = torch.full((n, logits.shape[1]), -torch.inf,
                         device=logits.device, dtype=logits.dtype)
    largest = largest.index_reduce_(0, index, logits, "amax",
                                    include_self=True).clamp_min(-1e30)
    weights = torch.exp(logits - largest[index])
    total = torch.zeros_like(largest).index_add_(0, index, weights)
    return weights / total[index].clamp_min(torch.finfo(logits.dtype).tiny)


class AttentionReadout(nn.Module):
    """k learnable seed vectors attend over each cloud's points.

    Drop-in replacement for `point_clouds.pointnet.pool`: the call signature is
    (values, index, n) and the result is (n, out_dim), except that out_dim is
    seeds * dim rather than dim, so whatever consumes it needs widening.

    Arguments
        dim              width of the incoming point embeddings.
        seeds            k, the number of learnable queries. Default 4.
        heads            attention heads. dim must divide evenly by it.
        value_mlp        the rFF(Z) of PMA, applied to points before attending.
        count_aware      append log(N / count_reference) as one extra output
                         column. False keeps the block count-blind. See the
                         module docstring for why this is a switch.
        count_reference  the N that makes the count column zero.
    """

    def __init__(self, dim: int, seeds: int = 4, heads: int = 4,
                 value_mlp: bool = True, count_aware: bool = False,
                 count_reference: float = 1000.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must divide evenly by heads {heads}")
        if seeds < 1 or heads < 1:
            raise ValueError(f"seeds and heads must be at least 1, got {seeds}, {heads}")
        if count_reference <= 0:
            raise ValueError(f"count_reference must be positive, got {count_reference}")

        self.dim, self.seeds, self.heads = dim, seeds, heads
        self.head_dim = dim // heads
        self.count_aware, self.count_reference = count_aware, count_reference

        self.seed_vectors = nn.Parameter(torch.empty(seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)

        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.to_output = nn.Linear(dim, dim)
        self.value_mlp = (nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
                          if value_mlp else nn.Identity())
        self.norm_attention = nn.LayerNorm(dim)
        self.norm_output = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(),
                                         nn.Linear(dim, dim))

    @property
    def out_dim(self) -> int:
        """Width of the vector this block returns per cloud."""
        return self.seeds * self.dim + (1 if self.count_aware else 0)

    @property
    def depends_on_count(self) -> bool:
        """Whether the output can change when a cloud is duplicated."""
        return self.count_aware

    def _attend(self, values: torch.Tensor, index: torch.Tensor,
                n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the (n, seeds, dim) summary and the (n_points, heads, seeds) weights."""
        if values.dim() != 2 or values.shape[1] != self.dim:
            raise ValueError(f"values must be (n_points, {self.dim}), got {tuple(values.shape)}")
        if index.dim() != 1 or len(index) != len(values):
            raise ValueError(f"index must be one long per point, got {tuple(index.shape)} "
                             f"for {len(values)} points")

        points = self.value_mlp(values)
        n_points = len(points)
        keys = self.to_key(points).view(n_points, self.heads, self.head_dim)
        payload = self.to_value(points).view(n_points, self.heads, self.head_dim)
        query = self.to_query(self.seed_vectors).view(self.seeds, self.heads, self.head_dim)

        logits = torch.einsum("phd,shd->phs", keys, query) * self.head_dim ** -0.5
        weights = segment_softmax(logits.reshape(n_points, self.heads * self.seeds),
                                  index, n).view(n_points, self.heads, self.seeds)

        # One seed at a time, so the largest live tensor is (n_points, dim) rather
        # than (n_points, seeds, dim). At 5000 points times 600 clouds that is the
        # difference between fitting in unified memory and not.
        gathered = [torch.zeros(n, self.dim, device=values.device, dtype=values.dtype
                                ).index_add_(0, index,
                                             (payload * weights[:, :, s:s + 1]
                                              ).reshape(n_points, self.dim))
                    for s in range(self.seeds)]
        attended = self.to_output(torch.stack(gathered, dim=1))

        hidden = self.norm_attention(self.seed_vectors.unsqueeze(0) + attended)
        return self.norm_output(hidden + self.feedforward(hidden)), weights

    def forward(self, values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
        summary, _ = self._attend(values, index, n)
        flat = summary.reshape(n, self.seeds * self.dim)
        if not self.count_aware:
            return flat
        counts = torch.zeros(n, 1, device=values.device, dtype=values.dtype).index_add_(
            0, index, torch.ones(len(index), 1, device=values.device, dtype=values.dtype))
        return torch.cat([flat, torch.log(counts.clamp_min(1.0) / self.count_reference)], dim=1)

    @torch.no_grad()
    def attention_weights(self, values: torch.Tensor, index: torch.Tensor,
                          n: int) -> torch.Tensor:
        """(n_points, heads, seeds) weights, for asking which galaxies a seed reads."""
        was_training = self.training
        self.eval()
        weights = self._attend(values, index, n)[1]
        self.train(was_training)
        return weights


def _duplicate(values: torch.Tensor, index: torch.Tensor,
               n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Every cloud with each of its points listed twice, so N doubles and shape does not."""
    order = torch.argsort(torch.cat([index, index]), stable=True)
    return torch.cat([values, values])[order], torch.cat([index, index])[order]


def _mean_pool(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    total = torch.zeros(n, values.shape[1], device=values.device).index_add_(0, index, values)
    counts = torch.zeros(n, 1, device=values.device).index_add_(
        0, index, torch.ones(len(index), 1, device=values.device))
    return total / counts.clamp_min(1)


def _sum_pool(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, values.shape[1], device=values.device).index_add_(0, index, values)


def _synchronise(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _time_forward_backward(step, repeats: int,
                           device: torch.device) -> Tuple[float, float]:
    """Mean and standard deviation of seconds per forward plus backward, after a warmup."""
    step().sum().backward()
    _synchronise(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        step().sum().backward()
        _synchronise(device)
        times.append(time.perf_counter() - start)
    measured = torch.tensor(times)
    return float(measured.mean()), float(measured.std())


def smoke_test(clouds: int, points: int, dim: int, seeds: int, heads: int,
               device: torch.device, repeats: int = 5) -> None:
    """Shape, finiteness, the count-blindness claim, the guards, and the cost."""
    values = torch.randn(clouds * points, dim, device=device)
    index = torch.arange(clouds, device=device).repeat_interleave(points)
    block = AttentionReadout(dim=dim, seeds=seeds, heads=heads).to(device)

    out = block(values, index, clouds)
    print(f"device               {out.device}")
    print(f"input                {tuple(values.shape)} points, {clouds} clouds "
          f"of {points}")
    print(f"output               {tuple(out.shape)}  expected ({clouds}, {block.out_dim})")
    print(f"all finite           {bool(torch.isfinite(out).all())}")
    print(f"output mean/std      {out.mean().item():+.4f} / {out.std().item():.4f}")
    print(f"output min/max       {out.min().item():+.4f} / {out.max().item():+.4f}")
    print(f"parameters           {sum(p.numel() for p in block.parameters()):,}")

    weights = block.attention_weights(values, index, clouds)
    per_cloud = torch.zeros(clouds, heads * seeds, device=device).index_add_(
        0, index, weights.reshape(len(values), heads * seeds))
    print(f"weights sum to one   max deviation {float((per_cloud - 1).abs().max()):.3e}")

    doubled_values, doubled_index = _duplicate(values, index, clouds)
    with torch.no_grad():
        block.eval()
        drift = (block(doubled_values, doubled_index, clouds) - out).abs().max()
        mean_drift = (_mean_pool(doubled_values, doubled_index, clouds)
                      - _mean_pool(values, index, clouds)).abs().max()
        sum_before = _sum_pool(values, index, clouds)
        sum_drift = (_sum_pool(doubled_values, doubled_index, clouds)
                     - sum_before).abs().max() / sum_before.abs().max()
        block.train()
    print(f"duplicate 2N drift   {float(drift):.3e} absolute, "
          f"{float(drift) / out.std().item():.3e} of one output std")
    print(f"  mean pool control  {float(mean_drift):.3e} absolute")
    print(f"  sum pool control   {float(sum_drift):.4f} relative (must be 1.0000)")

    aware = AttentionReadout(dim=dim, seeds=seeds, heads=heads, count_aware=True).to(device)
    with torch.no_grad():
        aware.eval()
        aware_drift = (aware(doubled_values, doubled_index, clouds)
                       - aware(values, index, clouds))[:, -1].abs().max()
        aware.train()
    print(f"count_aware=True     out_dim {aware.out_dim}, depends_on_count "
          f"{aware.depends_on_count}, count column moves {float(aware_drift):.4f} "
          f"(log 2 = {float(torch.log(torch.tensor(2.0))):.4f})")

    for name, call in (
            ("dim not divisible by heads", lambda: AttentionReadout(dim=64, heads=5)),
            ("wrong feature width", lambda: block(values[:, :dim - 1], index, clouds)),
            ("index length mismatch", lambda: block(values, index[:-1], clouds))):
        try:
            call()
            print(f"guard NOT tripped    {name}")
        except ValueError as problem:
            print(f"guard tripped        {name}: {problem}")

    empty_values = values[index != 0]
    empty_index = index[index != 0]
    with torch.no_grad():
        empty_out = block(empty_values, empty_index, clouds)
    expected_norm = (seeds * dim) ** 0.5
    print(f"empty cloud          finite {bool(torch.isfinite(empty_out).all())}, "
          f"row 0 norm {float(empty_out[0].norm()):.4f}, expected "
          f"{expected_norm:.4f} (seeds only, after LayerNorm)")

    # Mean pooling holds no parameters, so the baseline needs a leaf that carries
    # gradient or its backward pass has nothing to do and the comparison is empty.
    live = values.detach().clone().requires_grad_(True)
    attention_cost, attention_spread = _time_forward_backward(
        lambda: block(live, index, clouds), repeats, device)
    mean_cost, mean_spread = _time_forward_backward(
        lambda: _mean_pool(live, index, clouds), repeats, device)
    print(f"forward+backward     attention {attention_cost * 1e3:.1f} +/- "
          f"{attention_spread * 1e3:.1f} ms, mean pool {mean_cost * 1e3:.1f} +/- "
          f"{mean_spread * 1e3:.1f} ms, ratio {attention_cost / mean_cost:.1f}x "
          f"(n = {repeats})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clouds", type=int, default=32, help="clouds in the batch")
    parser.add_argument("--points", type=int, default=2000, help="points per cloud")
    parser.add_argument("--dim", type=int, default=64, help="feature width per point")
    parser.add_argument("--seeds", type=int, default=4, help="learnable queries, k")
    parser.add_argument("--heads", type=int, default=4, help="attention heads")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "mps", "cuda", "cpu"), help="compute device")
    args = parser.parse_args()

    if min(args.clouds, args.points, args.dim, args.seeds, args.heads, args.repeats) < 1:
        parser.error("clouds, points, dim, seeds, heads and repeats must all be at least 1")

    seed_all(args.seed)
    device = resolve_device(args.device)
    print(f"torch {torch.__version__}, seed {args.seed}")
    smoke_test(args.clouds, args.points, args.dim, args.seeds, args.heads,
               device, args.repeats)


if __name__ == "__main__":
    main()
