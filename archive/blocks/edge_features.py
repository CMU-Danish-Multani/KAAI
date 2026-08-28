"""Richer E(3)-invariant edge features for the radius graph.

WHY THIS BLOCK EXISTS
---------------------
Our message passing network hands each edge exactly one number, the separation
d_ij / Rc. That is a very thin pipe. A single scalar entering a linear layer can
only produce a straight line in d, so the first layer cannot express "pairs at
0.3 Rc matter and pairs at 0.7 Rc do not" without spending depth on it, and the
angular arrangement of a galaxy's neighbours is not passed at all. Meanwhile the
statistic we are losing to, the two-point correlation function, is itself a
histogram of d over many bins. We are feeding the network less resolution in d
than the baseline gets, and then asking why the baseline wins.

Two fixes, both standard in geometric deep learning and both cheap.

RADIAL BASIS. Expand d into a set of smooth bumps instead of passing it raw.
Gaussian bumps are the SchNet basis (arXiv:1706.08566). Spherical Bessel
functions are the DimeNet basis (arXiv:2003.03123); they are the radial
solutions of the Laplace equation inside a ball, so they are the natural
orthogonal basis for a spherical cutoff rather than an arbitrary choice. With 16
bumps the first layer can draw any smooth curve in d in one step, which is
exactly the freedom a binned correlation function has and our current edge
feature does not.

SMOOTH CUTOFF. A neighbour that drifts across Rc currently appears or vanishes
discontinuously, so the model is a discontinuous function of the positions. Both
bases are multiplied by an envelope that reaches zero at Rc with zero first and
second derivative (the DimeNet polynomial, arXiv:2003.03123) or by the raised
cosine (SchNet and Behler-Parrinello). The feature then fades in instead of
switching on.

THREE-BODY ANGLES. This is the part that can beat the two-point function rather
than merely match it. For a galaxy i, take every pair of neighbours j and k and
measure the angle between them. Filaments, sheets and voids differ in exactly
this quantity: three galaxies strung out along a filament sit at cos = -1 to
each other seen from the middle one, three in an isotropic blob average to
cos^2 = 1/3. The two-point correlation function is blind to all of it by
construction, because it only ever counts pairs. This is the information the
three-point function and the bispectrum carry, and it is the standard reason
directional message passing (arXiv:2003.03123) beats distance-only message
passing on molecules.

The naive way to compute it costs sum_i degree_i^2. We use the moment tensor
contraction instead, which costs O(n_edges). For unit vectors u pointing from
node i to each of its neighbours, with weights w,

    sum_{j,k} w_j w_k (u_j . u_k)^p  ==  || sum_j w_j u_j^{tensor p} ||^2

because (u_j . u_k)^p is the inner product of the two p-fold tensor powers. So
one scatter-add of a 3^p vector per edge gives every pair angle in the
neighbourhood at once. This identity is the basis of the SOAP power spectrum
(arXiv:1209.3140) and of Moment Tensor Potentials (arXiv:1512.06054). The smoke
test checks the fast path against brute force enumeration.

WHY EVERYTHING HERE IS INVARIANT
--------------------------------
The box is periodic and has no meaningful origin or orientation, so a universe
that is shifted, rotated or mirrored is the same universe. Every feature here is
built from either d_ij or a dot product u_ij . u_ik, and:

  translation  d and the dot products use minimum-image DIFFERENCES of positions,
               which do not move when everything moves.
  rotation     lengths and dot products are unchanged by any rotation.
  reflection   dot products are unchanged by improper rotations too, so this is
               O(3) invariance, not merely SO(3). Nothing here uses a triple
               product, which is the only thing that would tell a left-handed
               arrangement from a right-handed one.

The direction vectors passed in are equivariant, not invariant, and they are
never exposed: they only ever leave this module contracted against each other.
That is a deliberate deviation from CosmoBench (arXiv:2507.03707) Sec. 4.1,
which feeds absolute positions, and it follows the cosmic graphs construction of
arXiv:2204.13713.

DOES THE OUTPUT DEPEND ON THE NUMBER OF POINTS N?
-------------------------------------------------
SWITCHABLE. Count-blind by default. Per block:

    distance d / Rc                     NO
    gaussian basis                      NO
    bessel basis                        NO
    envelope channel                    NO
    angular, normalise_angular=True     NO      (default)
    angular, normalise_angular=False    YES     grows as degree^2
    local density, use_local_density    YES     degree / volume, proportional
                                                to number density, so directly
                                                proportional to N at fixed box

Read `n_dependence()` for the same table computed from the live configuration,
and `depends_on_point_count()` for the single summary bit.

This matters more here than it looks. In CAMELS the galaxy count per cloud
correlates 0.73 with Omega_m, because a halo is only recorded once it reaches
about 20 particles and the particle mass depends on Omega_m. Measured on this
repo: GNN sum pooling 0.8020 on CAMELS Omega_m against 0.6600 for mean pooling,
and 0.5170 against 0.5196 on CAMELS-SAM where the count is fixed at 5000. Local
density is that same leak spread over nodes: mean degree inside a fixed radius
is number density times the volume of the cutoff sphere, so switching
use_local_density on gives the network the galaxy count in a per-node disguise.
It defaults to OFF for that reason, and turning it on is an experiment, not a
tuning knob.

The angular features are normalised to a MEAN over distinct neighbour pairs
rather than a sum precisely so that they carry shape without carrying count. A
sum over pairs would grow as degree^2 and would leak harder than sum pooling
does. The mean over ordered distinct pairs is the sum with the self pairs
removed and divided by W^2 - Q, which is the exact number of ordered distinct
pairs when the weights are all one.

MPS SUPPORT
-----------
Everything here is elementwise arithmetic, reshape, and index_add_, all of which
run natively on MPS in torch 2.12.1 and backpropagate correctly (checked against
CPU by the smoke test). No fallback path is needed. Two MPS specifics are
handled rather than assumed: float64 does not exist on MPS, so float64 input is
rejected with a message instead of failing inside a kernel, and lengths are
taken as sqrt(sum of squares) rather than through torch.linalg, which keeps the
op set to kernels this repo already relies on.

MEMORY
------
The p = 3 moment tensor is 27 numbers per edge. At the smoke test size that is
1.3 million edges times 27 floats, about 142 MB in one transient buffer, and the
p = 4 version would be 425 MB. Edges are therefore processed in chunks of
chunk_size and scattered as they go, so peak memory is set by chunk_size and not
by the size of the batch.
"""

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ENVELOPES: Tuple[str, ...] = ("polynomial", "cosine", "none")
DENSITY_TRANSFORMS: Tuple[str, ...] = ("log1p", "raw")
NODE_FEATURE_PLACEMENTS: Tuple[str, ...] = ("both", "source", "target", "none")

# sin(n pi d) / d is finite as d goes to zero but the quotient is not, so the
# denominator is floored. Two galaxies at the same position would be a bug in
# the catalogue rather than a case to support.
_LENGTH_FLOOR = 1e-6

# W^2 - Q is exactly zero for a node with one neighbour, where no angle exists.
_PAIR_WEIGHT_FLOOR = 1e-12


def edge_vectors(positions: np.ndarray, edges: np.ndarray,
                 box: Optional[float]) -> np.ndarray:
    """Minimum-image displacement for each edge, as (n_edges, 3).

    Mirrors point_clouds.gnn.edge_lengths, which returns the norm of exactly
    this quantity, so the two stay consistent. The vector points from the
    RECEIVING node edges[1] towards the SENDING node edges[0], which makes
    edges[1] the centre of the neighbourhood whose angles are measured.

    box=None skips the wrap, which is what a free-space test cloud needs. Real
    catalogues are periodic and must pass the box side.
    """
    delta = positions[edges[0]] - positions[edges[1]]
    if box is not None:
        delta = delta - box * np.round(delta / box)
    return delta.astype(np.float32)


def polynomial_envelope(scaled: torch.Tensor, exponent: int = 6) -> torch.Tensor:
    """DimeNet envelope: 1 at d = 0, and 0 at d = Rc with zero slope and curvature.

    u(x) = 1 + a x^p + b x^(p+1) + c x^(p+2) with the coefficients of
    arXiv:2003.03123 eq. (8). Vanishing first and second derivatives at the
    cutoff are the point: a neighbour crossing Rc changes the features smoothly,
    so the model stays a continuous function of the galaxy positions.
    """
    p = float(exponent)
    a = -(p + 1.0) * (p + 2.0) / 2.0
    b = p * (p + 2.0)
    c = -p * (p + 1.0) / 2.0
    x = scaled.clamp(0.0, 1.0)
    x_p = x ** p
    return 1.0 + a * x_p + b * x_p * x + c * x_p * x * x


def cosine_envelope(scaled: torch.Tensor) -> torch.Tensor:
    """Raised cosine cutoff, 0.5 (cos(pi d / Rc) + 1). SchNet, arXiv:1706.08566.

    Reaches zero with zero slope but not zero curvature, so it is smoother than
    a hard cut and less smooth than the polynomial.
    """
    return 0.5 * (torch.cos(math.pi * scaled.clamp(0.0, 1.0)) + 1.0)


def smooth_cutoff(scaled: torch.Tensor, kind: str = "polynomial",
                  exponent: int = 6) -> torch.Tensor:
    """Dispatch over ENVELOPES. `none` returns ones, which is a hard cut."""
    if kind == "polynomial":
        return polynomial_envelope(scaled, exponent)
    if kind == "cosine":
        return cosine_envelope(scaled)
    if kind == "none":
        return torch.ones_like(scaled)
    raise ValueError(f"unknown envelope {kind!r}, expected from {ENVELOPES}")


def gaussian_basis(scaled: torch.Tensor, centres: torch.Tensor,
                   gamma: torch.Tensor) -> torch.Tensor:
    """exp(-gamma (d - mu)^2) for each centre. SchNet, arXiv:1706.08566 eq. (7).

    scaled is (n_edges, 1) holding d / Rc. Returns (n_edges, n_centres).
    """
    return torch.exp(-gamma * (scaled - centres) ** 2)


def bessel_basis(scaled: torch.Tensor, frequencies: torch.Tensor,
                 normalise: bool = True) -> torch.Tensor:
    """sqrt(2) sin(n pi d / Rc) / (d / Rc), the DimeNet radial basis.

    arXiv:2003.03123 eq. (7), written in units of Rc so the cutoff is 1. These
    are the zeroth order spherical Bessel functions with the boundary condition
    that they vanish at Rc.

    normalise divides channel n by n pi, which is that channel's value in the
    limit d goes to zero, so every channel enters the following linear layer at
    order 1 instead of channel 16 arriving 16 times larger than channel 1. It is
    a per-channel constant, so the linear layer that follows can undo it exactly
    and no expressivity is lost. It changes only the effective initialisation
    scale. This is a deviation from the reference implementation, recorded here
    rather than left silent.
    """
    safe = scaled.clamp_min(_LENGTH_FLOOR)
    raw = math.sqrt(2.0) * torch.sin(frequencies * math.pi * scaled) / safe
    return raw / (frequencies * math.pi) if normalise else raw


def node_degree(target: torch.Tensor, n_nodes: int, device: torch.device,
                dtype: torch.dtype) -> torch.Tensor:
    """Number of incoming edges per node, as (n_nodes, 1)."""
    ones = torch.ones(target.shape[0], 1, device=device, dtype=dtype)
    return torch.zeros(n_nodes, 1, device=device, dtype=dtype).index_add_(0, target, ones)


def local_density(degree: torch.Tensor, cutoff: float,
                  transform: str = "log1p") -> torch.Tensor:
    """Galaxies per unit volume around each node: degree / (4/3 pi Rc^3).

    DEPENDS ON N. This is a local estimate of the number density, and number
    density at fixed box size is the galaxy count divided by a constant. Every
    node carries a noisy copy of the quantity that leaks Omega_m in CAMELS.

    `log1p` is the default because the raw density is of order N in box units,
    about 2000 for a unit box holding 2000 galaxies, which is a terrible input
    scale for a linear layer. Taking log1p also matches how density is usually
    handled in cosmology, where the interesting range spans decades.
    """
    if transform not in DENSITY_TRANSFORMS:
        raise ValueError(f"unknown density transform {transform!r}, "
                         f"expected from {DENSITY_TRANSFORMS}")
    volume = 4.0 / 3.0 * math.pi * cutoff ** 3
    density = degree / volume
    return torch.log1p(density) if transform == "log1p" else density


def angular_invariants(unit: torch.Tensor, weight: torch.Tensor,
                       centre: torch.Tensor, n_nodes: int,
                       powers: Sequence[int] = (1, 2, 3),
                       normalise: bool = True, exclude_self_pairs: bool = True,
                       chunk_size: int = 262_144) -> torch.Tensor:
    """Weighted mean of (cos angle)^p over pairs of a node's incident edges.

    unit is (n_edges, 3) unit vectors pointing from the centre node to each of
    its neighbours, weight is (n_edges, 1) non-negative radial weights, and
    centre[e] says which node edge e belongs to. Returns (n_nodes, len(powers)).

    Computed by the moment tensor identity rather than by enumerating triplets:

        S_p = sum_j w_j u_j^{tensor p}          one scatter-add per edge
        ||S_p||^2 = sum_{j,k} w_j w_k (u_j . u_k)^p

    with W = sum_j w_j and Q = sum_j w_j^2. Removing the self pairs j = k
    subtracts Q, because (u_j . u_j)^p = 1 for unit vectors. Dividing by W^2 - Q
    turns the sum over ordered distinct pairs into their weighted mean, which is
    what makes the output independent of how many neighbours the node has.

    normalise=False returns the raw sum instead, which grows as degree^2 and
    therefore leaks the galaxy count. See the module docstring.

    A node with fewer than two neighbours has no angle, and gets exactly zero
    rather than a division by a vanishing denominator.
    """
    if unit.dim() != 2 or unit.shape[1] != 3:
        raise ValueError(f"unit must be (n_edges, 3), got {tuple(unit.shape)}")
    if len(powers) == 0:
        raise ValueError("at least one angular power is required")
    if any(int(p) < 1 for p in powers):
        raise ValueError(f"angular powers must be 1 or greater, got {tuple(powers)}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    device, dtype = unit.device, unit.dtype
    wanted = sorted({int(p) for p in powers})
    moments = {p: torch.zeros(n_nodes, 3 ** p, device=device, dtype=dtype)
               for p in wanted}
    total_weight = torch.zeros(n_nodes, 1, device=device, dtype=dtype)
    square_weight = torch.zeros(n_nodes, 1, device=device, dtype=dtype)

    for start in range(0, unit.shape[0], chunk_size):
        piece = slice(start, start + chunk_size)
        u, w, c = unit[piece], weight[piece], centre[piece]
        power = torch.ones(u.shape[0], 1, device=device, dtype=dtype)
        for p in range(1, wanted[-1] + 1):
            power = (power.unsqueeze(2) * u.unsqueeze(1)).reshape(u.shape[0], -1)
            if p in moments:
                moments[p].index_add_(0, c, w * power)
        total_weight.index_add_(0, c, w)
        square_weight.index_add_(0, c, w * w)

    out = []
    for p in powers:
        numerator = (moments[int(p)] ** 2).sum(dim=1, keepdim=True)
        if exclude_self_pairs:
            numerator = numerator - square_weight
        if not normalise:
            out.append(numerator)
            continue
        denominator = total_weight ** 2
        if exclude_self_pairs:
            denominator = denominator - square_weight
        usable = denominator > _PAIR_WEIGHT_FLOOR
        ratio = numerator / denominator.clamp_min(_PAIR_WEIGHT_FLOOR)
        out.append(torch.where(usable, ratio, torch.zeros_like(ratio)))
    return torch.cat(out, dim=1)


def angular_invariants_reference(unit: torch.Tensor, weight: torch.Tensor,
                                 centre: torch.Tensor, n_nodes: int,
                                 powers: Sequence[int] = (1, 2, 3),
                                 normalise: bool = True,
                                 exclude_self_pairs: bool = True) -> torch.Tensor:
    """Brute force enumeration of every neighbour pair. Reference only.

    Cost is sum_i degree_i^2, so this is for verifying angular_invariants on a
    small graph, not for training. It exists because the moment tensor identity
    is the one piece of maths in this file that is not obvious by inspection,
    and an identity nobody checked is an assumption.
    """
    out = torch.zeros(n_nodes, len(powers), dtype=unit.dtype)
    order = [[] for _ in range(n_nodes)]
    for e, node in enumerate(centre.tolist()):
        order[node].append(e)
    for node, members in enumerate(order):
        if not members:
            continue
        u = unit[members]
        w = weight[members].reshape(-1)
        cosine = u @ u.T
        pair_weight = w[:, None] * w[None, :]
        if exclude_self_pairs:
            keep = 1.0 - torch.eye(len(members), dtype=unit.dtype)
        else:
            keep = torch.ones(len(members), len(members), dtype=unit.dtype)
        mass = float((pair_weight * keep).sum())
        for column, p in enumerate(powers):
            total = float((pair_weight * keep * cosine ** int(p)).sum())
            out[node, column] = total / mass if normalise and mass > 0 else total
    return out


class EdgeFeatures(NamedTuple):
    """What the featuriser returns.

    edge  (n_edges, out_features)  ready to concatenate into a message MLP,
                                   with any node-level invariants already
                                   broadcast onto their incident edges.
    node  (n_nodes, node_features) the raw node-level invariants on their own,
                                   for use as initial node features instead of
                                   the constant the current network starts from.
                                   Empty with shape (n_nodes, 0) when no
                                   node-level block is enabled.
    """

    edge: torch.Tensor
    node: torch.Tensor


class EdgeFeaturiser(nn.Module):
    """Configurable E(3)-invariant edge features for a radius graph.

    Drop-in for the single `d / Rc` column that point_clouds.gnn.Batch carries
    today. Call it as

        features = featuriser(edge_vector, edges, n_nodes, cutoff)

    where edge_vector is (n_edges, 3) minimum-image displacements from
    edge_vectors(), edges is the (2, n_edges) long tensor from Batch, and cutoff
    is Rc in the SAME length units as edge_vector. Returns EdgeFeatures.

    out_dim=None returns the raw invariants, which is the right choice when the
    following layer is going to mix them anyway and the raw channels should stay
    readable. Setting out_dim adds one Linear plus SiLU, the standard learned
    radial embedding, so the block emits a fixed width whatever the basis size.

    use_local_density=True re-introduces the galaxy count leak. It is off by
    default. See the module docstring.
    """

    def __init__(self, n_basis: int = 16, use_distance: bool = True,
                 use_gaussian: bool = True, use_bessel: bool = True,
                 normalise_bessel: bool = True, envelope: str = "polynomial",
                 envelope_exponent: int = 6, apply_envelope_to_basis: bool = True,
                 use_local_density: bool = False, density_transform: str = "log1p",
                 angular_powers: Sequence[int] = (1, 2, 3),
                 normalise_angular: bool = True, exclude_self_pairs: bool = True,
                 node_features_on: str = "both", out_dim: Optional[int] = None,
                 chunk_size: int = 262_144):
        super().__init__()
        if envelope not in ENVELOPES:
            raise ValueError(f"unknown envelope {envelope!r}, expected from {ENVELOPES}")
        if density_transform not in DENSITY_TRANSFORMS:
            raise ValueError(f"unknown density transform {density_transform!r}, "
                             f"expected from {DENSITY_TRANSFORMS}")
        if node_features_on not in NODE_FEATURE_PLACEMENTS:
            raise ValueError(f"unknown node_features_on {node_features_on!r}, "
                             f"expected from {NODE_FEATURE_PLACEMENTS}")
        if (use_gaussian or use_bessel) and n_basis < 1:
            raise ValueError(f"n_basis must be positive, got {n_basis}")
        if any(int(p) < 1 for p in angular_powers):
            raise ValueError(f"angular powers must be 1 or greater, got {tuple(angular_powers)}")
        if out_dim is not None and out_dim < 1:
            raise ValueError(f"out_dim must be positive or None, got {out_dim}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.n_basis = n_basis
        self.use_distance = use_distance
        self.use_gaussian = use_gaussian
        self.use_bessel = use_bessel
        self.normalise_bessel = normalise_bessel
        self.envelope = envelope
        self.envelope_exponent = envelope_exponent
        self.apply_envelope_to_basis = apply_envelope_to_basis
        self.use_local_density = use_local_density
        self.density_transform = density_transform
        self.angular_powers = tuple(int(p) for p in angular_powers)
        self.normalise_angular = normalise_angular
        self.exclude_self_pairs = exclude_self_pairs
        self.node_features_on = node_features_on
        self.chunk_size = chunk_size

        # Fixed, not learned. Learned centres drift towards each other early in
        # training and leave holes in the coverage of d, and a fixed basis keeps
        # the feature meaning stable across seeds so runs stay comparable.
        centres = torch.linspace(0.0, 1.0, n_basis) if n_basis > 1 else torch.zeros(1)
        spacing = 1.0 / (n_basis - 1) if n_basis > 1 else 1.0
        self.register_buffer("centres", centres.reshape(1, -1))
        self.register_buffer("gamma", torch.tensor(0.5 / spacing ** 2))
        self.register_buffer("frequencies",
                             torch.arange(1, n_basis + 1, dtype=torch.float32).reshape(1, -1))

        self.edge_features = (int(use_distance) + int(use_gaussian) * n_basis
                              + int(use_bessel) * n_basis + int(envelope != "none"))
        self.node_features = int(use_local_density) + len(self.angular_powers)
        copies = {"both": 2, "source": 1, "target": 1, "none": 0}[node_features_on]
        self.raw_features = self.edge_features + copies * self.node_features
        if self.raw_features == 0:
            raise ValueError("every feature block is disabled, so there is nothing to return")

        self.out_features = out_dim if out_dim is not None else self.raw_features
        self.project = (nn.Sequential(nn.Linear(self.raw_features, out_dim), nn.SiLU())
                        if out_dim is not None else None)

    def depends_on_point_count(self) -> bool:
        """True when any enabled block is a function of N by construction."""
        return bool(self.use_local_density
                    or (self.angular_powers and not self.normalise_angular))

    def n_dependence(self) -> Dict[str, bool]:
        """Per-block N-dependence for the live configuration."""
        table: Dict[str, bool] = {}
        if self.use_distance:
            table["distance"] = False
        if self.use_gaussian:
            table["gaussian_basis"] = False
        if self.use_bessel:
            table["bessel_basis"] = False
        if self.envelope != "none":
            table["envelope"] = False
        if self.angular_powers:
            table["angular"] = not self.normalise_angular
        if self.use_local_density:
            table["local_density"] = True
        return table

    def feature_names(self) -> List[str]:
        """Name of every raw channel, in the order they are concatenated."""
        edge: List[str] = []
        if self.use_distance:
            edge.append("distance")
        if self.use_gaussian:
            edge += [f"gaussian_{i}" for i in range(self.n_basis)]
        if self.use_bessel:
            edge += [f"bessel_{i + 1}" for i in range(self.n_basis)]
        if self.envelope != "none":
            edge.append("envelope")
        node: List[str] = [f"angular_cos_power_{p}" for p in self.angular_powers]
        if self.use_local_density:
            node.append("local_density")
        placements = {"both": ("source", "target"), "source": ("source",),
                      "target": ("target",), "none": ()}[self.node_features_on]
        return edge + [f"{where}_{name}" for where in placements for name in node]

    def extra_repr(self) -> str:
        return (f"raw_features={self.raw_features}, out_features={self.out_features}, "
                f"n_basis={self.n_basis}, envelope={self.envelope!r}, "
                f"angular_powers={self.angular_powers}, "
                f"use_local_density={self.use_local_density}, "
                f"depends_on_N={self.depends_on_point_count()}")

    def forward(self, edge_vector: torch.Tensor, edges: torch.Tensor, n_nodes: int,
                cutoff: float = 1.0) -> EdgeFeatures:
        if edge_vector.dim() != 2 or edge_vector.shape[1] != 3:
            raise ValueError(f"edge_vector must be (n_edges, 3), "
                             f"got {tuple(edge_vector.shape)}")
        if edges.dim() != 2 or edges.shape[0] != 2:
            raise ValueError(f"edges must be (2, n_edges), got {tuple(edges.shape)}")
        if edges.shape[1] != edge_vector.shape[0]:
            raise ValueError(f"edges has {edges.shape[1]} columns but edge_vector has "
                             f"{edge_vector.shape[0]} rows")
        if cutoff <= 0.0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if edge_vector.dtype == torch.float64 and edge_vector.device.type == "mps":
            raise ValueError("MPS has no float64, so pass float32 edge vectors")

        device, dtype = edge_vector.device, edge_vector.dtype
        if edge_vector.shape[0] == 0:
            empty_edge = torch.zeros(0, self.out_features, device=device, dtype=dtype)
            return EdgeFeatures(empty_edge,
                                torch.zeros(n_nodes, self.node_features,
                                            device=device, dtype=dtype))

        source, target = edges[0], edges[1]
        length = edge_vector.pow(2).sum(dim=1, keepdim=True).sqrt()
        scaled = (length / cutoff).clamp(0.0, 1.0)
        envelope = smooth_cutoff(scaled, self.envelope, self.envelope_exponent)

        edge_parts: List[torch.Tensor] = []
        if self.use_distance:
            edge_parts.append(scaled)
        if self.use_gaussian:
            basis = gaussian_basis(scaled, self.centres, self.gamma)
            edge_parts.append(basis * envelope if self.apply_envelope_to_basis else basis)
        if self.use_bessel:
            basis = bessel_basis(scaled, self.frequencies, self.normalise_bessel)
            edge_parts.append(basis * envelope if self.apply_envelope_to_basis else basis)
        if self.envelope != "none":
            edge_parts.append(envelope)

        node_parts: List[torch.Tensor] = []
        if self.angular_powers:
            # The same envelope weights the angular sum, so a neighbour crossing
            # Rc fades out of the angle statistics instead of dropping out of
            # them. Without it the three-body term would be discontinuous even
            # though the two-body terms are not.
            unit = edge_vector / length.clamp_min(_LENGTH_FLOOR * cutoff)
            node_parts.append(angular_invariants(
                unit, envelope, target, n_nodes, self.angular_powers,
                self.normalise_angular, self.exclude_self_pairs, self.chunk_size))
        if self.use_local_density:
            degree = node_degree(target, n_nodes, device, dtype)
            node_parts.append(local_density(degree, cutoff, self.density_transform))

        node = (torch.cat(node_parts, dim=1) if node_parts
                else torch.zeros(n_nodes, 0, device=device, dtype=dtype))
        if self.node_features_on in ("both", "source"):
            edge_parts.append(node[source])
        if self.node_features_on in ("both", "target"):
            edge_parts.append(node[target])

        raw = torch.cat(edge_parts, dim=1)
        return EdgeFeatures(self.project(raw) if self.project is not None else raw, node)


def _uniform_clouds(n_clouds: int, points_per_cloud: int,
                    seed: int) -> List[np.ndarray]:
    """Poisson clouds in the unit box. Their angle statistics are known exactly."""
    rng = np.random.default_rng(seed)
    return [rng.random((points_per_cloud, 3)).astype(np.float32)
            for _ in range(n_clouds)]


def _assemble(clouds: List[np.ndarray], box: float, cutoff: float
              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """One batch of clouds as a single disconnected graph, the way GraphSet does it."""
    from point_clouds.gnn import radius_graph

    edges, vectors, offset = [], [], 0
    for positions in clouds:
        e = radius_graph(positions, box, cutoff)
        edges.append(e + offset)
        vectors.append(edge_vectors(positions, e, box))
        offset += len(positions)
    return np.concatenate(edges, axis=1), np.concatenate(vectors), offset


def _random_improper_rotation(seed: int) -> np.ndarray:
    """An orthogonal matrix with determinant -1, so it mirrors as well as rotates."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return (q * np.array([-1.0, 1.0, 1.0])).astype(np.float64)


def _predicted_log_density(n_points: int, box: float, cutoff: float) -> float:
    """Mean over nodes of log1p(degree / volume) for a Poisson cloud.

    The first term is log1p of the mean density. The second is the Jensen
    correction: log1p is concave and the degree is a Poisson count with a
    relative spread of 1 / sqrt(degree), so the mean of the transform sits
    BELOW the transform of the mean. Leaving it out predicts 7.601 where the
    correct answer is 7.577, which would look like a bug in the block.
    """
    volume = 4.0 / 3.0 * math.pi * cutoff ** 3
    density = (n_points - 1) / box ** 3
    variance = density * volume / volume ** 2
    return math.log1p(density) - variance / (2.0 * (1.0 + density) ** 2)


def _milliseconds(call, device: torch.device, repeats: int = 10) -> float:
    import time

    call()
    if device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    if device.type == "mps":
        torch.mps.synchronize()
    return 1e3 * (time.perf_counter() - start) / repeats


def _smoke_test(n_clouds: int = 32, points_per_cloud: int = 2000, out_dim: int = 64,
                cutoff_fraction: float = 0.135, seed: int = 0) -> None:
    """Shapes, finiteness, device, the analytic values, invariance, and every guard.

    Run from the repo root with: python -m point_clouds.blocks.edge_features

    PREDICTED BEFORE MEASURING, for 2000 Poisson points in a unit box with
    Rc = 0.135, so that a wrong implementation has somewhere to disagree:

        mean degree            (N - 1) 4/3 pi Rc^3 = 20.60
        mean cos^1 over pairs  0        neighbour directions are isotropic
        mean cos^2 over pairs  0.3333   E[cos^2] = 1/3 for isotropic pairs
        mean cos^3 over pairs  0        odd moments vanish by symmetry
        local density          7.577    log1p(1999) less the Jensen correction
        density at 2x points   8.282    and the angles must not move at all
    """
    from common.metrics import resolve_device, seed_all

    seed_all(seed)
    device = resolve_device("auto")
    box, cutoff = 1.0, cutoff_fraction
    clouds = _uniform_clouds(n_clouds, points_per_cloud, seed)
    edge_array, vector_array, n_nodes = _assemble(clouds, box, cutoff)
    edges = torch.as_tensor(edge_array).to(device)
    vectors = torch.as_tensor(vector_array).to(device)

    print(f"device                   {device}")
    print(f"input                    {n_clouds} clouds of {points_per_cloud} points, "
          f"{n_nodes} nodes, {edges.shape[1]} directed edges")
    print(f"mean degree              measured {edges.shape[1] / n_nodes:.2f}, "
          f"predicted 20.60")

    rich = EdgeFeaturiser(out_dim=out_dim).to(device)
    with torch.no_grad():
        features = rich(vectors, edges, n_nodes, cutoff)
    parameters = sum(p.numel() for p in rich.parameters())
    print(f"edge features            {tuple(features.edge.shape)}  "
          f"finite={bool(torch.isfinite(features.edge).all())}  "
          f"range=[{float(features.edge.min()):.4f}, {float(features.edge.max()):.4f}]")
    print(f"node features            {tuple(features.node.shape)}  "
          f"finite={bool(torch.isfinite(features.node).all())}")
    print(f"widths                   raw={rich.raw_features} out={rich.out_features} "
          f"parameters={parameters}")
    print(f"depends_on_N             {rich.depends_on_point_count()}  {rich.n_dependence()}")

    # Raw channels, with local density switched on so the leaky block is measured too.
    audit = EdgeFeaturiser(use_local_density=True, node_features_on="none").to(device)
    raw = audit(vectors, edges, n_nodes, cutoff)
    length = vectors.pow(2).sum(dim=1, keepdim=True).sqrt()
    unit = vectors / length
    weight = polynomial_envelope(length / cutoff)
    names = audit.feature_names()
    print(f"raw channels             {tuple(raw.edge.shape)} named {len(names)}  "
          f"first={names[0]} last={names[-1]}")
    print(f"depends_on_N with density {audit.depends_on_point_count()}  "
          f"{audit.n_dependence()}")
    expected = {"angular_cos_power_1": 0.0, "angular_cos_power_2": 1.0 / 3.0,
                "angular_cos_power_3": 0.0,
                "local_density": _predicted_log_density(points_per_cloud, box, cutoff)}
    node_names = [f"angular_cos_power_{p}" for p in audit.angular_powers] + ["local_density"]
    for column, name in enumerate(node_names):
        values = raw.node[:, column]
        error = float(values.std() / math.sqrt(values.shape[0]))
        print(f"  {name:<22} {float(values.mean()):+.4f} +/- {error:.4f}   "
              f"predicted {expected[name]:+.4f}")

    # A per-node mean is a ratio of two random sums. Pooling the numerators and
    # the denominators separately across every node is the same quantity without
    # that ratio bias, and it is the one that has to land on 1/3 if the moment
    # tensor contraction is right. Measured 2026-08-24 over 5 seeds of 8 clouds:
    # 0.33282, seed-to-seed standard deviation 0.00061, which is 1.9 standard
    # errors below 1/3. Unit vectors drawn isotropically by hand and pushed
    # through the same code give 0.33345. The offset is sampling noise in a
    # radius graph, not a bias in this block.
    totals = angular_invariants(unit, weight, edges[1], n_nodes,
                                audit.angular_powers, normalise=False)
    mass = node_degree(edges[1], n_nodes, device, torch.float32)
    total_weight = torch.zeros_like(mass).index_add_(0, edges[1], weight)
    square_weight = torch.zeros_like(mass).index_add_(0, edges[1], weight * weight)
    pairs = float((total_weight ** 2 - square_weight).sum())
    pooled = ", ".join(f"{float(totals[:, c].sum()) / pairs:+.5f}"
                       for c in range(len(audit.angular_powers)))
    print(f"  pair-weighted, unbiased [{pooled}]   predicted "
          f"[+0.00000, +0.33333, +0.00000]")

    # Doubling the density in the same box. Angles are a shape statistic and must
    # not move, local density must, and that difference is the whole flag.
    dense = _uniform_clouds(4, 2 * points_per_cloud, seed + 1)
    sparse = _uniform_clouds(4, points_per_cloud, seed + 1)
    for label, sample in (("N per cloud 2000", sparse), ("N per cloud 4000", dense)):
        predicted = _predicted_log_density(len(sample[0]), box, cutoff)
        e, v, n = _assemble(sample, box, cutoff)
        out = audit(torch.as_tensor(v).to(device), torch.as_tensor(e).to(device), n, cutoff)
        angles = ", ".join(
            f"{float(out.node[:, c].mean()):+.4f} +/- "
            f"{float(out.node[:, c].std()) / math.sqrt(n):.4f}"
            for c in range(len(audit.angular_powers)))
        print(f"{label:<24} angular [{angles}]  density "
              f"{float(out.node[:, -1].mean()):.4f} (predicted {predicted:.4f})")

    # E(3) invariance. The graph is held fixed and the POSITIONS are moved, so any
    # disagreement is the featuriser rather than a reshuffled edge list.
    # Confined to the middle of the box so that no pair is a neighbour through
    # the periodic wrap, which lets the moved copy use free-space displacements.
    # The differences are taken in float64 and cast once, so the number below is
    # float32 representation error and not cancellation in a large translation.
    interior = 0.25 + 0.5 * np.random.default_rng(seed + 2).random((400, 3))
    free_cutoff = 0.1
    from point_clouds.gnn import radius_graph
    free_edges = radius_graph(interior.astype(np.float32), box, free_cutoff)
    plain = edge_vectors(interior, free_edges, None)
    transform = _random_improper_rotation(seed + 3)
    moved = interior @ transform.T + np.array([7.5, -2.0, 13.25])
    turned = edge_vectors(moved, free_edges, None)
    invariant = EdgeFeaturiser(use_local_density=True).to(device)
    graph = torch.as_tensor(free_edges).to(device)
    before = invariant(torch.as_tensor(plain).to(device), graph,
                       len(interior), free_cutoff).edge
    after = invariant(torch.as_tensor(turned).to(device), graph,
                      len(interior), free_cutoff).edge
    print(f"E(3) invariance          det={np.linalg.det(transform):+.1f} (reflection), "
          f"max abs change {float((before - after).abs().max()):.3e}, "
          f"relative {float((before - after).abs().max() / before.abs().max()):.3e}")

    # The moment tensor identity against brute force enumeration of every pair.
    reference_edges = torch.as_tensor(free_edges)
    unit_vectors = torch.as_tensor(plain).float()
    unit_vectors = unit_vectors / unit_vectors.pow(2).sum(1, keepdim=True).sqrt()
    weights = polynomial_envelope(
        torch.as_tensor(plain).pow(2).sum(1, keepdim=True).sqrt() / free_cutoff)
    fast = angular_invariants(unit_vectors, weights, reference_edges[1], len(interior))
    slow = angular_invariants_reference(unit_vectors, weights, reference_edges[1], len(interior))
    print(f"angular fast vs brute    max abs difference {float((fast - slow).abs().max()):.3e} "
          f"over {len(interior)} nodes and {reference_edges.shape[1]} edges")
    for normalise, self_pairs in ((False, True), (True, False), (False, False)):
        gap = (angular_invariants(unit_vectors, weights, reference_edges[1], len(interior),
                                  normalise=normalise, exclude_self_pairs=self_pairs)
               - angular_invariants_reference(unit_vectors, weights, reference_edges[1],
                                              len(interior), normalise=normalise,
                                              exclude_self_pairs=self_pairs)).abs().max()
        print(f"  normalise={str(normalise):<5} self_pairs={str(self_pairs):<5} "
              f"max abs difference {float(gap):.3e}")

    # MPS against CPU, because a silently wrong kernel is the failure this misses.
    cpu = EdgeFeaturiser(out_dim=out_dim)
    cpu.load_state_dict(rich.state_dict())
    with torch.no_grad():
        cpu_out = cpu(vectors.cpu(), edges.cpu(), n_nodes, cutoff).edge
    print(f"mps vs cpu               max abs difference "
          f"{float((features.edge.cpu() - cpu_out).abs().max()):.3e}")

    # Gradients must reach the positions, since a later block may want to move them.
    leaf = vectors[:200_000].detach().clone().requires_grad_(True)
    rich(leaf, edges[:, :200_000], n_nodes, cutoff).edge.sum().backward()
    print(f"backward                 grad finite={bool(torch.isfinite(leaf.grad).all())}  "
          f"nonzero fraction={float((leaf.grad != 0).float().mean()):.4f}")

    # Cost against one mean-pooled message aggregation over the same edges.
    messages = torch.randn(edges.shape[1], out_dim, device=device)
    def mean_pool() -> torch.Tensor:
        gathered = torch.zeros(n_nodes, out_dim, device=device).index_add_(0, edges[1], messages)
        return gathered / node_degree(edges[1], n_nodes, device, torch.float32).clamp_min(1.0)
    with torch.no_grad():
        pool_ms = _milliseconds(mean_pool, device)
        rich_ms = _milliseconds(lambda: rich(vectors, edges, n_nodes, cutoff), device)
        two_body = EdgeFeaturiser(angular_powers=(), out_dim=out_dim).to(device)
        two_body_ms = _milliseconds(lambda: two_body(vectors, edges, n_nodes, cutoff), device)
    print(f"forward cost             mean pooling {pool_ms:.2f} ms, "
          f"two-body only {two_body_ms:.2f} ms ({two_body_ms / pool_ms:.1f}x), "
          f"full block {rich_ms:.2f} ms ({rich_ms / pool_ms:.1f}x)")

    with torch.no_grad():
        empty = rich(vectors[:0], edges[:, :0], n_nodes, cutoff)
    print(f"no edges                 edge {tuple(empty.edge.shape)} "
          f"node {tuple(empty.node.shape)}")

    for description, thunk in (
            ("unknown envelope", lambda: EdgeFeaturiser(envelope="gaussian")),
            ("unknown density transform", lambda: EdgeFeaturiser(density_transform="log10")),
            ("unknown node placement", lambda: EdgeFeaturiser(node_features_on="edges")),
            ("n_basis zero", lambda: EdgeFeaturiser(n_basis=0)),
            ("angular power zero", lambda: EdgeFeaturiser(angular_powers=(0, 1))),
            ("out_dim zero", lambda: EdgeFeaturiser(out_dim=0)),
            ("chunk_size zero", lambda: EdgeFeaturiser(chunk_size=0)),
            ("everything disabled", lambda: EdgeFeaturiser(
                use_distance=False, use_gaussian=False, use_bessel=False,
                envelope="none", angular_powers=(), use_local_density=False)),
            ("edge_vector not 3-D", lambda: rich(vectors[:, :2], edges, n_nodes, cutoff)),
            ("edges not (2, E)", lambda: rich(vectors, edges.T, n_nodes, cutoff)),
            ("edge count mismatch", lambda: rich(vectors[:10], edges, n_nodes, cutoff)),
            ("cutoff not positive", lambda: rich(vectors, edges, n_nodes, 0.0)),
            ("float64 on mps", lambda: rich(vectors.double(), edges, n_nodes, cutoff)),
            ("unit not 3-D", lambda: angular_invariants(
                unit_vectors[:, :2], weights, reference_edges[1], len(interior))),
            ("no angular powers", lambda: angular_invariants(
                unit_vectors, weights, reference_edges[1], len(interior), powers=()))):
        try:
            thunk()
            print(f"GUARD NOT TRIPPED        {description}")
        except (ValueError, RuntimeError, TypeError) as error:
            print(f"guard tripped            {description}: {str(error).splitlines()[0]}")


if __name__ == "__main__":
    _smoke_test()
