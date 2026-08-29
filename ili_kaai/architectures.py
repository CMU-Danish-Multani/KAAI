"""The zoo's architectures, expressed as LtU-ILI configurations.

These are the families Ho et al. (2024) implement and benchmark: normalising flows
(MAF, NSF), mixture density networks, and neural ratio estimators. Every entry is a
config the framework consumes, not a bespoke model, so a recommendation the skill
returns can be handed to a user as something they can actually run.

Matched compute is enforced by holding the training budget identical across entries
(same batch size, learning rate, epoch cap and early stopping rule). Wall clock is
recorded per run so compute can also be reported rather than only assumed equal.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Held identical across every entry. This is what "matched compute" means here.
TRAIN_ARGS: Dict[str, Any] = {
    "training_batch_size": 32,
    "learning_rate": 1e-3,
    "max_num_epochs": 300,
    "stop_after_epochs": 20,
    "validation_fraction": 0.2,
}

HIDDEN = 50          # same width for every entry that has a width


@dataclass(frozen=True)
class Architecture:
    key: str
    engine: str                       # NPE, NLE or NRE
    model: str                        # maf, nsf, mdn, made, mlp, resnet, linear, gf, cnf
    family: str                       # normalising_flow, mixture_density, ratio_estimator
    repeats: int                      # ensemble size
    backend: str = "sbi"              # sbi, or lampe which is NPE only
    model_args: Dict[str, Any] = field(default_factory=dict)
    # A heterogeneous ensemble. When set, each (model, args) pair contributes one
    # member and `model`/`repeats` are ignored. Ensembling clones was measured to
    # close under a fifth of the calibration gap, so mixing families is the next
    # thing to try rather than more clones.
    mixture: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    summary: str = ""
    known_failure_modes: List[str] = field(default_factory=list)

    @property
    def sample_method(self) -> str:
        """NPE supports direct sampling. NLE and NRE need a sampler over the learned
        proxy. VI is avoided: it raises RecursionError under sbi 0.22 on this stack."""
        return "direct" if self.engine == "NPE" else "emcee"


ZOO: Dict[str, Architecture] = {a.key: a for a in [
    Architecture(
        key="npeMaf", engine="NPE", model="maf", family="normalising_flow", repeats=1,
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Neural posterior estimation with a masked autoregressive flow. The "
                "default choice in most simulation based inference pipelines.",
        known_failure_modes=[
            "Single density estimators tend to be overconfident (Hermans et al. 2022), "
            "which is the dangerous direction because error bars come out too small.",
            "Autoregressive structure imposes a parameter ordering, so strongly "
            "degenerate parameters can be harder to capture than with a spline flow."]),
    Architecture(
        key="npeNsf", engine="NPE", model="nsf", family="normalising_flow", repeats=1,
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Neural posterior estimation with a neural spline flow. More flexible "
                "per transform than a MAF, at higher cost per epoch.",
        known_failure_modes=[
            "More parameters per transform, so it overfits sooner on small training "
            "sets, and CAMELS gives only 600 training simulations.",
            "Spline tail behaviour can be poorly constrained where the prior is wide "
            "and the data are uninformative."]),
    Architecture(
        key="npeMdn", engine="NPE", model="mdn", family="mixture_density", repeats=1,
        model_args={"hidden_features": HIDDEN, "num_components": 5},
        summary="Mixture density network posterior. Outputs Gaussian mixture "
                "parameters directly, so it is cheap and stable.",
        known_failure_modes=[
            "A finite Gaussian mixture cannot represent sharply non Gaussian or "
            "hard edged posteriors, so it can look calibrated while being biased.",
            "Component collapse when the number of components exceeds what the data "
            "supports."]),
    Architecture(
        key="npeMafEnsemble4", engine="NPE", model="maf",
        family="normalising_flow", repeats=4,
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="An ensemble of four masked autoregressive flows. This is the fix "
                "LtU-ILI Sections 3.2 and 6 recommend for single model overconfidence.",
        known_failure_modes=[
            "Four times the training cost for the same data.",
            "Averaging inflates uncertainty, which corrects overconfidence but can "
            "overshoot into under-confidence when the members already agree."]),
    Architecture(
        key="nleMaf", engine="NLE", model="maf", family="normalising_flow", repeats=1,
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Neural likelihood estimation with a MAF. Learns p(x|theta), so the "
                "prior can be changed after training without retraining.",
        known_failure_modes=[
            "Not amortised. Every new observation needs its own MCMC run, which is "
            "expensive when inference must be repeated over many observations.",
            "Learns a distribution over the data vector, so cost grows with dim(x) "
            "rather than dim(theta)."]),
    Architecture(
        key="nleMdn", engine="NLE", model="mdn", family="mixture_density", repeats=1,
        model_args={"hidden_features": HIDDEN, "num_components": 5},
        summary="Neural likelihood estimation with a Gaussian mixture likelihood.",
        known_failure_modes=[
            "Same sampling cost as any likelihood estimator.",
            "A mixture likelihood over a 25 dimensional data vector is a harder fit "
            "than a mixture posterior over two parameters."]),
    Architecture(
        key="nreMlp", engine="NRE", model="mlp", family="ratio_estimator", repeats=1,
        model_args={"hidden_features": HIDDEN},
        summary="Neural ratio estimation with a multilayer perceptron classifier. No "
                "density estimator is chosen at all, so no flow family assumption.",
        known_failure_modes=[
            "Known to degrade at high parameter dimensionality without truncation "
            "(Miller et al. 2021), and LtU-ILI Section 2.3 says so explicitly.",
            "Ranked worst of the three engines on the SLCP benchmark in LtU-ILI "
            "Figure 4."]),
    Architecture(
        key="nreResnet", engine="NRE", model="resnet", family="ratio_estimator",
        repeats=1, model_args={"hidden_features": HIDDEN, "num_blocks": 3},
        summary="Neural ratio estimation with a residual classifier. Deeper than the "
                "MLP variant at the same width.",
        known_failure_modes=[
            "Same high dimensional degradation as the MLP variant.",
            "More capacity on 600 training simulations means more overfitting risk."]),
    Architecture(
        key="lampeMaf", engine="NPE", model="maf", family="normalising_flow",
        repeats=1, backend="lampe",
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="The same masked autoregressive flow as npeMaf, built through the "
                "lampe backend instead of sbi. Paired with npeMaf it isolates the "
                "backend, holding the flow type fixed.",
        known_failure_modes=[
            "lampe is posterior estimation only. No likelihood or ratio engines.",
            "Shares the autoregressive ordering weakness of any MAF."]),
    Architecture(
        key="lampeNsf", engine="NPE", model="nsf", family="normalising_flow",
        repeats=1, backend="lampe",
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Neural spline flow through the lampe backend. LtU-ILI Section 3.4 "
                "states from experience that lampe produces tighter and better "
                "calibrated posteriors than sbi, without attaching a number. Paired "
                "with npeNsf this measures that claim.",
        known_failure_modes=[
            "Same small training set overfitting risk as any spline flow.",
            "lampe is posterior estimation only."]),
    Architecture(
        key="lampeGf", engine="NPE", model="gf", family="normalising_flow",
        repeats=1, backend="lampe",
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Gaussianization flow. LtU-ILI Section 5.6 uses this architecture in "
                "its own galactic wind example, so it is already in the framework "
                "paper and was missing from the zoo.",
        known_failure_modes=[
            "Reachable only through lampe, so no likelihood or ratio variant.",
            "Iterative Gaussianization can be slower per epoch than a plain MAF."]),
    Architecture(
        key="lampeCnf", engine="NPE", model="cnf", family="continuous_flow",
        repeats=1, backend="lampe",
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Continuous normalizing flow. Thiele 2026 calls continuous time flows "
                "the standard for many applications, while noting astrophysical "
                "posteriors are usually simple enough not to need them.",
        known_failure_modes=[
            "Sampling solves an ODE, so it can be far slower than a discrete flow "
            "despite similar training cost. Timing should be checked, not assumed.",
            "More machinery than a two parameter posterior plausibly requires."]),
    Architecture(
        key="npeMade", engine="NPE", model="made", family="autoregressive",
        repeats=1,
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Masked autoencoder for distribution estimation, the ancestor of the "
                "MAF and the fourth of sbi's four density estimators.",
        known_failure_modes=[
            "A single autoregressive pass, so less expressive than a stack of MAF "
            "transforms at the same width.",
            "Imposes a parameter ordering, like any autoregressive model."]),
    Architecture(
        key="nreLinear", engine="NRE", model="linear", family="ratio_estimator",
        repeats=1, model_args={},
        summary="The simplest ratio estimator sbi provides, a linear classifier. A "
                "floor: our best calibrated entry is already the simplest ratio "
                "estimator we have, so this tests whether simplicity is what buys it.",
        known_failure_modes=[
            "A linear classifier cannot represent a curved decision boundary, so it "
            "should underperform on accuracy if the mapping is at all nonlinear.",
            "Still needs MCMC per observation, so it is not cheap despite being "
            "small."]),
    Architecture(
        key="lampeNcsf", engine="NPE", model="ncsf", family="normalising_flow",
        repeats=1, backend="lampe",
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Neural circular spline flow. Built for periodic parameters, which "
                "sbi has no equivalent for. Cosmology rarely needs it, but a zoo that "
                "serves other astrophysics does: phases and angles are periodic.",
        known_failure_modes=[
            "Circular splines buy nothing when no parameter is periodic, and cost "
            "more than a plain spline flow.",
            "lampe is posterior estimation only."]),
    Architecture(
        key="lampeNaf", engine="NPE", model="naf", family="normalising_flow",
        repeats=1, backend="lampe",
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Neural autoregressive flow. Universal approximator for the transform, "
                "so more expressive than a MAF at higher cost.",
        known_failure_modes=[
            "Expressivity is not the binding constraint on 800 simulations, so this "
            "should overfit rather than win.",
            "No closed form inverse, so sampling is slower than a MAF."]),
    Architecture(
        key="lampeSospf", engine="NPE", model="sospf", family="normalising_flow",
        repeats=1, backend="lampe",
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Sum of squares polynomial flow. A different transform family from "
                "either the affine MAF or the spline flows.",
        known_failure_modes=[
            "Polynomial transforms can be numerically awkward in the tails.",
            "Least used of the flows here, so the least prior evidence to draw on."]),
    Architecture(
        key="lampeUnaf", engine="NPE", model="unaf", family="normalising_flow",
        repeats=1, backend="lampe",
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Unconstrained neural autoregressive flow. Removes the monotonicity "
                "constraint a NAF imposes on its transformer.",
        known_failure_modes=[
            "Same overfitting concern as the NAF at this training set size.",
            "Density evaluation requires numerical integration, so it is slower."]),
    Architecture(
        key="nleMade", engine="NLE", model="made", family="autoregressive",
        repeats=1,
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="MADE on the likelihood side, completing sbi's four density "
                "estimators across both the posterior and likelihood engines.",
        known_failure_modes=[
            "Every likelihood estimator needs MCMC per observation.",
            "A single autoregressive pass over a 25 dimensional data vector is a "
            "harder fit than over two parameters."]),
    Architecture(
        key="npeMafEnsemble2", engine="NPE", model="maf",
        family="normalising_flow", repeats=2,
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Two masked autoregressive flows. With the four member version this "
                "measures whether the calibration gain scales with ensemble size or "
                "saturates immediately.",
        known_failure_modes=[
            "Twice the training cost.",
            "If four members closed under a fifth of the gap, two should close less."]),
    Architecture(
        key="npeMafEnsemble8", engine="NPE", model="maf",
        family="normalising_flow", repeats=8,
        model_args={"hidden_features": HIDDEN, "num_transforms": 5},
        summary="Eight masked autoregressive flows, the top end of the size scan.",
        known_failure_modes=[
            "Eight times the training cost for a gain measured at under a fifth of "
            "the calibration gap with four members.",
            "Clones trained on the same 800 simulations agree with each other, which "
            "is exactly why averaging them buys little."]),
    Architecture(
        key="npeMdnEnsemble4", engine="NPE", model="mdn",
        family="mixture_density", repeats=4,
        model_args={"hidden_features": HIDDEN, "num_components": 5},
        summary="Four mixture density networks. The single MDN is the cheapest entry "
                "in the zoo at 0.7 s, so four of them still cost less than one spline "
                "flow. If ensembling helps anywhere, it should be affordable here.",
        known_failure_modes=[
            "A mixture of Gaussian mixtures is still a Gaussian mixture, so this "
            "widens uncertainty without adding shape flexibility.",
            "Component collapse affects every member the same way."]),
    Architecture(
        key="npeMixedEnsemble3", engine="NPE", model="maf",
        family="normalising_flow", repeats=1,
        mixture=(("maf", {"hidden_features": HIDDEN, "num_transforms": 5}),
                 ("nsf", {"hidden_features": HIDDEN, "num_transforms": 5}),
                 ("mdn", {"hidden_features": HIDDEN, "num_components": 5})),
        summary="A MAF, a spline flow and a mixture density network averaged "
                "together. Ensembling clones was measured to close under a fifth of "
                "the calibration gap; disagreement between families is what averaging "
                "needs, and clones on the same data do not disagree.",
        known_failure_modes=[
            "Three different failure modes rather than one, so a pathology in any "
            "member reaches the average.",
            "Members converge at different rates under one shared training budget."]),
]}


def families() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for a in ZOO.values():
        out.setdefault(a.family, []).append(a.key)
    return out


if __name__ == "__main__":
    print(f"  {len(ZOO)} architectures, matched compute {TRAIN_ARGS}")
    for fam, keys in families().items():
        print(f"    {fam:20s} {keys}")
