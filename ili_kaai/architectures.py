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
from typing import Any, Dict, List

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
    model: str                        # maf, nsf, mdn, mlp, resnet
    family: str                       # normalising_flow, mixture_density, ratio_estimator
    repeats: int                      # ensemble size
    model_args: Dict[str, Any] = field(default_factory=dict)
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
