"""What a zoo entry is, and what it must prove before admission.

The brief for this project asks, as an open design question: "What standardised
benchmark suite should every zoo entry be evaluated on before admission, and who
defines it?" This file is our answer to the first half.

WHY ADMISSION NEEDS A SCREEN AND NOT JUST A SCORE
-------------------------------------------------
A recommendation engine that confidently returns the highest-scoring
architecture, when that score is partly a dataset artefact, is worse than no
engine at all. The user cannot see the problem, the number looks good, and the
mistake propagates to everyone who follows the advice.

We measured a concrete instance. On the CAMELS suite the number of galaxies in a
box correlates with Omega_m at about 0.73, because an object is only recorded
once it holds roughly twenty simulation particles and particle mass depends on
Omega_m. Whether a model exploits that is decided entirely by its aggregation
step: summing preserves the count, averaging divides it out.

    measured, CAMELS Omega_m, identical architecture:
        sum pooling   0.8020        mean pooling   0.6600
    measured, CAMELS-SAM where the count is fixed at 5000:
        sum pooling   0.5170        mean pooling   0.5196

The second line is the control. Where the count cannot vary, the choice is worth
nothing. So the effect is the artefact and not the architecture.

An entry that scores 0.80 by counting and an entry that scores 0.68 by reading
structure are not comparable, and a zoo that ranks them together is misleading.
Every entry therefore carries a leakage screen result, and entries that leak are
admitted but LABELLED, never silently ranked alongside clean ones.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Modality(str, Enum):
    """What the model consumes."""
    POINT_CLOUD = "point_cloud"
    SUMMARY_VECTOR = "summary_vector"
    GRAPH = "graph"
    FIELD = "field"
    TREE = "tree"


class OutputKind(str, Enum):
    """What the model produces. This is the deepest split in the taxonomy.

    A point estimate and a posterior are not interchangeable. Recommending one
    where the user needs the other is a category error, not a ranking mistake,
    so it gates retrieval before any score is compared.
    """
    POINT_ESTIMATE = "point_estimate"
    POSTERIOR = "posterior"
    EMBEDDING = "embedding"


class Role(str, Enum):
    """Where the entry sits in a pipeline.

    Encoders and inference heads compose rather than compete: every posterior
    estimator needs something to turn raw data into a summary first. Ranking an
    encoder against a flow is meaningless, so role also gates retrieval.
    """
    ENCODER = "encoder"
    AGGREGATION = "aggregation"
    INFERENCE_HEAD = "inference_head"
    END_TO_END = "end_to_end"


class LeakStatus(str, Enum):
    CLEAN = "clean"
    LEAKS = "leaks"
    UNSCREENED = "unscreened"


@dataclass
class LeakScreen:
    """Result of the admission screen.

    `r2_recovering_count` is the held-out R2 of a linear probe reconstructing the
    number of input elements from the entry's output, measured with the point
    distribution held fixed so the count is the only thing that varies.

    The obvious alternative, feeding duplicated points and checking the output
    moves, is INVALID and we nearly shipped it. Duplicating every point cannot
    move a maximum, so max aggregation returns bit-identical output and appears
    perfectly clean while tracking the count at r = 0.87.
    """
    r2_recovering_count: Optional[float] = None
    status: LeakStatus = LeakStatus.UNSCREENED
    seeds: int = 0
    note: str = ""


@dataclass
class Calibration:
    """Whether a posterior entry's stated uncertainty is honest.

    R2 says how close a prediction lands. It says nothing about whether the
    error bar is trustworthy, and a model that is confidently wrong scores well
    on R2 while being useless for science.

    `coverage_90` is the fraction of test cases where the truth fell inside the
    model's stated ninety percent interval. Below nominal means OVERCONFIDENT,
    which is the dangerous direction: tight error bars around wrong answers.
    Above nominal wastes information but misleads nobody.
    """
    coverage_90: Optional[float] = None
    calibration_error: Optional[float] = None
    overconfident: Optional[bool] = None
    note: str = ""


@dataclass
class Measurement:
    """One score, with everything needed to trust or reproduce it.

    `why` is not decoration. A zoo that reports only numbers is a leaderboard,
    and a leaderboard cannot tell a user whether a score will transfer to their
    problem. The reason a number came out as it did is what makes a
    recommendation actionable.
    """
    task: str
    metric: str
    value: float
    spread: Optional[float] = None      # None means one run, never 0.0
    seeds: int = 1
    note: str = ""
    why: str = ""


@dataclass
class Entry:
    """One architecture in the zoo."""
    key: str
    name: str
    role: Role
    modality: Modality
    output: OutputKind
    source: str                          # arXiv id or library
    summary: str                         # one sentence, plain language
    parameters: Optional[int] = None
    minutes_per_fit: Optional[float] = None
    hardware: str = ""
    measurements: List[Measurement] = field(default_factory=list)
    leak_screen: LeakScreen = field(default_factory=LeakScreen)
    calibration: Optional[Calibration] = None
    failure_modes: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def best(self, task: str, metric: str = "R2") -> Optional[Measurement]:
        hits = [m for m in self.measurements if m.task == task and m.metric == metric]
        return max(hits, key=lambda m: m.value) if hits else None

    def admissible(self) -> bool:
        """Two admission checks, plus at least one measurement.

        1. Screened for leakage. Leaking entries are admitted but LABELLED, never
           silently ranked beside clean ones.
        2. If the entry outputs a posterior, its calibration must be measured.
           An unmeasured error bar is not a feature, it is a claim.
        """
        if self.leak_screen.status is LeakStatus.UNSCREENED:
            return False
        if not self.measurements:
            return False
        if self.output is OutputKind.POSTERIOR and self.calibration is None:
            return False
        return True

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k in ("role", "modality", "output"):
            d[k] = getattr(self, k).value
        d["leak_screen"]["status"] = self.leak_screen.status.value
        return d

    def headline(self, task: str) -> str:
        """One line a person can act on: the number and the reason for it."""
        m = self.best(task)
        if not m:
            return f"{self.name}: not measured on {task}"
        spread = f" +/- {m.spread:.4f}" if m.spread is not None else " (single run)"
        line = f"{self.name}: {m.value:+.4f}{spread}"
        return line + (f". {m.why}" if m.why else "")
