"""Turn raw merger trees into normalised, batched tensors ready for training.

HOW THE CODE FLOWS
------------------
There is one entry point, get_loaders(). Everything else is called by it.

    get_loaders()                      <- you call this
      |
      +-- get_split("train")           <- no stats passed, so it makes them
      |     |
      |     +-- load("train")          <- reads the .pt file off disk (slow)
      |     +-- feature_stats(trees)   <- measures mean/std of every feature
      |     +-- normalise(trees, ...)  <- rescales using those measurements
      |
      +-- get_split("val", stats)      <- stats PASSED IN, so it reuses them
      |     |
      |     +-- load("val")
      |     +-- normalise(trees, ...)  <- note: feature_stats NOT called again
      |
      +-- DataLoader(...)              <- PyG groups trees into batches

The one thing to notice: feature_stats() runs exactly ONCE, on train. Val and
test get train's numbers handed to them. That is the whole reason get_split()
takes a `stats` argument at all.

WHY NORMALISE AT ALL
--------------------
Raw features sit on wildly different scales -- mass is around 11, scale factor
around 0.4, and edge_attr runs 2 to 70. Inside a neural network a numerically
bigger input simply has more influence, whether or not it is more important.
Rescaling everything to mean 0 / std 1 puts them on equal terms.

WHY TRAIN-ONLY STATS
--------------------
If we measured mean/std across all splits, facts about the test set would leak
into training. Scores would come out higher and be wrong. This is silent when
it happens, which is why it is worth being deliberate about.
"""

import torch
from torch_geometric.loader import DataLoader

from merger_trees.load import load


def feature_stats(trees):
    """Measure the mean and spread of every feature.

    CALLED BY: get_split(), and only when stats weren't provided -- i.e. only
    for the training split.

    Stacks every node from every tree into one tall table, then takes column
    means and standard deviations. clamp_min avoids dividing by zero later if
    some feature never varies.
    """
    x = torch.cat([t.x for t in trees])          # all nodes: [total_nodes, 4]
    e = torch.cat([t.edge_attr for t in trees])  # all edges: [total_edges, 1]
    return {
        "x_mean": x.mean(0), "x_std": x.std(0).clamp_min(1e-6),
        "e_mean": e.mean(0), "e_std": e.std(0).clamp_min(1e-6),
    }


def normalise(trees, stats):
    """Rescale every feature to roughly mean 0, std 1.

    CALLED BY: get_split(), once per split.
    RECEIVES:  stats measured on TRAIN, whichever split this is.

    Modifies the trees in place, so nothing is copied.
    """
    for t in trees:
        t.x = (t.x - stats["x_mean"]) / stats["x_std"]
        t.edge_attr = (t.edge_attr - stats["e_mean"]) / stats["e_std"]
    return trees


def get_split(name, stats=None, limit=None):
    """Load one split from disk and normalise it. Returns (trees, stats).

    CALLED BY: get_loaders(), once per split.
    CALLS:     load() -> feature_stats() (train only) -> normalise()

    The `stats` argument is what keeps the splits honest:
      stats=None   -> measure them here.  Only ever used for train.
      stats=given  -> reuse train's.      Used for val and test.
    """
    trees = load(name)                  # -> data_load/trees.py, reads the file
    if limit:
        trees = trees[:limit]           # small subset, for quick checks
    if stats is None:                   # only true for train
        stats = feature_stats(trees)
    return normalise(trees, stats), stats


def get_loaders(batch_size=64, limit=None, splits=("train", "val")):
    """Build dataloaders for the requested splits. THE ENTRY POINT.

    CALLS: get_split() once per split, then wraps each in a PyG DataLoader.

    A DataLoader hands out batches during training. PyG's version does
    something specific: instead of padding trees to equal size, it glues N
    trees into one big disconnected graph and adds a `batch` vector recording
    which tree each node came from. That vector is how pooling later knows
    where one tree ends and the next begins.

    Only the training loader is shuffled -- shuffling val would change nothing
    except make results harder to compare between runs.
    """
    train, stats = get_split("train", limit=limit)     # stats born here
    loaders = {}
    for name in splits:
        # train is already loaded above; other splits reuse its stats
        trees = train if name == "train" else get_split(name, stats, limit)[0]
        loaders[name] = DataLoader(trees, batch_size=batch_size,
                                   shuffle=(name == "train"))
    return loaders, stats
