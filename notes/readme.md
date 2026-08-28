# KAAI

Neural architecture search for cosmology: predict a universe's settings from
simulation data, and search for the best model design automatically instead of
guessing it.

Plain-language summary of the whole project:
<https://claude.ai/code/artifact/4059d55a-a709-4bef-a834-bf2139cb3d27>

## Layout

The two datasets are kept **completely separate** — separate loading,
exploration, plots and notes. Understand one at a time.

```
merger_trees/          family histories of dark matter blobs
    load.py                reading + summarising
    dataset.py             normalising + batching for training
    explore.py             the analysis (7 sections)
    notes.md               what we found
    plots/                 01-06
    training/              step checks
    validate/              (empty)

point_clouds/          cubes of space full of galaxies
    load.py                reading + summarising
    explore.py             the analysis (5 sections)
    notes.md               what we found
    plots/                 07-10
    training/              (empty)
    validate/              (empty)

common/viz.py          chart style + correlation helpers, shared by both
notes/                 project-level: this file, plans.md, cheatsheet.md
data/                  downloaded datasets (gitignored, 1.5 GB)
resources/             the two papers
```

## Run it

```bash
conda activate kaai

python merger_trees/explore.py                          # trees: 7 sections
python merger_trees/training/step1_check_dataloader.py  # verify batching
python point_clouds/explore.py                          # clouds: 5 sections
python point_clouds/load.py                             # print one cloud
```

## Where to start reading

| If you want | Read |
|---|---|
| What any term means | [cheatsheet.md](cheatsheet.md) |
| What's in the trees | [../merger_trees/notes.md](../merger_trees/notes.md) |
| What's in the clouds | [../point_clouds/notes.md](../point_clouds/notes.md) |
| The research plan | [plans.md](plans.md) |
| Related papers, and where the novelty is | [literature.md](literature.md) |
| What has actually been measured | [results.md](results.md) |
| Papers read, and what to take from each | [related_papers.md](related_papers.md) |

## The two papers

**resources/KAAI_Dataset.pdf → CosmoBench** (NeurIPS 2025, arXiv 2507.03707).
The data: 34k point clouds and 25k merger trees from simulations costing 41
million core-hours, with baselines for four tasks. Headline finding: on Quijote
a **49-parameter least-squares fit beats a 671,000-parameter GNN**.

- Data portal: <https://cosmobench.streamlit.app/>
- Reference code: <https://github.com/nhuang37/cosmology_benchmark>

**resources/Biological_FM.pdf → BioArc** (ICML 2026, arXiv 2512.00283). The
method, not data. Searches a heterogeneous space of network designs with a
weight-sharing supernet, then adds an LLM agent that learns from past searches.
Built for DNA and protein models — whether it transfers to cosmology is part of
the question.

## Environment

Env `KAAI` at `~/miniconda3/envs/KAAI`, Python 3.12, torch on Mac GPU (MPS).

Rebuilding it is **not** `pip install -r`. Read
[../KAAI_requirements.txt](../KAAI_requirements.txt) first — it is a manifest,
and four things must be done a specific way:

1. torch from conda-forge, not pip (OpenMP clash → hard crash)
2. Pylians needs three patches to its `setup.py`
3. HDF5 pinned to 2.1.0 to match h5py
4. The package is `Pylians`, not `Pylians3`

## Status

Done: environment, data downloaded, both datasets explored, tree batching
verified. **No model exists yet** — the next step is the first code that learns
anything.
