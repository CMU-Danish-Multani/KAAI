# The data — shared overview

Detail lives with each dataset. This page is only what applies to both.

- **Merger trees** → [../merger_trees/notes.md](../merger_trees/notes.md)
- **Point clouds** → [../point_clouds/notes.md](../point_clouds/notes.md)
- **Every term defined** → [cheatsheet.md](cheatsheet.md)

---

## 1. The problem in one page

Someone simulated ~1,000 universes on a supercomputer. Each was made by setting
**two dials**, then letting gravity run for 13 billion years:

| Dial | Plain meaning | Range |
|---|---|---|
| **Ω\_m** ("omega-m") | How much **stuff** the universe contains | 0.1 – 0.5 |
| **σ\_8** ("sigma-8") | How **lumpy** it started out | 0.6 – 1.0 |

**Your job: look at the finished universe, guess the two dials.**

Why it matters: a model that reads the dials off *simulated* universes can be
pointed at real telescope data to measure the dials of *our* universe.

### Two ways of looking at a universe

| | **Point cloud** | **Merger tree** |
|---|---|---|
| What it is | Where everything is *today* | The family history of **one** object |
| Analogy | A photograph | A family tree |
| Item = | one cube of space | one object's ancestry |

Same question, two very different inputs.

---

## 2. Vocabulary

Six terms cover almost everything here:

| Term | Means |
|---|---|
| **halo** | A blob of invisible ("dark") matter. Galaxies form inside them. |
| **galaxy** | The visible thing sitting inside a halo. |
| **merger** | Two blobs colliding and becoming one. |
| **scale factor `a`** | A clock. `a = 1` is today; `a = 0.5` is when the universe was half its current size. |
| **concentration** | How squished a blob is toward its centre. Secretly records *when* it formed. |
| **cMpc/h** | A distance unit. One is about 3 million light years. |

**Full glossary: [cheatsheet.md](cheatsheet.md)** — every field name, every
parameter, every unit, plus the traps.

## 3. What we have on disk

**1.5 GB downloaded** out of 324.5 GB available. We took only what's needed.

| Dataset | Type | Items | Size | Status |
|---|---|---|---|---|
| **CS-Trees** | merger trees | 24,996 | 1.2 GB | ✅ explored |
| **CAMELS** | point clouds | 1,000 | 96 MB | ✅ explored |
| **CAMELS-SAM** | point clouds | 1,000 | 175 MB | ✅ explored |
| **Quijote** | point clouds | 32,752 | 4.1 GB | ⬜ not downloaded |

Skipped: the velocity-task files (137 GB) and pre-built graphs (78 GB) — we can
rebuild those. Source: `users.flatironinstitute.org/~fvillaescusa/CosmoBench/`

---
