# Related papers

What this file is: a working reference for the KAAI project, whose goal is to find the optimal neural network for predicting cosmological parameters (Omega_m, sigma_8) from CosmoBench galaxy and halo point clouds. Entries are organised by what each paper is useful for, not alphabetically or by date. Twelve papers were read in full and summarised here from those reads. Six further entries were read directly by the human earlier and are recorded from those notes.

Convention used throughout: a line beginning MEASURED reports a number taken from a paper or from our own runs. A line beginning INTERPRETED is a reading of what that number means. Confidence is tagged where a claim is not settled.

Our own reproduced baselines, for reference in what follows:

- MEASURED, 2PCF + MLP: CAMELS Omega_m R^2 0.860, CAMELS-SAM 0.778.
- MEASURED, LLS with 49 parameters: CAMELS 0.803, CAMELS-SAM 0.752.
- MEASURED, our GNN (67k parameters): CAMELS Omega_m 0.660 mean pool, 0.802 sum pool.
- MEASURED, our DeepSets: CAMELS Omega_m -0.001 mean pool, 0.523 sum pool.
- MEASURED, galaxy count alone with no model: CAMELS Omega_m 0.506.
- MEASURED, on CAMELS-SAM (count fixed at 5000) the pooling choice makes no difference. This is our null.

---

## 1. Papers that define the problem

### CosmoBench (arXiv 2507.03707, NeurIPS 2025 Datasets and Benchmarks)

The dataset and the baseline suite this whole project sits on. 34k point clouds and 25k merger trees drawn from CAMELS, CAMELS-SAM and Quijote. Recorded from the human's earlier direct read, not re-read here.

The headline that matters to us: on Quijote, a least squares fit with 49 parameters beats a graph neural network with 671k parameters. INTERPRETED: the benchmark itself already reports that expressive deep models are not winning on this data, so "we built a bigger network" is not a result. The publishable shape has to be either a network that genuinely beats the hand built summary statistic, or an explanation of why the hand built statistic keeps winning. Our pooling and count finding is currently the strongest candidate for the second.

### Leakage and the Reproducibility Crisis in ML-based Science (arXiv 2207.07048, Kapoor and Narayanan, 2022)

Relevance HIGH. Zero architecture content, but it supplies the vocabulary and the reporting standard for our central finding.

It surveys 20 review papers across 17 fields covering 329 affected papers, and defines leakage as a spurious relationship between inputs and target that arises as an artifact of data collection, sampling or preprocessing. The taxonomy has three families. L1 is train and test not kept apart (L1.1 no test set, L1.2 joint preprocessing, L1.3 feature selection on train and test together, L1.4 duplicates). L2 is the model using a feature that is not legitimate, meaning a proxy for or a downstream consequence of the outcome. L3 is the test set not being drawn from the distribution of scientific interest (L3.1 temporal, L3.2 non-independence, L3.3 sampling bias).

Where our finding sits, and this is the load bearing paragraph for the paper we want to write:

- Our galaxy count leak is L2. The definition fits without stretching: the count to Omega_m correlation of roughly 0.73 arises because a halo is only recorded once it holds about 20 particles, and particle mass depends on Omega_m. That is data collection, not physics.
- Do not frame it as L3.3 sampling bias. A reviewer defeats that framing by saying the benchmark defines its own distribution of interest.
- The novelty hook is a real gap in their framework. In all 329 papers they survey, L2 is decided by inspecting a named column and corrected by deleting that column. Their appendix Table A5 is literally a list of column names to drop. Our input is bare 3D positions with no features at all. The illegitimate feature (N, the cardinality of the point set) is manufactured by the architecture: sum pooling computes it, mean pooling cannot. Their model info sheet question Q21 asks the researcher to argue that each feature is legitimate, which is vacuous on a featureless point cloud, so the sheet passes clean while the leak is live. Proposed claim: architecture-induced L2 leakage, where the illegitimate feature is not in the dataset but is constructed by an operator choice.
- Second hook, a generalisation of their L1.3. Feature selection on train and test together is their L1.3. Architecture search ranked by held out test R^2 is the same error one level up. Combine that with an open L2 channel and the search converges on the leak, because the leak is the cheapest available signal. INTERPRETED, and this is the sentence we want: an automated search finds the illegitimate feature faster than a human reviewer does.
- Useful citation for a likely referee objection: 7 of their 20 reviews found leakage despite the field using a standard, widely used dataset, because the field lacked fixed splits and fixed metrics. Being a NeurIPS Datasets and Benchmarks paper does not immunise CosmoBench.

Caveat, and it is the sharpest one in this file. Their own L2 test can be turned against us. Legitimacy is explicitly a domain judgement they refuse to formalise. A referee can say that if the distribution of scientific interest is the CAMELS suite itself, then N is a perfectly legitimate feature and we have only found that N is predictive, which is neither leakage nor news. Pre-empting this needs a measurement, not an assertion: show that the count to Omega_m relation shifts or breaks when the resolution threshold or the particle mass changes. Until that is measured, calling it leakage is our opinion. Second caveat: their domain is tabular social science and clinical prediction with binary classification, and every correction they apply is dropping a column or fixing an imputation. Using their language without adapting it will read as borrowed. Third: v1 from July 2022, and the journal version was not checked, so subtype labels should be re-verified against whatever we cite.

### On the effects of parameters on galaxy properties in CAMELS and the predictability of Omega_m (arXiv 2503.22654, Contardo, Trotta, Hogg, Villaescusa-Navarro)

Recorded from the human's earlier direct read. It explains Omega_m predictability in CAMELS through a physical displacement of the galaxy feature distribution.

Why it belongs in this section: notably it never mentions resolution, particle mass, selection, or abundance, and it treats galaxy count only as a nuisance variable to be balanced during classifier training. INTERPRETED: the physical explanation and the artifact explanation are not in competition in the literature yet, because the artifact explanation has not been stated. That is the open lane. HONEST CAVEAT: their physical mechanism may well be real and may coexist with the artifact, so our claim should be that part of the measured Omega_m predictability is resolution driven, not that all of it is.

### Understanding Pooling in Graph Neural Networks (arXiv 2110.05292, Grattarola, Zambon, Bianchi, Alippi)

Relevance HIGH. Implementable items are in section 2. Its role here is establishing that the prior art gap is real.

This is the canonical pooling survey. It defines every pooling operator as select, reduce, connect (Sel assigns N nodes to K supernodes through a membership matrix S; Red is almost always X' = S^T X; Con is almost always A' = S^T A S), taxonomises thirty-plus methods on four axes, and evaluates eight of them on attribute preservation, structure preservation and task performance rather than accuracy alone. Global readout is placed in the taxonomy as pooling that is fixed with K = 1, and then excluded from every experiment.

MEASURED, from a grep of the full text: the words leak, shortcut, spurious, confound, dataset bias and artifact return zero occurrences. The single closest sentence reads "if the relative graph size is important for solving a particular task, adaptive methods should be preferred", which treats size sensitivity as a capability to preserve rather than a hazard to screen. INTERPRETED: the field's position is that no pooling operator is best a priori and it depends on the task. Nobody has said that the pooling choice can decide whether the model is allowed to cheat. That is our lane, and this is the citation that shows it is open.

Caveat that cuts at us as well: their own classification architecture uses an unquestioned sum readout in every experiment, across datasets with variable graph size, including the count-flavoured Colors-3 and Triangles. By our own finding that is a confounded design, so their operator ranking in Table 6 may partly measure how each operator interacts with a size-leaking readout. Do not treat that table as clean evidence.

### The Role of Node Features in Graph Pooling (arXiv 2605.06250, von Pichowski et al., 2026)

Relevance MEDIUM, and there is a terminology collision to be careful about. Their "pooling" means hierarchical coarsening, not the global readout our finding is about. Their global readout is fixed to max in every experiment and never varied. Node count, graph size and cardinality never appear as a signal anywhere in the paper. Do not cite it as support for anything we claim about cardinality.

What it is good for here: its headline claim is that whether pooling helps is a property of the dataset rather than a shortcoming of the operator. That has the same shape as our measured CAMELS versus CAMELS-SAM result, so it is the nearest published statement in form and we should cite it and draw the boundary ourselves rather than let a referee draw it. They study which nodes get grouped together; we study whether the global readout can see how many nodes there are. Neither subsumes the other.

---

## 2. Building blocks we can put in the search space

This is the most important section. Each item says what to implement.

### 2.1 Pooling and readout

This is the axis our finding lives on, so the search space needs to cover it properly rather than treat sum versus mean as a nuisance hyperparameter.

**Fishnets aggregation (arXiv 2310.03812, Makinen, Alsing, Wandelt). Relevance CRITICAL.**

The idea in plain terms: if you have many independent measurements of the same quantity, the log-likelihood adds up, so its gradient (the score, t) adds up and its curvature (the Fisher information, F, meaning how much a measurement tells you) also adds up. The best combined estimate is F_total^{-1} t_total, which is the classical inverse-variance weighted average. Fishnets learns t and F instead of assuming a likelihood.

Implement, roughly 15 lines, as a drop in for the global pool:

- Per element: t_i = Linear(h_i) -> R^{n_p}; l_i = Linear(h_i) -> R^{n_p(n_p+1)/2}; pack l_i into a lower triangular L_i; apply softplus to the diagonal of L_i; F_i = L_i L_i^T, which is positive definite by construction.
- t = scatter_sum(t_i, batch); F = scatter_sum(F_i, batch); out = solve(F + eps * I, t). Use torch.linalg.cholesky plus cholesky_solve, not an explicit inverse. The ridge eps * I is in their own equation 28 as a prior. If MPS lacks cholesky_solve, move that one operation to CPU.
- Sizes: bottleneck n_p 8 to 16, a single Linear (not an MLP) before the aggregation, 3 message passing layers. One tiny solve per box is negligible cost. A per-node version inside message passing is one small solve per neighbourhood per layer and may be too slow on MPS.

Why this is the most valuable single block in this file: the sum appears in both numerator and denominator, so the output does not grow with N while its variance shrinks like 1/N. That splits the two things we are trying to separate into two named tensors. F_total^{-1} t_total is the count-suppressed shape channel. F_total (or log det F_total, which grows like n_p log N) is the count channel. Concatenate log det F_total onto the readout head as one extra scalar and we have an on and off switch for the resolution artifact inside a single architecture, instead of inferring its size by comparing two different architectures. Run it both ways on CAMELS and the R^2 delta is the measured value of the artifact.

Write the prediction down first, and it follows from their own appendix E: if the Fisher net collapses to a constant F_i = F0, then (N F0)^{-1} sum_i t_i = F0^{-1} mean_i(t_i), so Fishnets with flat weights is mean pooling composed with a linear map. Fishnets therefore sits on the mean pool side of our leak, and the prediction is that on CAMELS Omega_m it lands near 0.660 and not near 0.802. If it lands well above 0.660, something is leaking and we go and find it.

Caveat, and it is structural: the additivity of Fisher matrices assumes independent set elements. Galaxies are strongly clustered, so on our data the additivity is a heuristic inductive bias, not a theorem. The paper never tests on correlated set members, and in their own GNN experiments they drop the Fisher loss entirely and keep only the aggregation. Second, their set size invariance was demonstrated where more elements means more samples of the same distribution. In CAMELS, more galaxies means a different cosmology, so N is signal and not sample size. Fishnets does not remove the leak, it relocates it into F_total, and our raw CAMELS number may go down relative to 0.802 while being the more honest figure. Expect that, do not read it as failure.

**Neuralized Kolmogorov Mean, QUANN (arXiv 2602.04941, Tokar and Sanner, ICLR 2026). Relevance CRITICAL.**

A classical quasi-arithmetic mean is M_f(X) = f^{-1}((1/n) sum_i f(x_i)). f = identity gives the ordinary mean, f = log gives the geometric mean, f = x^p gives the power mean, f = exp(w x) approaches max. The paper learns f with an invertible network (a RevNet coupling block: y1 = x1 + F(x2), y2 = x2 + G(y1), undone by subtracting in reverse).

MEASURED, their Table 2: approximation error is O(1) for mean-decomposable targets, O(log(n)/w) for max-decomposable targets, and O(n), that is unbounded, for sum-decomposable targets. So this is a learnable pooling that interpolates mean to max and provably cannot express sum. For us that exclusion is the most valuable property in the paper, not a shortcoming: it gives a pooling arm that is strictly more expressive than fixed mean pooling yet provably cannot read the count.

Implement: RevNet block is about 25 lines. Divide by the true per-cloud count taken from the scatter index.

IMPLEMENTATION TRAP, verified by the reader against the released code. In their utils/modelutils.py the forward is h = psi(x); h[~mask] = 0; mu = h.mean(dim=1); return psi.inverse(mu). That mean(dim=1) divides by the padded length, not the true cardinality, and their collate pads to the per-batch maximum. MEASURED on a synthetic reproduction: the released form gives a pooled vector ratio of 2.0 between n = 200 and n = 100 clouds padded together, while the paper's equation 3 gives 1.0. Their MeanSet baseline has the same issue. Our scatter-index layout in point_clouds/pointnet.py avoids padding entirely, so this does not arise if we divide by the real count. Do not copy the reference code verbatim.

MEASURED, invertibility on our hardware: maximum absolute round trip error |psi^{-1}(psi(x)) - x| in float32 was 1.9e-06 with 2 blocks on both CPU and MPS, 3.5e-06 with 8 blocks on CPU. Invertibility is not a practical obstacle.

MEASURED cost on our hardware: batch 32, 5000 points, latent 64, MPS: 7.9 ms per training step with mean pooling versus 45.6 ms with NKM pooling, about 5.8x (4 runs each, mean pool 7.5 to 8.1 ms, NKM 44.2 to 46.2 ms). Against the 5.5 minute budget that is the difference between 5.5 minutes and roughly half an hour for a pooling-dominated model.

Sizes for a parameter matched comparison against our 67k GNN: latent 64, psi hidden [64], 2 blocks is about 30k parameters; latent 64, psi hidden [32] is about 21k. Their own point cloud config (latent 128, psi hidden [128], encoder [256,256], predictor [256]) is 174k to 209k.

Use the equivariant variant, not the plain DeepSets shaped one. Their equation 14: f_i(X) = rho([x_i, NKM(X)]), concatenating each point's own feature with the pooled global summary. MEASURED, ModelNet40: equivariant QUANN-2 0.846 / 0.882 versus equivariant DeepSets 0.798 / 0.844 (n = 4 replicates), whereas the invariant QUANN-1 scored 0.676 / 0.688 / 0.686 against PointNet max pooling at 0.788 / 0.788 / 0.791. INTERPRETED, using the authors' own explanation: 3D geometry is read through extreme values, which max pooling captures natively and a mean-family operator only approximates. Galaxy clouds are 3D point clouds, so the plain variant is the one least likely to help us.

Also useful: multi-head pooling (several parallel NKMs with independent psi), so one head can converge to mean-like and another to max-like behaviour without us choosing. And the diagnostic use: after fitting, probe the learned psi on a 1D sweep and read off what family it converged to. The one-number version is the HPDS baseline they compare against (Kimura et al. 2024): pooling = (mean_i softplus(h_i)^p)^(1/p) with a single learned scalar p, about 8 lines. p near 1 is mean, large p is max, p near 0 is geometric. Running HPDS on CAMELS and CAMELS-SAM gives a directly readable scalar answer to what the data wanted.

**Alpha normalisation sweep (from the SRC framework, arXiv 2110.05292).**

Under select, reduce, connect, global sum pooling is Red(S, X) = S^T X with S a single all ones column, and global mean pooling is the identical operator with S divided by its column sum. The only difference is a normalisation exponent. So implement Red_alpha(X) = (sum_i x_i) / N**alpha with alpha a scalar in [0, 1]: alpha = 0 is sum, alpha = 1 is mean. Sweep alpha over {0, 0.25, 0.5, 0.75, 1.0} on CAMELS with everything else frozen, and plot two curves on one axis: R^2 versus alpha, and corr(prediction, galaxy count) versus alpha. Five runs at 5.5 minutes is about 28 minutes, and it converts a binary claim into a monotone dose response curve, which is much harder to dismiss than two points. Run the same sweep on CAMELS-SAM, where the curve should be flat, giving a null with a shape rather than a single number.

Related and independent: NequIP divides the aggregated message by sqrt(|E|) (from arXiv 2410.20516 appendix B, equation 18). alpha = 0.5 is exactly that. INTERPRETED: it grants partial count access, so the three point comparison sum, sqrt(N), mean decomposes how much of our 0.660 to 0.802 gain is raw count and how much is count-weighted structure. Nobody in this literature has reported that decomposition.

**Attention readout, not attention message passing (arXiv 2410.20516 appendix D, Table 4).**

MEASURED: swapping the GNN's mean readout for a global multi-head attention aggregation moved Omega_m MSE from 2.77 to 2.60 and sigma_8 from 4.84 to 2.84, with fewer parameters (1441k to 915k). Local attention inside message passing made it worse (3.00 / 8.82). Invariant attention was catastrophic on sigma_8 (13.33, which is exactly the prior variance, so R^2 = 0).

INTERPRETED, and this is the best fit to our situation in the whole file: softmax attention weights sum to 1, so attention pooling is count-blind in the same algebraic way mean pooling is. It is a leak-safe pooling that still beats mean. Directly droppable into our 67k GNN, no new symmetry machinery.

**Janossy pooling at k = 2, which is a learnable 2PCF (arXiv 1811.01900, Murphy et al., ICLR 2019). Relevance HIGH.**

Any order-blind function can be built by averaging an order-sensitive function over all reorderings. That is intractable, so they restrict the inner function to look at only the first k items of each ordering, which collapses the sum to all ordered k-tuples. They prove the reachable function class at order k-1 is a strict subset of that at order k, and that k = 1 is exactly DeepSets.

Implement: replace the global sum or mean over N galaxy embeddings with a mean over pairs of MLP([h_i, h_j, pair_features]). Make the inner function symmetric in its two arguments (easy for us, since physical pair features such as the minimum-image separation are already symmetric), which collapses ordered pairs to unordered pairs for a factor of two saving. For CAMELS (N in the hundreds to low thousands) exact k = 2 is affordable. For CAMELS-SAM, exact k = 2 is roughly 25M ordered pairs per cloud, so sample M around 8k pairs per cloud per step, which is roughly 1e8 FLOPs with a 64 wide MLP and fits the budget. Average roughly 8 stochastic draws at test time, which also hands us a free spread estimate. Optional variance reduction auxiliary loss: penalise ||f(h_s) - f(h_s')||^2 for two independent draws, costing one extra forward pass through the pooling head only.

Why it matters for framing: Janossy pooling comes from Janossy densities in point process theory, and the cosmological n-point correlation functions are the factorial cumulant densities of exactly that formalism. So the pooling order k is the order of the correlation function the network can represent. Our own baseline ordering then falls out of their Theorem 2.1: our 2PCF + MLP at 0.860 is a hand built k = 2 with a fixed inner function, our DeepSets at 0.523 is a k = 1, and the k = 1 model loses. INTERPRETED, not proved on our data.

It also separates two things our sum versus mean result currently confounds. Their Definition 2.1 passes |h| to the inner function as an explicit argument and normalises by a factorial, so it is mean-like and count-normalised by default. That lets us make N an explicit ablatable scalar input to the head (for example log N) while k separately controls capacity. Pooling order k is capacity; the explicit N input is leak access. Prediction to write down first: the N switch should reproduce the sum versus mean gap on CAMELS, and should be a measured null on CAMELS-SAM.

Caveat, and it is the honest one: their own real structured-data experiment does not support higher order pooling. MEASURED: on Cora, plain mean pooling 0.860 versus LSTM-Janossy 0.860; on PPI, mean pooling 0.767 versus best LSTM-Janossy 0.757. Every large Janossy win is on synthetic integer arithmetic over sequences of length 5 to 10. Also a sampling trap specific to us: uniformly sampled pairs from a 5000 point box are dominated by large separations, where the clustering signal is weakest. Importance sampling near pairs via a radius or kNN graph fixes the signal but changes the estimand and quietly encodes local density, which is exactly the back door our leak lives behind. Any claim that k-ary pooling is leak free by construction needs measuring, not asserting.

**Multi-aggregation concat readout (arXiv 2306.12327, Wu and Jespersen).**

Their model concatenates global_add_pool, global_mean_pool and global_max_pool and feeds the 3 x latent vector to the final MLP. Cheap, and it gives the network the count-sensitive statistic and the count-insensitive ones at once, so we can zero channels at inference and measure which one the model leans on.

### 2.2 Message passing and edge features

**PNA aggregators and degree scalers (arXiv 2004.05718, Corso et al., NeurIPS 2020). Relevance CRITICAL.**

Two ideas bolted together. Four aggregators (mean, standard deviation, min, max) concatenated, on the argument that separating multisets of size n needs n aggregators. And degree scalers: they reframe sum as mean multiplied by a scaler linear in degree, then replace the linear scaler with a logarithmic one, S(d, alpha) = (log(d+1) / delta)^alpha with alpha in {+1, 0, -1} and delta a single global constant equal to the mean of log(d+1) over all training nodes. Four aggregators times three scalers gives 12 channels, plus the node's own features gives 13F, so the update MLP maps R^{13F} -> R^F.

Direct warning first, because this is the block most likely to silently undo our finding. In a radius graph over a fixed volume periodic box, the expected degree is exactly (N-1) * (4/3) pi Rc^3 / L^3 * (1 + xibar(Rc)), so N is a multiplicative factor inside every degree. Published PNA therefore injects the galaxy count at every node and every layer, and it survives mean pooling. Bolting on stock PNA re-opens the 0.73 count shortcut in one line without touching the pooling argument. On a benchmark where node count correlates with the label, published PNA is a leak amplifier by design, and the paper would score that as a win.

The repair, and the most valuable single line in that paper for us: feed the scaler a density normalised degree. Set dtilde_i = d_i / lambda_c with lambda_c = (N_c - 1) * (4/3) pi (cutoff / box)^3, the degree that cloud c would have under a uniform random process with the same number of galaxies in the same box. Then dtilde_i = 1 + delta_i is the local overdensity at scale Rc and is invariant to N by construction. Use S(dtilde, alpha) = (log(1 + dtilde) / delta)^alpha. lambda_c is one float per cloud, computable in GraphSet.__init__ next to n_nodes.

The controlled experiment this enables, leaning on the null we already have. Three arms: (A) identity scaler only, (B) raw degree scalers as published, (C) density normalised scalers. Run all three on CAMELS and on CAMELS-SAM. Prediction written down first: on CAMELS, B beats A by roughly the size of the leak and C lands between them, so the B minus C gap measures the leak in R^2 units. On CAMELS-SAM, B and C become near identical because lambda_c is a constant there, so any gain of B or C over A there is genuine clustering physics and cannot be count.

The std aggregator is the obvious missing piece for sigma_8. Our GNN scores 0.660 on Omega_m but 0.193 on sigma_8, and sigma_8 is a variance of the density field, while our layer aggregates neighbours with mean only and therefore discards spread by construction. Implement their equation 3: sigma_i = sqrt(ReLU(mu_i(X^2) - mu_i(X)^2) + eps). The ReLU and the eps are load bearing for gradient stability. Two lines, and the cheapest shot at the sigma_8 number we have.

Cost control. MEASURED parameter counts at hidden = 64: the per-layer update goes from 8,256 to 53,312, so a 3 layer model goes from about 67k to about 200k. Towers fix this: split hidden into T groups, aggregate inside each, then mix. T = 1 costs 53,312 per layer, T = 2 costs 26,688, T = 4 costs 13,376, T = 8 costs 6,720. T = 4 gives full PNA at roughly half our current update cost.

Also worth taking: a GRU after the update plus weight sharing across layers 2 to M (encode, process, decode). Their appendix I states the GRU is what stops oversmoothing and helps most the architectures with no skip across the aggregation, which is ours. This lets us go from 3 layers to 6 or 8 at nearly constant parameter count, which matters because at cutoff 0.015 x box each hop reaches only 0.375 Mpc/h on CAMELS.

Two more items. Normalised moments, their equation 4: M_n(X) = (E[(X - mu)^n])^(1/n) for n > 1, where the nth root is the whole point because it makes every aggregator scale linearly with feature magnitude. Their appendix D ablation says gains saturate around n = 3. Skewness of the local density field is a known non-Gaussianity probe in cosmology, so n = 3 is the one extra arm worth a run. And delta is a train-set statistic: compute it on the train split and reuse it unchanged for validation and test, exactly like label_mean and label_spread, which GraphSet already threads through. Computing it per split or per batch would be a quiet leak.

Hard blocker to check before spending any run on this. At cutoff 0.015 x box on CAMELS (Rc = 0.375 Mpc/h, mean N = 2377) the Poisson expected degree is 0.034, so under a uniform field almost every galaxy would have zero neighbours. Real clustering at sub-Mpc scales raises this by an unknown factor. If the actual median degree is 0 or 1 then log(d+1) is near constant, min is degenerate, std is undefined for most nodes, and the whole apparatus does nothing. Plot the degree histogram first. This has not been measured on our data.

Two honest limits on the repair. The no-scalers arm is not perfectly count blind either: max over d samples grows with d by extreme value statistics and the sample std has a 1/d bias. And the density normalisation removes only the leading order count dependence, because Var(dtilde) = (1 + delta)/lambda_c + Var(delta), where the first term is shot noise scaling as 1/N, so a std aggregator on the normalised degree still sees N through the shot noise floor. State the fix as partial.

Free citation, worth quoting verbatim: their appendix I says the Set2Set readout "helps the most architectures without scalers as it can provide an alternative counting mechanism". The PNA authors themselves flag Set2Set as a counting channel. So do not use Set2Set or any attention or LSTM readout that can count on CAMELS, and cite this as an independent, mainstream-GNN corroboration of our pooling finding.

**Bessel radial basis on edge lengths (arXiv 2410.20516 appendix B, equations 6 and 7).**

Project each edge distance onto n = 64 radial Bessel functions: B_n(r, c) = sqrt(2/c) * sin(n pi r / c) / r, with B_n(0, c) = sqrt(2/c) * n pi / c, and radial cutoff c = 0.6 on Z-scored positions. The paper states in section 4 that this "was found to be crucial for downstream performance in the graph-level prediction task to predict sigma_8". About ten lines, no new symmetry machinery, works on our existing GNN edge features. Their sweep tried 32 and 64. Caveat: the paper never says whether Z-scoring is per point cloud or global, and for a periodic box that distinction either normalises away box size or does not, and any per-cloud normalisation can create or destroy a leak. Read the repo before copying the number 0.6.

**Three E(3)-invariant edge features (arXiv 2306.12327, Wu and Jespersen).**

dist = ||x_i - x_j||, cos1 = unit(x_i - centroid) . unit(x_j - centroid), cos2 = unit(x_i - centroid) . unit(x_i - x_j). This is the standard cosmic-graph edge set inherited from HaloGraphNet. Self-loop edge attributes must be set by hand to (dist, cos1, cos2) = (0, 1, 0), which is the geometrically correct limit; leaving them as zeros silently corrupts a run.

Caveat, and it is important: cos1 and cos2 are ill defined in a periodic box. The centroid of a statistically uniform periodic box is the box centre plus noise, so unit(x_i - centroid) degenerates into "direction from box centre", which encodes where a galaxy sits in the box frame rather than any physical structure, and it is not single valued under periodic wrapping. Their own code comment calls the centroid arbitrary. Port dist unchanged, and treat cos1 and cos2 as suspect for CosmoBench and ablate them.

The same defect, in a worse form, appears in arXiv 2405.13119, which replaces (x, y, z) with (d_ic, cos alpha, cos beta) where both cosines are built from the absolute position r_i measured from the box origin. Translate every halo by t and both cosines change, so only d_ic is genuinely translation invariant. INTERPRETED (this is the reader's diagnosis, not the authors'): this is a more likely cause of that paper's sigma_8 failure on positions alone than the hierarchy or the pooling that they blame. For us that is an opportunity, not just a criticism: feeding true minimum-image pair separations |r_j - r_i| plus the cosines of angles between neighbour pairs at halo i (which are genuinely E(3) invariant) is both correct and strictly more informative. If that recovers sigma_8 on positions alone, it is a publishable diagnosis of a published architecture. Not yet tested.

**Periodic radius graph construction (arXiv 2306.12327).**

scipy.spatial.KDTree(pos, leafsize=25, boxsize=box_size) then kd_tree.query_pairs(r=r_link, output_type='ndarray'). The boxsize argument makes neighbour finding wrap for free, removing the need for ghost padding. Note their own runs set periodic=False, so they did not actually use it. Linking length sweep grid that they already ran: D_link in [0.3, 0.5, 1, 2, 3, 5, 7.5, 10] Mpc, with the metric reported separately for low and high mass targets and for 3D versus 2D graphs. Their plotted axis spans 0.08 to 0.20 dex, so radius moves the metric by roughly a factor of two over the range. Copy the grid and the stratified reporting, not the numbers. They also flag that their own 5 Mpc came from preliminary tests and is untuned.

Reproduction warning for anyone porting their code: in the version read, EdgePointGNN.forward recomputes edge_index with torch_cluster.radius_graph(data.pos, r=D_link) rather than using the periodic KDTree edge_index stored on the Data object, and EdgePointLayer.message never receives edge_attr, instead recomputing a single squared distance from the first three columns of x. So the three invariant edge features and the periodic wrapping are built, saved, and then discarded at train time. Their reported numbers validate a non-periodic radius graph with one distance feature.

Free axis nobody has tested: arXiv 2405.13119 uses kNN (k = 32 or 64) and explicitly flags in its limitations that Villanueva-Domingo found a radius graph beats kNN for GNNs on this data, but they never tested it. That is a named, unexplored axis for our graph construction search, from the group that owns the data.

**Geometric affine module (arXiv 2405.13119, PointMLP).**

For centre point i with neighbours j: fhat_ij = alpha * (f_ij - f_i) / (sigma + 1e-5) + beta, where alpha and beta are learnable vectors of length m (the channel width) and sigma is a single scalar, sqrt(mean over all i, j and channels of (f_ij - f_i)^2), computed per tensor. Then f_i^{l+1} = Phi_post(max_j Phi_pre(fhat_ij)), where Phi_pre and Phi_post are Conv1d plus BatchNorm plus ReLU with a residual connection. About 10 lines. This is the piece that makes PointMLP work without any geometric kernels, and it is a genuinely different design point from both our DeepSets and our GNN.

### 2.3 Hierarchical and point cloud backbones

**PointNeXt set abstraction (arXiv 2211.12346, ETH / CosmoGrid).**

Build one set abstraction block and stack it four times: farthest point sample the cloud down by the stride, ball-query K = 32 neighbours per surviving centre, concatenate each neighbour's relative offset (p_j - p_i) with its features, run a shared MLP, then reduce over the K neighbours. Strides [4,4,4,4] on a 5000 point CAMELS-SAM cloud gives 5000 -> 1250 -> 313 -> 78 -> 20 -> global pool.

Their full hyperparameter table, as a starting configuration: radius 0.1 in normalised coordinates, strides [4,4,4,4], blocks [4,4,7,4], width 32, expansion 4, nsample (K) 32, learning rate 3e-4 with cosine schedule, Adam betas (0.9, 0.999), weight decay 1e-4, batch size 64, MSE loss. Anything unstated inherits from PointNeXt (Qian et al. 2022) defaults. First stage radius 0.1 of the box side and double per stage: 2.5 Mpc/h on CAMELS, 10 Mpc/h on CAMELS-SAM. That is a principled starting linking radius rather than our current hand chosen threshold.

Why this is interesting for our leak question specifically: the fixed-K ball query plus max reduce is count blind at the local level by construction. A ball holding 200 galaxies and a ball holding 40 both contribute exactly 32 neighbours, so density reaches the network through neighbour radii rather than through a count. That is a mechanically third option we have not tested. Testable prediction: PointNeXt with max should land near our mean pool number (0.660) rather than our sum pool number (0.802), and if it lands well above both we have found genuine geometric information.

Two implementation notes. Their pooling operator is never written down anywhere in the paper, so max reduction is an inference from the PointNeXt defaults they say they follow, not a stated fact. Verify against the PointNeXt source. And farthest point sampling has no fused Metal kernel and will dominate our runtime; our clouds are static, so precompute the FPS index hierarchy and ball-query neighbour lists once per cloud and cache them, or use voxel grid subsampling. Restricting augmentation to the periodic box symmetry group (90 degree rotations and axis flips) keeps cached neighbour lists valid up to an index permutation; arbitrary periodic translations do not.

Their config is a PointNeXt-L class model in the several million parameter range, roughly two orders of magnitude above our 67k GNN, and their heavy runs used 8 Tesla P100 GPUs. Scale it down, and do not read a scaled-down failure as a failure of the architecture.

**PointMLP stage depth (arXiv 2405.13119).**

MEASURED, Omega_m R^2 by number of stages: 3 -> 0.89, 4 -> 0.97, 5 -> 0.95, 6 -> 0.98. sigma_8: 0.75, 0.81, 0.85, 0.81. Big jump from 3 to 4, then flat. Bound hierarchical depth at 3 to 5 in the search space and spend budget elsewhere. All single run with no seed spread, so treat this as suggestive.

MEASURED, sampling: FPS versus random subsampling at 8192 points gives Omega_m 0.97 versus 0.95, sigma_8 0.81 versus 0.83. Single run each, so the equality is suggestive, not established, but random subsampling is free and FPS is O(N n) sequential, so test random first.

Training recipe that worked for them: SGD with Nesterov momentum 0.9, cosine annealing, batch size 32, 250 epochs, 1600 training clouds, with Optuna over exactly three hyperparameters (maximum learning rate, minimum learning rate, weight decay) for 100+ trials. Note they used SGD, not Adam, at roughly our data volume.

**Hierarchical graph pooling, if we try it at all.**

Do not build Top-K or SAGPool on this data. MEASURED reconstruction error against the adjacent-point baseline gamma (arXiv 2110.05292): Grid2d gamma = 7.812 versus TopK 18.86 +/- 3.923 and SAGPool 16.61 +/- 3.270; Ring gamma = 4.815 versus TopK 132.2 +/- 4.133 and SAGPool 148.5 +/- 30.10. Average rank across seven point clouds puts them last (TopK 6.14, SAGPool 6.57 of 8). Diagnosed cause: trainable sparse selection amputates whole regions of the graph, which for spatial clustering statistics over a periodic box is close to the worst possible failure mode.

Start instead from Graclus or NDP: non-trainable, sparse, roughly uniform subsampling, zero added parameters, and best on attribute preservation (NDP 1.86, NMF 2.14, Graclus 3.43 average rank). Graclus was also the single best method on ModelNet10, the only 3D point cloud classification task in their suite, at 83.9 +/- 1.9 against a no-pool baseline of 81.0 +/- 0.5. Being non-trainable means the coarsening hierarchy can be precomputed once per catalogue and cached to disk, which matters a lot on one laptop. MinCut is the cheapest trainable dense option, because its Sel is an MLP on node features only with an auxiliary normalised-cut objective that is unsupervised and therefore adds no pressure from the Omega_m label.

Fixed-K hierarchical pooling is also a structural leak screen. Fixed methods return the same K for every input, so a fixed-K stage erases N from the representation before the readout, and even a sum readout downstream cannot read the count directly. Swapping sum for mean only hides N at the last layer. That gives a clean 2x2: {fixed-K, adaptive-K} crossed with {sum, mean} readout, which localises where the count enters. Note that this points the opposite way from the survey's own guideline, which prefers adaptive methods because they preserve relative graph size. Their advice is written for tasks where size is signal; ours is a task where size is contamination.

If we do use hierarchical pooling with a learned selector, take Algorithm 3 from arXiv 2605.06250: compute a Laplacian positional encoding rho once from A, form H = concat(detach(X), rho), then S = softmax(H W), with RED as X_p = S^T X' and CON as A_p = S^T A S. The two non-obvious details are that the encoding is concatenated only into the selector and never message passed, and that the intermediate embedding is detached so a working pooling branch does not perturb the downstream regressor. Their hyperparameters: PE dimension 6, 200 hidden channels, AdamW, lr 1e-3, dropout 0.5. MEASURED warning from their own tables: a few message passing refinement steps help raw features but degrade positional encodings, with Laplacian PE on MUTAG dropping Q from 0.79 to 0.72 to 0.63 to 0.39 across zero to three steps. Keep a skip path that reaches the readout unrefined.

Serious leak warning on that block: a Laplacian positional encoding computed on a radius or kNN graph in a periodic box is a function of local density, and density is exactly what carries our count leak. Adding such an encoding could open a channel that mean pooling currently blocks. If we add one, measure its correlation with galaxy count first and use CAMELS-SAM as the control.

Their cheap pre-flight diagnostic, if we want it: discretise node features into colours by cosine similarity at threshold tau, spectral-cluster A for reference groups, then Q = min(Gamma, Lambda) where Gamma is the fraction of colours appearing in exactly one group and Lambda is the fraction of unseen graphs whose group colour sets are subsets of some seen graph's. Their "Random: same" row, which is the featureless case and therefore ours, scores Q = 0.00 at zero refinement steps. INTERPRETED: their framework predicts community-based hierarchical pooling on a featureless point cloud fails outright unless we inject positional encodings first. That is a cheap falsifiable prediction. Caveat: their graphs average about 18 nodes, their transferability measure loops over all seen and unseen graph pairs, so Q will be far more expensive at N = 5000, and every Q value in their tables is a single number with no uncertainty.

### 2.4 Heads, losses and hybrid blocks

**2PCF as a global context vector (arXiv 2410.20516 section 5.3).**

Concatenate the 24 dimensional 2PCF to the pooled graph embedding after the final message passing layer, before the readout MLP. MEASURED: SEGNN alone 2.37 -> 1.66 on Omega_m MSE, while 2PCF alone was 2.03, so the hybrid beats both parents. The scale ablation is the useful part: only the large scale bins (r > 80 Mpc/h) recover Omega_m, only the small scale bins (r < 30 Mpc/h) help sigma_8, and the full vector beats both splits, meaning mid-range scales carry information neither end has. Caveat: those thresholds are set by Quijote's roughly 1 Gpc boxes and are meaningless inside a CAMELS 25 Mpc/h box. Re-derive the split per suite.

The same hybrid is named but never built in arXiv 2405.13119. We already have point_clouds/tpcf.py at CAMELS Omega_m 0.860 and point_clouds/gnn.py at 0.802 sum pool, so this is an afternoon and is the cheapest plausible way to beat both. It belongs in the search palette as a block, not as a separate model.

**Additive skip head (arXiv 2306.12327).**

return self.fc(x) + self.galaxy_halo_mlp(data.x), where the second branch is an MLP on raw per-node features only, so the graph branch only learns the correction. Our analogue: an explicit head on a cheap summary (galaxy count, or the 2PCF bins) added to the GNN head, so we can zero one branch at inference and measure how much each contributes. INTERPRETED: this is the single most valuable item from that paper for us, because it turns the leak question from a pair of contrasting runs into a decomposition measurable inside one model.

**Moment network loss, for calibrated error bars.**

Two variants exist, both cheap. Wu and Jespersen: predict y_hat and Sigma_hat, loss = ||y_hat - y||^2 + ||Sigma_hat - (y_hat - y)^2||^2, taking the logarithm of each term before summing for stability, output layer 2 x n_out, so 4 outputs for (Omega_m, sigma_8). Chatterjee and Villaescusa-Navarro, their equation 12: L = sum_i log(sum_batch (theta - mu)^2) + sum_i log(sum_batch ((theta - mu)^2 - sigma^2)^2), then chi2 = mean((theta - mu)^2 / sigma^2) should be 1.

Why this matters beyond publication convention: a search objective of pure R^2 is exactly what will chase the count leak, whereas an objective that also demands calibrated uncertainty is harder to game. Worth testing as a leak-resistant search objective. Not yet tested.

The Fishnets loss is the third option and is the tightest if we set n_p = 2 at the final readout so the aggregate output is (Omega_m, sigma_8): L = 0.5 (theta - theta_hat)^T F (theta - theta_hat) - 0.5 ln det F. F_total is then literally the predicted inverse covariance of our two parameters, per simulation. No CosmoBench baseline we have reproduced reports per-box uncertainty.

**Do not spend budget on exotic losses beyond that.** MEASURED, arXiv 2211.12346: they tested plain MSE on the parameters, a likelihood loss that also predicts the covariance, and a mutual-information-maximising summary, and found the differences negligible in both MSE and posterior width. That is direct evidence to keep plain MSE plus one uncertainty head and spend our 5.5 minutes per model elsewhere.

**Per-point features by concatenation.** The input is a per-point vector, so (x, y, z) becomes (x, y, z, f). CosmoBench carries velocities. Two rules. First, from arXiv 2410.20516: velocities fed as ordinary node features scored 1.13 on Omega_m MSE, worse than a plain GNN at 1.10, while the same model with velocities as the steerable conditioning attribute scored 0.84. Bolting a vector on as scalars wastes it and can actively hurt. Second, on halo or stellar mass, see the trap in section 4.

---

## 3. How to run the search

### What is already taken

Four of the six directly-read entries close specific openings. Recorded from the human's earlier reading.

- **BioArc (arXiv 2512.00283, ICML 2026).** Heterogeneous NAS with a weight-sharing supernet plus an agentic architecture predictor. Its own appendix reports that supernet ranking correlates only rho = 0.32 with train-from-scratch ground truth unless each path is finetuned individually, rising to rho = 0.73 when it is. INTERPRETED, and this is the practical takeaway: a weight-sharing supernet ranking is close to uninformative without per-path finetuning, and at 5.5 minutes per model on one laptop we can afford to train candidates from scratch, so the supernet shortcut buys us little and costs us rank fidelity. Use it as a citation for why we train from scratch rather than as a method to adopt.
- **SARA (arXiv 2608.00316, Meta, August 2026).** An LLM agent drives a Bayesian optimisation loop and can reconfigure bounds, acquisition function and even the objective mid-run. MEASURED, as reported: on hyperparameter benchmarks the final performance is similar to classical BO, and the gain is in early evaluations. INTERPRETED: the value is sample efficiency under a tight budget, which is exactly our regime, but the claim "LLM agent drives the optimiser" is taken.
- **AgentNAS (arXiv 2607.07984, July 2026).** An LLM designs a slotted search space, then conventional NAS fills the slots. This closes the "LLM proposes architectures" opening.
- **Multi-agent Architecture Search via Agentic Supernet (arXiv 2502.04180, February 2025).** Combines agents with weight-sharing supernets, which closes the opening BioArc's own future work named.

INTERPRETED, taken together: the generic agentic-NAS lane is crowded and closing fast. Our differentiator cannot be the search machinery. It has to be the search space (pooling and aggregation as first class axes rather than nuisance hyperparameters), the objective (leak resistant rather than raw R^2), and the finding (that an unconstrained search converges on the artifact).

### A ready-made search space to adopt rather than invent

From arXiv 2410.20516, appendix C, Tables 2 and 3.

Fixed: d_hidden 128, 3 layer MLPs, mlp_readout_widths (4, 2, 2), residual on, GELU scalar activation, sigmoid gate activation, integral-normalised spherical harmonics.

Swept: lr {1e-3, 1e-4, 1e-5}, weight decay {1e-4, 1e-5}, n_radial_basis {32, 64}, k {5, 10}, message_passing_agg {sum, mean, max}, readout_agg {sum, mean, max}, message_passing_steps {2, 4, 6}. PointNet++ only: n_downsamples {2, 3, 4}, downsampling_factor {2, 5, 10}, k_downsample {10, 20}, r_downsample {0.05, 0.2}, combine_hierarchies {mean, concat}.

MEASURED, and this is the opening: they swept message_passing_agg and readout_agg independently over {sum, mean, max}, but the paper never reports which won for any model, and section 4 hard-sets message passing aggregation to mean without discussion. INTERPRETED: the pooling axis is sitting in everyone's grid search as a nuisance hyperparameter while being, on our measurement, the thing that decides whether a model exploits a resolution artifact. That gap is ours to fill.

### Protocol rules for the search itself

- **Parameter matching is mandatory for any pooling claim.** Janossy's Table 8 holds trainable parameters fixed across k (3061, 3061, 3031 for k = 1, 2, 3) by setting the embedding dimension to floor(100/k), and separately reports the unmatched version. QUANN's ablation set does the same job in four arms: (1) psi = identity, which is plain mean pooling; (2) psi = identity with phi and rho grown to match parameter count and depth; (3) the 1/n normalisation removed, which is literally our sum pooling; (4) full NKM. Arm 2 is the one that separates "learnable pooling helps" from "more parameters help", a confound our current 67k GNN comparison carries.
- **Pooling isolation with a frozen encoder** (QUANN RQ2b): train an encoder once, freeze it, then swap only the pooling and retrain the rest. MEASURED in their MNIST results: QUANN-1 sum-task MSE 4.07 -> 3.07 frozen, DeepSets 5.92 -> 15.24. This is a clean way to test whether our sum versus mean gap is a property of the pooling or of the encoder adapting to it, which our current design cannot separate.
- **Count and report the number of test set evaluations.** Search on a validation split, touch the test split once, and print the count. It is a cheap credibility signal and nobody in the NAS literature reports it. Budget for multiple comparisons as well: Kapoor and Narayanan explicitly decline to correct for them and note that correction would weaken their result further, and with hundreds of candidates that is our problem too.
- **Repeat and report spread, always.** MEASURED evidence that this is not pedantry: in arXiv 2605.06250, DiffPool on Mutagenicity is 54.99 +/- 4.61 without encodings and 79.95 +/- 1.23 with node2vec, over 5 seeds. In arXiv 2410.20516, the GNN sigma_8 baseline is 4.84 +/- 2.90, a spread wider than its gap to most competitors, and several entries print +/- 0.00, which by our own standing rule reads as no result rather than a tight one. In arXiv 2405.13119, every ablation is a single run with no seed spread, and differences of 0.02 to 0.03 in R^2 are interpreted.
- **Ensembling is affordable at our runtime.** Fishnets ensembles 10 networks with different initialisations and combines by weighted average. At 5.5 minutes per model that is under an hour and it satisfies the no-single-run rule at the same time.
- **Training and validation protocol worth copying** (arXiv 2410.20516): 5000 steps, AdamW, cosine decay, 5-fold cross validation on a 2048 train / 512 validation / 512 test split, checkpoint at lowest validation loss, results reported as mean and spread across folds.
- **Learning curve as a diagnostic, not just a plot.** Their Figure 3 protocol plots test loss against training set size from 10^1 to 10^4. INTERPRETED: that curve is exactly what would show whether our sum pool advantage on CAMELS is a real inductive bias or a shortcut that only appears once there is enough data to fit the count.

### Cost budgeting on one M5 Pro, 24 GB, MPS

- MEASURED, ours: about 5.5 minutes per model.
- MEASURED, ours: NKM pooling costs about 5.8x a mean pool training step at batch 32, 5000 points, latent 64. Plan roughly half an hour for a pooling-dominated model.
- ESTIMATED: a group-wise multi-view scheme at phi = 6 views means 6 forward passes per sample, so roughly 20 to 35 minutes per model.
- ESTIMATED: exact k = 2 Janossy on CAMELS is affordable; CAMELS-SAM needs subsampled pairs.
- Dense hierarchical pooling carries an N by K matrix plus A' = S^T A S, running from O(1) for sparse-and-fixed to O(N^2) for dense-and-adaptive. Their 2021 measurement was that sparse methods pool graphs roughly four times larger than dense ones before running out of memory. Do not retype their absolute memory figures for a 24 GB MPS machine.
- Periodic neighbour search by box replication takes 8192 halos to roughly 80,000 points (arXiv 2405.13119). An 8192 point exact kNN is an 8192^2 distance matrix, about 268 MB in fp32 per cloud, so batch 32 needs chunking or a spatial grid.
- Cholesky solves and e3nn tensor products on MPS have not been checked for support or speed. That is the main reason to try attention readout and the Bessel basis first: both are architecture agnostic and cost nothing extra.

---

## 4. Validity and leakage

### Reporting layout to adopt

Steal the Kapoor and Narayanan table shape (their Figure 1 and Table A4), because referees already recognise it. Rows: Reported (sum pool, as-published protocol) / Reproduced / Corrected (mean pool) / Corrected (fixed-N resampling) / Count-only baseline / LLS-49 / 2PCF + MLP. Models as columns. Two independent corrections landing on the same number is a far stronger causal argument than one.

Their sharpest single criticism of the papers they redo is "weak baseline": in the Kaufman case the authors compared against a model that always predicts peace, while a baseline predicting last year's outcome scored 97.5 percent and beat every ML model. Our count-only 0.506 on CAMELS Omega_m is the exact analogue. Put it in every table, and add a stronger trivial row as well, for example count plus mean nearest-neighbour distance, since a two-number statistic is the honest trivial ceiling.

### Corrections and controls to run

- **Fixed-N resampling.** Subsample every CAMELS box down to a common N before the model sees it. This removes the leak channel from the data rather than from the architecture, so sum pooling and mean pooling should converge. If they do not converge, our mechanism story is wrong and we need to know that. Cost is a dataloader flag plus a few hours.
- **The twin-dataset rule, as a standing diagnostic.** CAMELS has N varying, so the channel is open. CAMELS-SAM has N fixed at 5000, so it is closed by construction. Rule for the whole search: any architecture or block that improves CAMELS but shows no effect on CAMELS-SAM is presumed to be exploiting the leak until shown otherwise. Kapoor and Narayanan have no such control anywhere, because their only correction tool is deletion. A naturally leak-free twin of the same task is methodologically stronger, and worth saying so explicitly.
- **Domain precedent for the fixed-N control.** arXiv 2211.12346 states outright that the halo count in the volume is cosmology dependent and therefore fixes the number of points. Their execution is worse than ours can be: they discard whole cosmologies that have too few halos, and they note those are the low sigma_8 ones, so their sigma_8 prior is silently truncated. Subsampling every box to a fixed N while keeping all cosmologies is strictly cleaner and is a small defensible novelty point.
- **Feature drift, at the representation level (arXiv 2307.09788, GCL).** For a model producing a global embedding z(X), define drift(p) = ||z(subsample(X, p)) - z(X)||_2 averaged over simulations and subsample seeds, for p in {1.0, 0.8, 0.6, 0.4, 0.2}, and plot drift against p for each pooling choice. This measures how much of the embedding is count rather than geometry, which the headline R^2 cannot see. Cost is one forward pass per (simulation, p, seed), no retraining. Write the expected curve first: sum pool drift should grow roughly linearly in (1 - p); mean pool drift should be near zero for a pure DeepSets but NOT for a radius-graph GNN.
- **The radius-graph GNN mean pool result may still leak, and drift is the test.** Our claim is that mean pooling denies access to the count. That holds for a pooling operator applied to count-independent node features. In a radius graph the node features are already density dependent, because node degree inside a fixed radius is a direct count proxy, so the count reaches the head through the messages regardless of pooling. Our mean pool GNN at 0.660 sits above the count-only baseline of 0.506, which is consistent with either real geometry or routed count. Concrete cheap experiment: retrain the mean pool GNN with degree-normalised message aggregation and no degree in the node features, and see whether 0.660 falls toward 0.506 or holds. Three seeds, report a spread.
- **A derived algebraic fact worth checking in our own code.** For a linear pool, L2 normalising the pooled vector removes the count exactly: sum = N * mean, so sum / ||sum|| = (N * mean) / (N * ||mean||) = mean / ||mean||. Sum pool followed by L2 normalisation is algebraically identical to mean pool followed by L2 normalisation. GCL normalises all features onto the unit sphere after the final layer and never comments on this. Two consequences: if we want the count provably gone from a DeepSets-style pool, normalise rather than train for invariance; and our sum versus mean contrast is only a clean contrast because we do not normalise, which must be stated explicitly in the paper.
- **XGBoost control, the point cloud analogue of our count-only baseline** (arXiv 2405.13119). Feed a gradient boosted tree the same per-halo features with no neighbour information at all. MEASURED at 8192 halos with positions plus mass plus velocity: XGBoost Omega_m 0.83 / sigma_8 0.76 against the network's 0.97 / 0.95. Costs minutes and separates "the model learned geometry" from "the model learned the feature distribution".
- **Hand-made environment control before claiming the GNN learned environment** (arXiv 2306.12327). Per node, overdensity_i = log10(sum of neighbour halo masses within r_link), fed to a random forest alongside the other per-node features. Add the analogous per-node local count or local density feature to the LLS and MLP baselines before claiming the GNN learned something a summary statistic cannot.
- **Counting-sensitivity control tasks.** Colors-3 and Triangles (Knyazev et al. 2019, arXiv 1905.02850) are tiny synthetic graph datasets built to probe generalisation to larger graphs than seen in training. MEASURED in arXiv 2110.05292 Table 6, Colors-3: MinCut 60.1 +/- 4.0 and DiffPool 55.2 +/- 1.5 against no-pool 40.8 +/- 2.1 and NDP 25.4 +/- 1.8. INTERPRETED: if an architecture we designed for cosmology also spikes on these, that is independent evidence it has learned a size-reading circuit rather than a geometry-reading one. CPU cheap.
- **Coordinate reconstruction autoencoder, as a label-free geometry check** (arXiv 2110.05292 appendix A.3). MLP -> GNN -> Pool -> UpScale -> GNN -> MLP, trained on MSE against the input coordinates, where UpScale is U X' with U the transposed pseudo-inverse of S, and where the output GNN is fed the original adjacency A so that Red is isolated from Con. Their reference baseline gamma is the mean squared distance between adjacent nodes, and any operator whose MSE exceeds gamma cannot on average place a reconstructed point nearer its true position than to its neighbour. For us this measures whether a pooling stage keeps the galaxy geometry while never touching Omega_m, so it is a leak-free architecture check. Note their version fits one autoencoder per point cloud, which measures compression capacity and not generalisation; if we adopt it we must decide deliberately whether we want a train and test version, which is untested by them.

### Calibration, uncertainty and split hygiene

- **Bootstrap and paired tests.** Kapoor and Narayanan Finding 2: 9 of 12 papers reported no confidence intervals or significance tests, and they show a reported AUC of 0.85 with a bootstrapped 95 percent interval of [0.66, 0.95], which destroys the paper's ranking claim. Implement: bootstrap over test boxes for a 95 percent interval on R^2, and a paired test between architectures on the same test boxes.
- **PIT plus KS calibration test** (Fishnets section 4.3). Take the predicted Gaussian N(theta_hat, F_total^{-1}), compute the probability integral transform of the true parameter under it for every test box, and KS test the results for uniformity. They report p = 0.628 and 0.233 for their two parameters. Alternative from arXiv 2211.12346: expected coverage probability (Hermans et al. 2021), fitting a three-component Gaussian mixture density estimator on the two outputs and checking whether the nominal credible interval contains the truth at the nominal rate.
- **Split hygiene audit on CosmoBench, against L3.2 non-independence.** Concrete checks: does the same initial-conditions seed, the same cosmology pair, or multiple snapshots or sub-boxes of one simulation appear on both sides of the split? For the merger tree track this is acute, since many trees drawn from one box share the same large-scale modes. If CosmoBench splits by box this is fine, but it needs verifying rather than assuming. Not yet checked.
- **Contiguous spatial cross validation, if we ever cut boxes into patches** (arXiv 2306.12327). They split into 6^3 = 216 subvolumes and make each validation fold a block of 36 consecutive subvolumes, a full slab. Randomly shuffling subvolumes leaks large-scale structure across the boundary because neighbouring subvolumes share the same filaments.

### Traps that would create new leaks

- **Do not use Set2Set or any counting readout on CAMELS.** The PNA authors themselves flag it as an alternative counting mechanism.
- **Do not feed per-point halo or stellar mass on CAMELS without checking first.** In CAMELS a halo is recorded only once it holds about 20 particles and particle mass scales with Omega_m, so the low-mass cutoff of the mass catalogue is a more direct readout of Omega_m than the count is. This is the same mechanism we measured, entering through a different channel. MEASURED context, arXiv 2405.13119: adding halo mass moves sigma_8 from R^2 0.01 to 0.83 and Omega_m from 0.80 to 0.89, and they attribute the sigma_8 jump to the halo mass function, which is physics and fine, but the Omega_m gain is not cleanly attributable to structure and they never check. INTERPRETED, not measured by us: their positions-plus-mass numbers are not a target for us to chase, and this is a second concrete example for the leak paper. If we test mass at all, test it on CAMELS-SAM first and check the CAMELS minimum recorded mass against Omega_m before drawing any conclusion.
- **Do not read a large MSE or R^2 improvement as an information gain without checking outliers.** MEASURED, arXiv 2211.12346: adding mass dropped MSE by more than a factor of two, but they report the gain was mainly a reduction in outliers, and the (Omega_m, sigma_8) posterior contour was not strongly affected. R^2 is outlier dominated, so this is a live risk for our own leaderboard.
- **Periodic boundaries in every neighbour operation.** The ball query and the relative offset (p_j - p_i) must both use the minimum-image convention. Otherwise every point near a face gets a truncated, artificially low-density neighbourhood and the model can read distance-to-boundary as a spurious feature. arXiv 2211.12346 did not need this because their data is a lightcone patch with open boundaries; we do.
- **Train-set statistics stay train-set statistics.** PNA's delta, any Z-scoring constant, and any density normalisation constant must be computed on the train split and reused unchanged. Per-split or per-batch computation is a quiet L1.2 leak.
- **Augmentation design, if we adopt any invariance loss.** GCL enforces density invariance by a loss, not by construction, and the paper itself concedes that complete density invariance is unattainable. Two consequences. First, mean pooling is a hard algebraic invariance and is strictly stronger, so a GCL-style loss cannot be the answer to a leak mean pooling already blocks. Second, and more serious: for LiDAR, density is pure nuisance, but our galaxy count is part real cosmology and part artifact, and count alone already reaches 0.506, so an invariance loss cannot separate the two channels and would discard real signal along with the artifact. If we use it at all, the faithful augmentation is a threshold shift (drop the lowest-mass galaxies), not random thinning, because the artifact is a mass threshold effect. Random subsampling is a crude stand-in and invariance to it may not transfer.
- **The augmentation control a referee will demand.** GCL anticipated the objection that their scheme just sees more data and scaled the baseline's training clouds from 1x to 20x. MEASURED: recall moved 24.0 to 28.8 while GCL reached 72.3. Our equivalent: train the plain model with the same number of subsample-augmented examples and the same wall clock, with no group loss. One extra run.
- **If we use a group-wise scheme, the group size matters.** MEASURED, GCL Table 4: phi = 2 gives 46.3, phi = 4 gives 68.8, phi = 6 gives 72.3, phi = 8 gives 71.4, phi = 10 gives 69.7. A cheap two-view version will most likely fail to reproduce the effect. Their other two non-obvious details: do not stop-gradient the anchor (blocking it drops recall from 72.1 to 50.1), and the variance loss alone is worse than the plain pair loss (64.4 versus 70.3), so only the anchored combination wins.

### An objective that resists the leak

INTERPRETED, and this is a design proposal rather than a measured result: extend the Kapoor and Narayanan model info sheet for featureless geometric data. Their Q21 asks the researcher to argue that each feature is legitimate. Our replacement: enumerate every global permutation-invariant functional the architecture can compute from raw coordinates (N, centroid, second moments, N-normalised densities, k-NN distance distributions) and argue the legitimacy of each. Then use it as a search space constraint, not a post hoc checklist, so illegitimate functionals are unreachable by construction. Note their own honest limit: the info sheet is a reporting instrument, not a detector, and they say plainly it cannot be verified without computational reproducibility. Shipping one is a credibility artifact, not a result.

Also worth taking from them: their D1 / R1 recommendation to lower-bound the Bayes error for a task, so that once achievable accuracy is reached effort stops and claims beating the bound get extra scepticism. That is a ready-made justification for spending compute on estimating the irreducible-noise ceiling instead of only chasing R^2.

---

## 5. Domain precedents

Numbers others got on this or similar data. Read this section as calibration, not as targets, because almost none of them are directly comparable to ours.

### arXiv 2410.20516, Cosmic-scale benchmark for symmetry-preserving data processing (MIT / IAIFI, NeurIPS 2024 NeurReps workshop)

Quijote, 5000 most massive halos per box, so N is fixed by construction and our leak cannot exist in their data.

Derived conversion, verified by command: Quijote Sobol draws Omega_m uniform on [0.10, 0.50] and sigma_8 uniform on [0.60, 1.00], both of width 0.4, so the prior variance is 0.4^2 / 12 = 13.333e-3 for both, and every MSE in their Table 1 converts as R^2 = 1 - MSE / 13.333e-3. Their EGNN entry of 13.33e-3 lands on 0.000 to three decimals, which both confirms the conversion and identifies that failure as mean collapse rather than a tuning problem.

Converted to our scoreboard, Omega_m: 2PCF 0.848, GNN 0.792, SEGNN lmax=1 0.827, best model (SEGNN lmax=2 with steerable velocities) 0.937. Their 2PCF at 0.848 sits beside our reproduced 0.860 on CAMELS and 0.778 on CAMELS-SAM, which is a useful independent anchor that our 2PCF pipeline is in the right place.

Measured do-not-build list, which saves runs: EGNN 13.33 / 13.37, a total collapse to the training mean. PointNet++ on sigma_8 at 9.00, R^2 = 0.325, worse than every message passing model. Local attention in message passing and invariant attention both worse than the plain baseline. And lmax=1 matched or beat lmax=2 on positions only (2.31 / 2.34 versus 2.37 / 2.36) at nearly twice the speed (4.42 versus 2.40 iterations per second), so if we ever port a steerable model, use lmax=1.

Their diagnosis of where GNNs lose: Omega_m lives in long range correlations, message passing over a kNN graph with k of 5 or 10 cannot reach that far, and so a 24-bin 2PCF fed to a small MLP beats every network they tried on Omega_m.

Caveats. Four A100s, 14 to 44 minutes per run. JAX code (eqnn-jax). Several Table 1 entries print +/- 0.00, and the GNN sigma_8 baseline is 4.84 +/- 2.90, so the equivalence headline is Plausible, not established. It is a workshop paper rather than an archival one, so it is cheap to cite and cheap for a referee to say we merely extended it. Their own closing sentence calls for "specialized architectures tailored to cosmology data, which would be sensitive to the local gravitational clustering environment as well as the long-range correlations", which is a citable statement that the field asked for the thing we are building.

### arXiv 2405.13119, Cosmology from point clouds with dark matter halos from Quijote (Chatterjee and Villaescusa-Navarro, 2024)

PointMLP-elite on Quijote halos, 1000 Mpc/h boxes, 2000 simulations, 5 varied parameters, 80/10/10 split.

MEASURED, positions only, which is our actual protocol: their network gets Omega_m R^2 0.80 and sigma_8 R^2 0.01, while their own 2PCF + MLP gets 0.78 and 0.46. The simple statistic wins on sigma_8 by a mile and ties on Omega_m. They say so plainly, and note that Makinen et al. found the same with a more expressive GNN on the same data. INTERPRETED: this is a third independent instance of the hand-built summary beating the deep model on halo point clouds.

MEASURED, with extra features at 8192 halos: positions plus mass plus velocity gives Omega_m 0.97 and sigma_8 0.95. See the mass trap in section 4 before treating that as a target.

MEASURED, pooling ablation under fixed N: max versus mean versus sum gives Omega_m 0.97 / 0.94 / 0.95 and sigma_8 0.81 / 0.79 / 0.83, and they conclude no significant change. INTERPRETED: this is a second group, a different suite, and a different architecture reproducing our measured CAMELS-SAM null. That is exactly the external corroboration our mechanism claim needs.

But cite the direction, not the digits. Their Table 2 is internally inconsistent. The caption states all ablation rows use 8192 halos, 32 neighbours, positions plus mass plus velocity. The 4-layer baseline row reads Omega_m (7.1 percent, R^2 0.97, MSE 4.6e-4), which matches Table 1's positions plus mass plus velocity row exactly, but its sigma_8 reads (5.2 percent, 0.81, 2.6e-3), which matches Table 1's positions-plus-mass row (5.1 percent, 0.83, 2.5e-3) rather than the positions plus mass plus velocity row (2.6 percent, 0.95, 6.7e-4). Separately, the random subsampling row reports MSE 4.2e-3 alongside R^2 0.95, an order of magnitude off its FPS counterpart at 4.6e-4 and almost certainly a typo for 4.2e-4. And every ablation is a single run with no seed spread. Our CAMELS-SAM null is the better evidenced of the two.

Their 2PCF minimum-scale ablation is a usable diagnostic ruler in its own right (appendix A). Recompute xi(r) with r_min raised and retrain the MLP. MEASURED at 8192 halos: all scales gives sigma_8 R^2 0.46; cutting everything below 10 Mpc/h gives 0.00. At 4096 points, 0.43 -> -0.02. Omega_m survives, sigma_8 does not. We already have point_clouds/tpcf.py with configurable binning, so this is a few lines, and it tells us which scales any candidate architecture is actually using: if a network matches the r > 10 Mpc/h 2PCF but not the all-scales one, it is blind to small scales.

### arXiv 2211.12346, Cosmology from galaxy redshift surveys with PointNet (ETH Zurich, NeurIPS ML4PS 2022)

PointNeXt on CosmoGridV1, a 1000 square degree lightcone patch over z in [0.4, 1], dark matter only, halos as a stand-in for galaxies, in real space with open boundaries.

MEASURED: PointNeXt on positions beats a Landy-Szalay 2PCF baseline by 1.4x to 2.3x in MSE at matched point counts, with a further 2.2x to 3.6x when halo mass is added as a per-point feature. The advantage over 2PCF is largest at the fewest points (2.31x at 8000, 1.38x at 32000). INTERPRETED, and it points our way: our clouds are 5000 points or fewer, so we sit in the regime where their measured gap is widest. They also state explicitly that they did no hyperparameter or architecture search, so the recipe is a floor rather than a ceiling.

Their metric is a single MSE pooled over both parameters on an unstated normalisation, so no number here converts to our R^2. There are no error bars anywhere in their Table 1 and the number of runs is never stated, so treat every value as single run.

Derived warning about their scaling claim, which they do not make themselves: fitting MSE against N as a power law over their three points gives an exponent of 0.735 for PointNeXt on positions and 1.103 for the 2PCF, meaning the 2PCF improves faster with point count and a naive extrapolation has it catching up near 7.8e4 points. Three points, illustrative only. This is good news for us (we are far below any crossover) but it is a reason not to cite their result as evidence that point clouds beat two-point statistics at survey scale.

Baseline fairness rule worth copying: compute the 2PCF on exactly the same points fed to the network, not on the full catalogue. Their setup is Landy-Szalay with 2.56e5 randoms and 50 linearly spaced bins from 0 to 300 Mpc. In a periodic box we do not need randoms at all, since RR is analytic, but the matched-N rule must hold or our 0.860 is not comparable to any network number.

### arXiv 2306.12327, Learning the galaxy-environment connection with GNNs (Wu and Jespersen, ICML 2023)

Different task (per-node stellar mass regression on Illustris TNG300, with halo mass and maximum circular velocity given as node features), so the scientific result does not transfer. It is here because it is the direct methodological ancestor of the CAMELS cosmic-graph GNNs we are competing with, so matching its graph construction is table stakes.

MEASURED: 0.129 dex with self-loops, 0.145 dex with self-loops removed so a node sees only its neighbours, and 0.148 dex for a random forest on halo mass and maximum circular velocity alone. INTERPRETED by them: environment matters. Caveat on that ablation: x_i is concatenated into every message, so a node with any neighbour still sees its own features even with loops removed, and the isolation is therefore not clean.

Do not cite their pooling result as support for ours. Their statement that max beats sum by 0.001 dex, and that concatenating sum, max, mean and variance gives no significant gain, is about neighbourhood aggregation inside message passing, not about the global readout. It is at the wrong level and a referee will catch that.

Domain transfer result worth pre-registering as an expectation: TNG300 to TNG50 gives worse than 0.2 dex, but training on a mixture of both recovers roughly 0.13 and 0.14 dex on each. If we try CAMELS to CAMELS-SAM transfer, expect it to fail outright, and expect mixed-suite training to be the fix.

Optimisation recipe that fits our budget: AdamW, lr 1e-2, weight decay 1e-4, batch of 9 graphs, 1000 epochs, with both lr and weight decay divided by 5 at 50 percent and by 25 at 75 percent of training, quoted at 10 minutes on one A10G. Caution: they anneal by constructing a brand new AdamW, which throws away the momentum state. LayerNorm throughout, never BatchNorm, which matters for small graph batches on MPS.

### arXiv 2310.03812, Fishnets, parameter efficiency numbers

MEASURED on ogbn-proteins: GCN-112 at 1,887,144 parameters scores 0.8425 +/- 0.0018, fishnets-16 at 280,740 parameters scores 0.8444 +/- 0.0018, fishnets-8 at 146,596 parameters scores 0.8410 +/- 0.0013. Roughly 15 percent of the parameters for the same score, which is exactly our regime at 67k.

MEASURED under covariate shift: trained at n = 500 and tested at n = 850 with shifted noise and covariate distributions, fishnets 0.007 +/- 0.017 MSE on the slope, mean-deepsets 0.120 +/- 0.178, learned-softmax 0.042 +/- 0.069, at 10,855 parameters versus 87,810.

Honest discount on their benchmark evidence: ogb-arxiv got slightly worse (0.7100 GCN versus 0.7062 fishnets), ogbn-proteins is a tie inside the quoted error bars, and only ogb-molhiv (0.76 to 0.80) is a clear accuracy win. The defensible headline is parameter efficiency and robustness under distribution shift, not accuracy. Also, their error bars are the standard deviation of the metric over the last ten training epochs, which is training noise rather than seed-to-seed variation, so it understates the real spread.

### Cross-suite transfer tests these precedents license

- Train Fishnets on CAMELS (variable N) and evaluate on CAMELS-SAM (fixed N = 5000) with zero retraining, and the reverse. Sum pooling cannot survive this by construction; mean pooling can; Fishnets should. That turns our CAMELS-SAM null into a positive transfer claim. Their own scaling demonstration is training at n = 500 and evaluating at n = 10^4, a 20-fold change, with no retraining and no accuracy loss.
- Train on full CAMELS catalogues and test on random 25, 50 and 75 percent subsamples of the same boxes. A model riding the count artifact should fall apart because N changed while cosmology did not. A model that learned clustering should degrade gently. That is a leak detector, not just a robustness check.

---

## 6. Closing table

| Paper | Relevance | What we take from it |
|---|---|---|
| CosmoBench, 2507.03707 | Foundational | The data, the splits and the baselines. Its own headline (49-parameter fit beats a 671k GNN) sets the bar we have to explain or beat. |
| Kapoor and Narayanan, 2207.07048 | HIGH | The L1 / L2 / L3 vocabulary, the Reported versus Corrected versus Stronger Baseline table layout, and a demonstrable hole in their framework (architecture-induced L2) that our finding fills. |
| Contardo et al., 2503.22654 | HIGH | The physical explanation of Omega_m predictability that never mentions resolution or selection, which is the gap our artifact explanation sits in. |
| PNA, 2004.05718 | CRITICAL | The std aggregator (cheapest shot at sigma_8), the towers trick for cost, the density-normalised degree scaler as a leak-free repair, and a verbatim warning that Set2Set is a counting readout. |
| Fishnets, 2310.03812 | CRITICAL | A ~15 line inverse-variance pooling that splits shape (F^{-1}t) from count (log det F) into two separately ablatable tensors, plus a loss giving calibrated per-box error bars. |
| QUANN / NKM, 2602.04941 | CRITICAL | A learnable pooling that is provably count-blind (O(n) error on sum-decomposable targets), a 4-arm ablation whose third arm is our sum pooling, and a verified trap in their released code. |
| Cosmic-scale symmetry benchmark, 2410.20516 | CRITICAL | Attention readout (leak-safe and beats mean), the Bessel radial edge basis, 2PCF as a global context vector, sqrt(N) pooling, a ready-made search space, and a measured do-not-build list. |
| Janossy pooling, 1811.01900 | HIGH | k = 2 pooling as a learnable 2PCF, the point-process framing where k is the correlation order, and a way to separate pooling capacity from count access as two orthogonal knobs. |
| Understanding pooling in GNNs, 2110.05292 | HIGH | The select-reduce-connect vocabulary, the alpha normalisation dose-response sweep, the reconstruction diagnostic, a do-not-build list (TopK, SAGPool), and proof that the leak framing is unoccupied. |
| PointNet cosmology, 2211.12346 | HIGH | A complete copyable set-abstraction hyperparameter table, fixed-K ball query as a locally count-blind third option, and independent domain precedent for the fixed-N control. |
| PointMLP on Quijote, 2405.13119 | HIGH | The geometric affine module, the 4-stage depth knee, the 2PCF minimum-scale diagnostic ruler, the XGBoost no-neighbour control, and an independent pooling null under fixed N. |
| Wu and Jespersen halo-GNNs, 2306.12327 | HIGH | Periodic KDTree radius graph in one line, the invariant edge feature set and its periodic-box caveat, contiguous slab cross validation, and the additive skip head that turns the leak into a decomposition. |
| Node features in graph pooling, 2605.06250 | MEDIUM | Algorithm 3 (Laplacian PE into the selector, detached), the Q pre-flight diagnostic, and a prior-art boundary to draw explicitly. Terminology collision: their pooling is coarsening, not readout. |
| GCL density-invariant features, 2307.09788 | MEDIUM | Feature drift as a representation-level leak diagnostic, the L2-normalisation identity that removes count exactly, group-size numbers, and the augmentation control a referee will demand. |
| BioArc, 2512.00283 | MEDIUM | Its own appendix number (supernet rho = 0.32 without per-path finetuning) is the citation for why we train candidates from scratch instead. |
| SARA, 2608.00316 | MEDIUM | LLM-driven BO gains sample efficiency early and ties classically at the end. Relevant to our budget, and it closes the "agent drives the optimiser" opening. |
| AgentNAS, 2607.07984 | MEDIUM | Closes the "LLM proposes architectures" opening. Read as a boundary on our novelty claim. |
| Agentic supernet MAS, 2502.04180 | MEDIUM | Closes the opening BioArc's own future work named. Read as a boundary on our novelty claim. |

---

## 7. Not yet read

No paper in this file was flagged ABSTRACT_ONLY. All twelve summarised papers were read at FULL depth, with the per-paper read caveats noted inline (several readers saw figure captions rather than the figures themselves; specific instances are noted in the relevant entries).

Six entries are recorded from the human's own earlier direct reads and the descriptions supplied with them, not from a full pass in this round: CosmoBench (2507.03707), BioArc (2512.00283), SARA (2608.00316), AgentNAS (2607.07984), Multi-agent architecture search via agentic supernet (2502.04180), and Contardo et al. (2503.22654). Numbers quoted for those six are second hand and should be re-verified against the sources before any of them goes into a deliverable.

Open items that need a source and do not have one yet:

- A citable treatment of the CAMELS halo-finder resolution threshold and its dependence on particle mass, which is the mechanism our whole L2 claim rests on. We are currently asserting it from the simulation setup rather than citing a measurement.
- Villanueva-Domingo's CosmoGraphNet and HaloGraphNet, cited by two of the papers above as the origin of the invariant edge feature set and as evidence that a radius graph beats kNN on this data. Not read here.
- Makinen et al., "The Cosmic Graph", cited by arXiv 2405.13119 as independently finding that the simple statistic beats a more expressive GNN on Quijote halos. Not read here.
- PointNeXt (Qian et al. 2022), which supplies every unstated default in arXiv 2211.12346, including the pooling operator that paper never writes down. Needs checking at source before we build on the max-reduction assumption.
- Knyazev et al. 2019 (arXiv 1905.02850), the origin of the Colors-3 and Triangles counting-sensitivity datasets. Not read here.