# DrinkMe

**Hierarchical clustering of massive MSAs via MCA-based dimensionality reduction followed by Ward linkage.**

The name is a nod to Alice in Wonderland: just as the "Drink Me" potion makes Alice small enough to pass through the tiny door, DrinkMe compresses a high-dimensional categorical alignment — potentially thousands of sequences over hundreds of columns — into a compact continuous coordinate space, making large-scale hierarchical clustering tractable. The key step is Multiple Correspondence Analysis (MCA), which projects each sequence from the $N \times J$ binary indicator space (where $J$ can be in the thousands) down to $k$ principal dimensions (typically 2–10), before Ward linkage is applied. This means pairwise distances are never computed in the original space: clustering operates entirely on the low-dimensional MCA coordinates.

**Pipeline:** FASTA MSA → column cleansing → Multiple Correspondence Analysis (shrink to $k$ dimensions) → Ward hierarchical clustering → Newick tree

## Method

### Notation

Let $N$ be the number of sequences and $L$ the number of alignment columns after cleansing. Let $\mathcal{A}_j$ be the set of distinct characters observed in column $j$, with $|\mathcal{A}_j|$ elements. Define the column offset $\text{off}(j) = \sum_{l < j} |\mathcal{A}_l|$ and the total category count $J = \text{off}(L)$.

### Column cleansing

Column $j$ is retained if and only if its non-gap count meets the threshold:

```math
\text{non-gap}(j) \;\geq\; \lceil \theta\, N \rceil,
```

where $\theta$ is `--threshold` (default 0.5). Lowercase residues are treated as gaps unless `--keep-lowercase` is set; all input sequences are kept regardless of their own gap content.

### Multiple Correspondence Analysis

**Binary indicator matrix.** The $N \times J$ binary matrix $\mathbf{Z}$ has exactly $N \cdot L$ non-zeros, one per alignment cell:

```math
Z_{i,\,\text{off}(j)+\alpha} = 1
\quad \text{if sequence } i \text{ carries character } \alpha \in \mathcal{A}_j \text{ at column } j.
```

**Row and column masses.** With total weight $T = NL$:

```math
r_i = \frac{1}{T}\sum_j Z_{ij} = \frac{1}{N},
\qquad
c_j = \frac{1}{T}\sum_i Z_{ij}.
```

Row masses are uniform because each sequence contributes exactly one character per column.

**Standardised residual operator.** The $N \times J$ matrix

```math
\mathbf{S} = \mathbf{D}_r^{-1/2}
             \!\left(\frac{\mathbf{Z}}{T} - \mathbf{r}\mathbf{c}^\top\right)
             \mathbf{D}_c^{-1/2},
```

where $\mathbf{D}_r = \operatorname{diag}(r_i)$ and $\mathbf{D}_c = \operatorname{diag}(c_j)$, is never formed explicitly. Matrix–vector products are computed as a sparse operation plus a rank-1 correction that removes the trivial mean:

```math
\mathbf{S}\mathbf{X}
  = \mathbf{D}_r^{-1/2}\frac{\mathbf{Z}}{T}\mathbf{D}_c^{-1/2}\mathbf{X}
    - \sqrt{\mathbf{r}}\,(\sqrt{\mathbf{c}}^\top \mathbf{X}),
\qquad
\mathbf{S}^\top\mathbf{Y}
  = \mathbf{D}_c^{-1/2}\frac{\mathbf{Z}^\top}{T}\mathbf{D}_r^{-1/2}\mathbf{Y}
    - \sqrt{\mathbf{c}}\,(\sqrt{\mathbf{r}}^\top \mathbf{Y}).
```

**Randomised truncated SVD (Halko, Martinsson & Tropp 2011).** To extract the $k+1$ leading singular triples of $\mathbf{S}$ at $O(NLk)$ cost:

1. Draw $\boldsymbol{\Omega} \in \mathbb{R}^{J \times r}$, $r = \min(k + 10,\, N,\, J)$, with i.i.d. Gaussian entries.
2. Subspace iteration ($n_\text{iter} = 4$): $\mathbf{Y}_0 = \mathbf{S}\boldsymbol{\Omega}$; then for $t = 1, \ldots, n_\text{iter}$ set $\mathbf{Y}_t = \mathbf{S}\,\operatorname{qr}(\mathbf{S}^\top\operatorname{qr}(\mathbf{Y}_{t-1}))$.
3. Project: $\mathbf{B} = (\mathbf{S}^\top\mathbf{Q})^\top$ where $\mathbf{Q} = \operatorname{qr}(\mathbf{Y}_{n_\text{iter}})$.
4. Factorise $\mathbf{B} = \tilde{\mathbf{U}}\boldsymbol{\Sigma}\mathbf{V}^\top$ and recover $\mathbf{U} = \mathbf{Q}\tilde{\mathbf{U}}$.

The leading singular triple (with $\sigma_1 \approx 1$, corresponding to the grand-mean component) is discarded; the remaining $k$ triples are retained.

**Row principal coordinates.** The $N \times k$ coordinate matrix used for clustering is:

```math
\mathbf{F} = \mathbf{D}_r^{-1/2}\,\mathbf{U}_k\,\boldsymbol{\Sigma}_k,
```

where $\mathbf{U}_k$ comprises columns $2, \ldots, k+1$ of $\mathbf{U}$ and $\boldsymbol{\Sigma}_k = \operatorname{diag}(\sigma_2, \ldots, \sigma_{k+1})$.

### Ward hierarchical clustering

**Merge distance.** For clusters $A$ and $B$ with sizes $n_A$, $n_B$ and centroids $\mathbf{c}_A$, $\mathbf{c}_B$ in $\mathbf{F}$-space:

```math
d(A, B) = \sqrt{\frac{n_A\, n_B}{n_A + n_B}}\,\|\mathbf{c}_A - \mathbf{c}_B\|_2.
```

This equals $\sqrt{2\,\Delta\text{Var}}$, the increase in total within-cluster variance caused by the merge (Ward 1963).

**Nearest-neighbour chain algorithm.** A stack-based traversal exploits Ward distance reducibility ($d(AB, C) \geq \min(d(A,C), d(B,C))$) to find reciprocal nearest-neighbour pairs without recomputing all pairwise distances after each merge. When the top two stack entries are mutual nearest neighbours, they are merged:

```math
\mathbf{c}_m = \frac{n_A\,\mathbf{c}_A + n_B\,\mathbf{c}_B}{n_A + n_B},
\qquad n_m = n_A + n_B,
\qquad h_m = d(A, B).
```

The procedure runs in $O(N^2 k)$ time and $O(Nk)$ memory, producing the same dendrogram as the naïve $O(N^3)$ scan (Murtagh 1983).

**Branch lengths.** Leaf nodes are assigned height $h = 0$; internal node $m$ has height $h_m$. The branch length from any node $x$ to its parent $p$ is:

```math
\text{branch}(x) = h_p - h_x.
```

### References

- Halko N, Martinsson PG, Tropp JA. (2011). Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Rev*, 53(2), 217–288. <https://doi.org/10.1137/090771806>
- Ward JH. (1963). Hierarchical grouping to optimize an objective function. *J Am Stat Assoc*, 58(301), 236–244. <https://doi.org/10.1080/01621459.1963.10500845>
- Murtagh F. (1983). A survey of recent advances in hierarchical clustering algorithms. *Comput J*, 26(4), 354–359. <https://doi.org/10.1093/comjnl/26.4.354>

## Performance

Indicative timings on a modern laptop (~10k sequences, ~1k alignment columns, ~200 retained after cleansing):

| Step | Time |
| ------ | ------ |
| Parse + cleanse | ~100 ms |
| MCA | ~200 ms |
| Ward | ~500 ms |
| Newick | ~10 ms |
| **Total** | **< 1 s** |

Memory scales as O(N·L) for the sparse indicator matrix, plus O((N+J)·k) workspace for the SVD — no dense N×J matrix is ever allocated.

## Dependencies

- C++17 compiler (GCC ≥ 7, Clang ≥ 5)
- CMake ≥ 3.15
- [Eigen3](https://eigen.tuxfamily.org/) ≥ 3.3 — auto-fetched by CMake if not installed

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The binary is placed at `build/drinkme`.

## Usage

```text
drinkme <alignment.fasta> [options]

Options:
  -k, --components N   MCA dimensions to keep (default: 2)
  -t, --threshold F    Min non-gap fraction to keep a column (default: 0.5)
  --keep-lowercase     Treat lowercase residues as valid (not as gaps)
  -m, --mode STR       Clustering mode: agglomerative (default) or divisive
  -s, --stop-size N    (divisive) Stop recursing at clusters <= N sequences (default: 3)
  -v, --verbose        Print diagnostics to stderr
```

### Agglomerative mode (default)

Global MCA on all sequences followed by a single Ward dendrogram.

```bash
./build/drinkme sequences.fasta -k 4 -v > tree.nwk
```

### Divisive mode

Recursively bisects the alignment: at each level, MCA and Ward clustering are applied to the local subset, the root split defines two groups, and the procedure recurses on each group until it reaches clusters of at most `--stop-size` sequences. Because each level performs its own column cleansing and MCA, dimensionality reduction is context-aware at every split.

```bash
./build/drinkme sequences.fasta --mode divisive --stop-size 5 -k 4 -v > tree.nwk
```

Branch lengths in divisive mode represent the Ward root-merge distance in the local MCA space at each split and are not globally comparable across levels.

## Output

A single Newick string on stdout, e.g.:

```text
((seq1:0.312,seq2:0.312):0.891,(seq3:0.541,seq4:0.541):0.662);
```

Labels containing Newick-reserved characters (spaces, parentheses, colons, etc.) are automatically single-quoted.

## Citation

If you use this tool in your research, please cite the original software and the associated publication (if any). The original module was developed by **Lucas Carrijo de Oliveira** (<lucas@ebi.ac.uk>). For now, you may reference the repository directly:

```text
Lucas C. de Oliveira. DrinkMe. 2026.
https://github.com/lucas-ebi/drinkme
```

A formal publication is in preparation; check the repository for updates.
