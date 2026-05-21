# DrinkMe

**Hierarchical clustering of massive MSAs via MCA-based dimensionality reduction followed by Ward linkage.**

The name is a nod to Alice in Wonderland: just as the "Drink Me" potion makes Alice small enough to pass through the tiny door, DrinkMe compresses a high-dimensional categorical alignment — potentially thousands of sequences over hundreds of columns — into a compact continuous coordinate space, making large-scale hierarchical clustering tractable. The key step is Multiple Correspondence Analysis (MCA), which projects each sequence from the $N \times J$ binary indicator space (where $J$ can be in the thousands) down to $k$ principal dimensions (typically 2–10), before Ward linkage is applied. This means pairwise distances are never computed in the original space: clustering operates entirely on the low-dimensional MCA coordinates.

**Pipeline:** FASTA MSA → column cleansing → H&H sequence weighting → cardinality + entropy + CV column weighting → Multiple Correspondence Analysis (shrink to $k$ dimensions) → Ward hierarchical clustering → Newick tree

## Method

### Notation

Let $N$ be the number of sequences and $L$ the number of alignment columns after cleansing. Column $j$ has an observed alphabet $\mathcal{A}_j$ of size $Q_j$. The column offset and total indicator dimension are

```math
\text{off}(j) = \sum_{l < j} Q_l, \qquad J = \text{off}(L).
```

### Column cleansing

Column $j$ is retained if and only if its non-gap count meets the threshold:

```math
\text{non-gap}(j) \;\geq\; \lceil \theta\, N \rceil,
```

where $\theta$ is `--threshold` (default 0.5). Lowercase residues are treated as gaps unless `--keep-lowercase` is set; all input sequences are kept regardless of their own gap content.

### Multiple Correspondence Analysis

**Sequence weighting (Henikoff & Henikoff 1994).** Over-represented sequence clusters would dominate the MCA by sheer count, biasing the principal coordinates towards the majority subfamily. To correct for this without requiring an explicit identity threshold, each sequence $i$ is assigned a positional weight

```math
w_i = \frac{1}{L}\sum_{j\,:\,s_{ij}\neq\text{gap}} \frac{1}{k_j\, c_{ij}},
\qquad \sum_i w_i = 1,
```

where $k_j$ is the number of distinct non-gap characters in column $j$ and $c_{ij}$ is the count of sequences carrying the same character as $i$ at column $j$. A sequence that shares a rare character with few others (small $c_{ij}$) at a diverse column (large $k_j$) receives a larger weight contribution; redundant clusters — where many sequences agree and $c_{ij}$ is large — are automatically downweighted.

**Binary indicator matrix.** The $N \times J$ matrix $\mathbf{Z}$ has exactly $N \cdot L$ non-zeros, one per alignment cell:

```math
Z_{i,\,\text{off}(j)+\alpha} = N w_i
\quad \text{if sequence } i \text{ carries character } \alpha \in \mathcal{A}_j \text{ at column } j.
```

Setting $Z_{ij} = N w_i$ (rather than $1$) keeps the total $T = NL$ unchanged and preserves the structure of the rank-1 correction; row masses become $r_i = w_i$ automatically.

**Row and column masses.** With total weight $T = NL$:

```math
r_i = \frac{1}{T}\sum_j Z_{ij} = w_i,
\qquad
c_j = \frac{1}{T}\sum_i Z_{ij}.
```

**Standardised residual operator.** The $N \times J$ matrix

```math
\mathbf{S} = \mathbf{D}_r^{-1/2}
             \!\left(\frac{\mathbf{Z}}{T} - \mathbf{r}\mathbf{c}^\top\right)
             \mathbf{D}_c^{-1/2},
```

where $\mathbf{D}_r = \mathrm{diag}(r_i)$ and $\mathbf{D}_c = \mathrm{diag}(c_j)$, is never formed explicitly. Matrix–vector products are computed as a sparse operation plus a rank-1 correction that removes the trivial mean:

```math
\mathbf{S}\mathbf{X}
  = \mathbf{D}_r^{-1/2}\frac{\mathbf{Z}}{T}\mathbf{D}_c^{-1/2}\mathbf{X}
    - \sqrt{\mathbf{r}}\,(\sqrt{\mathbf{c}}^\top \mathbf{X}),
\qquad
\mathbf{S}^\top\mathbf{Y}
  = \mathbf{D}_c^{-1/2}\frac{\mathbf{Z}^\top}{T}\mathbf{D}_r^{-1/2}\mathbf{Y}
    - \sqrt{\mathbf{c}}\,(\sqrt{\mathbf{r}}^\top \mathbf{Y}).
```

**Randomised truncated SVD (Halko, Martinsson & Tropp 2011).** To extract the $k+1$ leading singular triples of $\mathbf{S}$ at $O(NLk)$ cost, draw $\boldsymbol{\Omega} \in \mathbb{R}^{J \times r}$ with $r = \min(k+10, N, J)$ i.i.d. Gaussian entries and perform 4 steps of subspace iteration:

```math
\mathbf{Y}_0 = \mathbf{S}\boldsymbol{\Omega}; \qquad
\mathbf{Y}_t = \mathbf{S}\,\mathrm{qr}\!\left(\mathbf{S}^\top\mathrm{qr}(\mathbf{Y}_{t-1})\right),
\quad t = 1,\ldots,4.
```

Project $\mathbf{Q} = \mathrm{qr}(\mathbf{Y}_4)$, form $\mathbf{B} = (\mathbf{S}^\top\mathbf{Q})^\top$, factorise $\mathbf{B} = \tilde{\mathbf{U}}\boldsymbol{\Sigma}\mathbf{V}^\top$, and recover $\mathbf{U} = \mathbf{Q}\tilde{\mathbf{U}}$. The leading singular triple ($\sigma_1 \approx 1$, the grand-mean component) is discarded; the remaining $k$ triples are retained.

**Row principal coordinates.** The $N \times k$ coordinate matrix used for clustering is:

```math
\mathbf{F} = \mathbf{D}_r^{-1/2}\,\mathbf{U}_k\,\boldsymbol{\Sigma}_k,
\qquad \boldsymbol{\Sigma}_k = \mathrm{diag}(\sigma_2, \ldots, \sigma_{k+1}),
```

where $\mathbf{U}_k$ comprises columns $2, \ldots, k+1$ of $\mathbf{U}$.

**Column weighting.** Three multiplicative layers are applied to the scaling vectors $\mathbf{D}_c^{-1/2}$ and $\sqrt{\mathbf{c}}$ for each column $j$:

| Layer | Weight | Effect |
| --- | --- | --- |
| Cardinality | $w_j^{(a)} = Q_j^{-1/2}$ | Equalises chi-square contribution across alphabet sizes |
| Gap-aware entropy | $w_j^{(b)} = \exp(-S_j)$ | Downweights high-entropy and gap-rich columns |
| Frequency CV | $w_j^{(c)} = \mathrm{std}_{20}(\mathbf{p}_j) / (1/20)$ | Upweights columns with residues concentrated relative to flat background |

The combined per-column weight is $w_j = w_j^{(a)} w_j^{(b)} w_j^{(c)}$. Columns with $Q_j \leq 1$ are left unscaled. The SVD is therefore applied to the weighted residual operator

```math
\mathbf{S}_w = \mathbf{D}_r^{-1/2}
\!\left(\frac{\mathbf{Z}}{T} - \mathbf{r}\mathbf{c}^\top\right)
\mathbf{W}\,\mathbf{D}_c^{-1/2},
\qquad
W_{\text{off}(j)+\alpha,\,\text{off}(j)+\alpha} = w_j,
```

where $\mathbf{W}$ is absorbed into the precomputed scaling vectors so the implicit matvec structure is unchanged.

**Cardinality layer.** Standard MCA assigns column $j$ a total chi-square weight proportional to $Q_j$ (the non-gap alphabet size): the contribution of column $j$'s block to the Frobenius norm of $\mathbf{S}$ is $\sum_{\alpha} c_{j\alpha}^{-1} \cdot c_{j\alpha} = Q_j$. Noise columns with large alphabets therefore dominate subfamily-discriminating (SDP) columns with small alphabets. The weight $w_j^{(a)} = Q_j^{-1/2}$ equalises this contribution so every column carries unit total chi-square weight.

**Gap-corrected Shannon entropy.** Without a gap correction, a column where most sequences carry a gap can appear spuriously informative: the few observed residues may happen to agree, and the column's low raw entropy would be rewarded with a high weight. To penalise gap-rich columns correctly, entropy is defined over the full column including gaps, with the gap mass mapped to its maximum-uncertainty equivalent.

Let $g_j$ be the weighted gap fraction at column $j$, and let $p_{j\alpha}$ be the weighted frequency of residue $\alpha$ relative to all sequences (gaps included), so the residue frequencies sum to $1 - g_j$. The residue distribution conditioned on non-gap sequences is

```math
\tilde{p}_{j\alpha} = \frac{p_{j\alpha}}{1 - g_j},
\qquad \sum_\alpha \tilde{p}_{j\alpha} = 1.
```

The actual Shannon entropy of the non-gap part is

```math
H^\text{actual}_j = -\sum_\alpha p_{j\alpha} \ln p_{j\alpha}.
```

A reference column with the same gap fraction $g_j$ but maximally uncertain non-gap content — residues distributed uniformly over all 20 amino acids — has entropy

```math
H^\text{ref}_j = -(1-g_j)\ln\frac{1-g_j}{20}.
```

Shifting $H^{\mathrm{actual}}_j$ so that a reference column maps to exactly $\ln 20$ (the maximum possible uncertainty) gives the gap-corrected score

```math
S_j = H^\text{actual}_j + \bigl(\ln 20 - H^\text{ref}_j\bigr).
```

Substituting $p_{j\alpha} = (1-g_j)\tilde{p}_{j\alpha}$ and expanding both terms, the $(1-g_j)\ln(1-g_j)$ factors cancel, leaving

```math
S_j = g_j\ln 20 + (1-g_j)\,H(\tilde{\mathbf{p}}_j),
\qquad H(\tilde{\mathbf{p}}_j) = -\sum_\alpha \tilde{p}_{j\alpha}\ln\tilde{p}_{j\alpha}.
```

A fully conserved, gap-free column gives $S_j = 0$ and $w_j^{(b)} = 1$ (maximum weight); a fully gapped column or a uniformly distributed column gives $S_j = \ln 20$ and $w_j^{(b)} = 1/20$ (minimum weight).

**Frequency CV layer.** The coefficient of variation of residue frequencies relative to a flat background rewards columns where a small number of residues dominate. For each column $j$, let $p_{j\alpha}^{(20)}$ be the weighted frequency of amino acid $\alpha$ across the standard 20-letter alphabet (zero for absent residues). Then

```math
w_j^{(c)} = \frac{\mathrm{std}_{20}(\mathbf{p}_j)}{1/20},
\qquad \mathrm{std}_{20}(\mathbf{p}_j) = \sqrt{\frac{1}{20}\sum_{\alpha=1}^{20}\!\left(p_{j\alpha}^{(20)} - \tfrac{1}{20}\right)^2}.
```

A column concentrated on a single residue (SDP-like) deviates strongly from the flat background; a column with frequencies spread uniformly over many residues deviates little.

### Ward hierarchical clustering

**Merge distance.** For clusters $A$ and $B$ with sizes $n_A$, $n_B$ and centroids $\mathbf{c}_A$, $\mathbf{c}_B$ in $\mathbf{F}$-space:

```math
d(A, B) = \sqrt{\frac{n_A\, n_B}{n_A + n_B}}\,\|\mathbf{c}_A - \mathbf{c}_B\|_2.
```

This equals $\sqrt{2\Delta\mathrm{Var}}$, the increase in total within-cluster variance caused by the merge (Ward 1963).

**Nearest-neighbour chain algorithm.** A stack-based traversal exploits Ward distance reducibility to find reciprocal nearest-neighbour pairs without recomputing all pairwise distances after each merge. When the top two stack entries are mutual nearest neighbours, they are merged:

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

> Halko, N., Martinsson, P. G. and Tropp, J. A. (2011). Finding structure with randomness:
> probabilistic algorithms for constructing approximate matrix decompositions.
> *SIAM Review*, 53(2), 217–288.
>
> Henikoff, S. and Henikoff, J. G. (1994). Position-based sequence weights.
> *Journal of Molecular Biology*, 243(4), 574–578.
>
> Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms.
> *The Computer Journal*, 26(4), 354–359.
>
> Shannon, C. E. (1948). A mathematical theory of communication.
> *Bell System Technical Journal*, 27(3), 379–423.
>
> Ward, J. H. (1963). Hierarchical grouping to optimize an objective function.
> *Journal of the American Statistical Association*, 58(301), 236–244.

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
  -v, --verbose        Print diagnostics to stderr
```

```bash
./build/drinkme sequences.fasta -k 4 -v > tree.nwk
```

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
