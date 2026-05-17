# DrinkMe

Shrink a multiple sequence alignment to achieve phylogeny reconstruction.

**Pipeline:** FASTA MSA → column cleansing → Multiple Correspondence Analysis → Ward hierarchical clustering → Newick tree

## Method

1. **Column cleansing** — drops alignment columns whose gap fraction exceeds `1 - threshold`. Lowercase residues (HMMER insert states) are counted as gaps by default. All input sequences are retained regardless.
2. **MCA** — the standardised residual matrix S is never formed explicitly; instead it is applied as a sparse matrix–vector product (via the binary indicator matrix Z, N×L non-zeros) plus a rank-1 correction. A randomised truncated SVD (Halko–Martinsson–Tropp 2011) with subspace iteration extracts only the *k+1* components needed, at O(N·L·k) cost. The trivial first component (σ ≈ 1) is discarded.
3. **Ward clustering** — agglomerative clustering minimising within-cluster variance at each merge step. The implementation uses the **nearest-neighbour chain algorithm** (O(N²·d) time, O(N·d) memory), which exploits the reducibility of Ward's distance (`d(AB,C) ≥ min(d(A,C), d(B,C))`), and produces the same dendrogram as the naïve O(N³) scan. Branch lengths equal Ward merge distances.
4. **Newick output** — written to stdout, ready for any tree viewer (e.g. FigTree, iTOL, ETE3).

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

```
drinkme <alignment.fasta> [options]

Options:
  -k, --components N   MCA dimensions to keep (default: 2)
  -t, --threshold F    Min non-gap fraction to keep a column (default: 0.5)
  --keep-lowercase     Treat lowercase residues as valid (not as gaps)
  -v, --verbose        Print diagnostics to stderr
```

### Example

```bash
./build/drinkme sequences.fasta -k 4 -v > tree.nwk
```

## Output

A single Newick string on stdout, e.g.:

```
((seq1:0.312,seq2:0.312):0.891,(seq3:0.541,seq4:0.541):0.662);
```

Labels containing Newick-reserved characters (spaces, parentheses, colons, etc.) are automatically single-quoted.
