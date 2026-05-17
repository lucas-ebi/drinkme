# DrinkMe

Shrink a multiple sequence alignment into a phylogenetic tree.

**Pipeline:** FASTA MSA → Multiple Correspondence Analysis → Ward hierarchical clustering → Newick tree

## Method

1. **MCA** — treats each alignment column as a categorical variable and projects sequences into a low-dimensional Euclidean space via SVD of the standardised residual matrix. The trivial first component (σ ≈ 1) is discarded; the next *k* components are retained.
2. **Ward clustering** — agglomerative clustering minimising within-cluster variance at each merge step. Branch lengths in the output tree equal Ward merge distances.
3. **Newick output** — written to stdout, ready for any tree viewer (e.g. FigTree, iTOL, ETE3).

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
