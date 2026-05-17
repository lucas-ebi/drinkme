# DrinkMe

Shrink a multiple sequence alignment to achieve phylogeny reconstruction.

**Pipeline:** FASTA MSA → column cleansing → Multiple Correspondence Analysis → Ward hierarchical clustering → Newick tree

## Method

1. **Column cleansing** — drops alignment columns whose gap fraction exceeds `1 - threshold`. Lowercase residues (HMMER insert states) are counted as gaps by default. All input sequences are retained regardless.
2. **MCA** — treats each surviving column as a categorical variable and projects sequences into a low-dimensional Euclidean space via SVD of the standardised residual matrix. The trivial first component (σ ≈ 1) is discarded; the next *k* components are retained.
3. **Ward clustering** — agglomerative clustering minimising within-cluster variance at each merge step.  
   The implementation uses the **nearest‑neighbour chain algorithm** (O(N²·d) time, O(N·d) memory) which exploits the reducibility of Ward's distance:
   `d(AB, C) ≥ min(d(A,C), d(B,C))`  
   This produces exactly the same dendrogram as the naïve O(N³) scan (as used in scipy & fastcluster) but scales to tens of thousands of sequences. Branch lengths in the output tree equal Ward merge distances.
4. **Newick output** — written to stdout, ready for any tree viewer (e.g. FigTree, iTOL, ETE3).

## Performance

On a modern laptop, clustering ~10k sequences from an MSA of ~1k+ columns (~50 informative columns after column filtering) completes in under **3 seconds**. Memory usage remains modest because only the reduced MCA coordinate matrix (size N × k) and O(N) auxiliary vectors are stored.

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
  -t, --threshold F    Min non-gap fraction to keep a column (default: 0.9)
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
