#pragma once
// Ward's hierarchical clustering on the MCA coordinate matrix.
//
// Ward distance between clusters A and B:
//   d(A,B) = sqrt( n_A · n_B / (n_A + n_B) ) · ‖c_A − c_B‖₂
//
// This equals sqrt(2 · ΔVariance) — the standard Ward D metric (not D²).
// Clusters are merged greedily by minimum d; centroids are updated as the
// size-weighted mean of the merged pair.  Heights are non-decreasing by
// the Ward criterion, guaranteeing a valid dendrogram.
//
// Node indices follow the scipy/fastcluster convention:
//   0 .. N-1        leaf nodes (one per input sequence)
//   N .. 2N-2       internal nodes (merge i → node N+i)
//   2N-2            root
//
// Implementation uses the nearest‑neighbour chain algorithm (O(N²·d) time,
// O(N·d) memory), which exploits the reducibility of Ward's distance:
//   d(AB, C) ≥ min(d(A,C), d(B,C))   for all C.
// This guarantees the same dendrogram as the naive O(N³) scan.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

struct LinkageRow {
    int    left;    // index of left child
    int    right;   // index of right child
    double height;  // Ward merge distance at this step
    int    size;    // total leaves under this internal node
};

inline std::vector<LinkageRow> ward_linkage(const Eigen::MatrixXd& X) {
    const int N = static_cast<int>(X.rows());
    if (N < 2) throw std::runtime_error("need at least 2 sequences for clustering");

    // Pre-allocate for 2N-1 potential nodes
    const int M = 2 * N - 1;
    const int d = static_cast<int>(X.cols());

    // char instead of bool avoids std::vector<bool>'s bit-packing overhead.
    std::vector<char> active(M, 0);
    std::vector<int>  sz(M, 0);

    // Row-major layout: centroid of cluster i is two adjacent doubles at row i.
    // Contiguous storage eliminates the per-VectorXd heap indirection.
    using CtrMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    CtrMat ctr(M, d);
    ctr.topRows(N) = X;

    for (int i = 0; i < N; ++i) { active[i] = 1; sz[i] = 1; }

    // Work in D² to avoid two sqrts per distance call; sqrt only for heights.
    auto ward_d2 = [&](int a, int b) -> double {
        double s = static_cast<double>(sz[a]) * sz[b] / (sz[a] + sz[b]);
        return s * (ctr.row(a) - ctr.row(b)).squaredNorm();
    };

    // NN‑chain algorithm — discovers merges in chain order, then sorts by height.
    struct RawMerge {
        int a, b;          // children (original IDs)
        double height;
        int size;
        int orig_node;     // internal node ID assigned during the chain loop
    };
    std::vector<RawMerge> raw_merges;
    raw_merges.reserve(N - 1);

    std::vector<int> chain;
    chain.reserve(N);
    int next       = N;          // next available internal‑node id
    int num_active = N;

    while (num_active > 1) {
        if (chain.empty()) {
            for (int i = 0; i < next; ++i)
                if (active[i]) { chain.push_back(i); break; }
        }

        const int top  = chain.back();
        const int prev = (chain.size() >= 2) ? chain[chain.size() - 2] : -1;

        // Nearest neighbour of `top`, tie‑breaking toward `prev`.
        // Scan only [0, next): slots beyond next have never been written.
        int    nn    = -1;
        double nn_d2 = std::numeric_limits<double>::infinity();
        for (int j = 0; j < next; ++j) {
            if (!active[j] || j == top) continue;
            double d2 = ward_d2(top, j);
            if (d2 < nn_d2 || (d2 == nn_d2 && j == prev)) { nn = j; nn_d2 = d2; }
        }

        if (nn == prev) {          // reciprocal nearest neighbours → merge
            chain.pop_back();
            chain.pop_back();

            int left  = std::min(top, prev);
            int right = std::max(top, prev);

            const int nab = sz[left] + sz[right];
            ctr.row(next) = (sz[left] * ctr.row(left) + sz[right] * ctr.row(right)) / nab;
            sz[next]      = nab;
            active[next]  = 1;
            active[left]  = 0;
            active[right] = 0;

            raw_merges.push_back({left, right, std::sqrt(nn_d2), nab, next});
            ++next;
            --num_active;
        } else {
            chain.push_back(nn);
        }
    }

    // Ward guarantees non‑decreasing heights after sorting (valid dendrogram).
    std::sort(raw_merges.begin(), raw_merges.end(),
              [](const RawMerge& a, const RawMerge& b) { return a.height < b.height; });

    // Re‑number internal nodes so that the i‑th merge becomes node N+i,
    // matching the scipy/fastcluster convention.
    std::vector<int> old_to_new(M, -1);
    for (int i = 0; i < N; ++i) old_to_new[i] = i;       // leaves stay the same
    for (int i = 0; i < (int)raw_merges.size(); ++i) {
        old_to_new[raw_merges[i].orig_node] = N + i;
    }

    std::vector<LinkageRow> result;
    result.reserve(N - 1);
    for (const auto& rm : raw_merges) {
        int l = rm.a, r = rm.b;
        if (l >= N) l = old_to_new[l];   // map old internal id → new N+index
        if (r >= N) r = old_to_new[r];
        result.push_back({l, r, rm.height, rm.size});
    }

    return result;
}
