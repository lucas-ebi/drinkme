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

#include <Eigen/Dense>
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
    std::vector<bool>            active(M, false);
    std::vector<int>             sz(M, 0);
    std::vector<Eigen::VectorXd> ctr(M);

    for (int i = 0; i < N; ++i) {
        active[i] = true;
        sz[i]     = 1;
        ctr[i]    = X.row(i);
    }

    auto ward_d = [&](int a, int b) -> double {
        double scale = std::sqrt(
            static_cast<double>(sz[a]) * sz[b] / (sz[a] + sz[b]));
        return scale * (ctr[a] - ctr[b]).norm();
    };

    std::vector<LinkageRow> result;
    result.reserve(N - 1);
    int next = N;

    for (int step = 0; step < N - 1; ++step) {
        double best = std::numeric_limits<double>::infinity();
        int bi = -1, bj = -1;

        for (int i = 0; i < M; ++i) {
            if (!active[i]) continue;
            for (int j = i + 1; j < M; ++j) {
                if (!active[j]) continue;
                double d = ward_d(i, j);
                if (d < best) { best = d; bi = i; bj = j; }
            }
        }

        const int nab = sz[bi] + sz[bj];
        ctr[next] = (sz[bi] * ctr[bi] + sz[bj] * ctr[bj]) / nab;
        sz[next]  = nab;
        active[next] = true;

        result.push_back({bi, bj, best, nab});

        active[bi] = false;
        active[bj] = false;
        ++next;
    }

    return result;
}
