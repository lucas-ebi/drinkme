#pragma once
// Multiple Correspondence Analysis on a categorical MSA — scalable version.
//
// Algorithm: randomized truncated SVD (Halko–Martinsson–Tropp, 2011) applied
// implicitly to the row-mass-standardised indicator matrix S, where S is never
// materialised — it's represented as a sparse matvec minus a rank-1 correction.
// Recovers the top n_components+1 singular triples; the trivial σ ≈ 1 leading
// component is discarded, as in standard MCA.
//
// Pipeline:
//   1. Build sparse indicator matrix Z (N × J) — exactly N·L non-zeros.
//   2. Compute row/column masses; never form P, rcᵀ, or S explicitly.
//   3. Compute Dc^{-½} and √c from column masses, then apply cardinality
//      normalisation: scale column j's block by 1/√Qⱼ (Qⱼ = alphabet size)
//      so every column contributes equally to the chi-square metric.
//      Without this, high-cardinality noise columns dominate SDP columns.
//      The weighted operator Sw is applied implicitly:
//        Sw     x  =  Dr^{-½} (Z/total) W̃Dc^{-½} x  −  √r (W̃√cᵀ x)
//        Swᵀ    y  =  W̃Dc^{-½} (Zᵀ/total) Dr^{-½} y  −  W̃√c (√rᵀ y)
//      where W̃ absorbs the per-column 1/√Qⱼ factors into dc_inv_sqrt/√c.
//   4. Randomized SVD (Halko–Martinsson–Tropp, 2011) extracts top k+1
//      singular triples in O((nnz(Z) + N + J) · (k+p) · iters) time.
//   5. Trivial σ ≈ 1 component is dropped; row principal coords returned.
//
// Memory: O(N·L) for Z, plus O((N+J)·(k+p)) workspace.  No dense N×J matrix
// is ever allocated, so this scales to MSAs that wouldn't fit otherwise.

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct MCAResult {
    Eigen::MatrixXd coords;   // N × n_components row principal coordinates
    Eigen::VectorXd inertia;  // squared singular values per retained component
};

namespace mca_detail {

// Implicit operator representing the standardised residual matrix S.
// Applies S and Sᵀ to vectors / matrices without ever materialising S or P.
struct SOperator {
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& Z;   // N × J
    const Eigen::SparseMatrix<double, Eigen::ColMajor>& Zt;  // J × N (= Zᵀ)
    const Eigen::VectorXd& dr_inv_sqrt;  // N
    const Eigen::VectorXd& dc_inv_sqrt;  // J
    const Eigen::VectorXd& sqrt_r;       // N
    const Eigen::VectorXd& sqrt_c;       // J
    double total;
    int N, J;

    // Y = S · X, where X is J × b and Y is N × b.
    Eigen::MatrixXd apply(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd Xs = dc_inv_sqrt.asDiagonal() * X;
        Eigen::MatrixXd Y  = (Z * Xs) / total;
        Y = dr_inv_sqrt.asDiagonal() * Y;
        // Rank-1 correction: subtract √r · (√cᵀ X) for each column.
        Eigen::RowVectorXd coef = sqrt_c.transpose() * X;  // 1 × b
        Y.noalias() -= sqrt_r * coef;
        return Y;
    }

    // X = Sᵀ · Y, where Y is N × b and X is J × b.
    Eigen::MatrixXd apply_t(const Eigen::MatrixXd& Y) const {
        Eigen::MatrixXd Ys = dr_inv_sqrt.asDiagonal() * Y;
        Eigen::MatrixXd X  = (Zt * Ys) / total;
        X = dc_inv_sqrt.asDiagonal() * X;
        Eigen::RowVectorXd coef = sqrt_r.transpose() * Y;  // 1 × b
        X.noalias() -= sqrt_c * coef;
        return X;
    }
};

// Randomized SVD via subspace iteration with re-orthogonalisation.
// Returns top `k` singular triples (U: N×k, s: k, V: J×k), ordered descending.
struct TruncatedSVD {
    Eigen::MatrixXd U;
    Eigen::VectorXd s;
    Eigen::MatrixXd V;
};

inline TruncatedSVD randomized_svd(const SOperator& op,
                                   int k,
                                   int oversample = 10,
                                   int n_iter     = 4,
                                   uint64_t seed  = 0xC0FFEE) {
    const int N = op.N, J = op.J;
    const int r = std::min({k + oversample, N, J});

    // Draw J × r Gaussian test matrix.
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nrm(0.0, 1.0);
    Eigen::MatrixXd Omega(J, r);
    for (int j = 0; j < J; ++j)
        for (int c = 0; c < r; ++c) Omega(j, c) = nrm(rng);

    // Range finder with subspace iteration.  Re-orthogonalise each pass to
    // damp roundoff that would otherwise wash out the trailing singular vectors.
    Eigen::MatrixXd Y = op.apply(Omega);  // N × r
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(N, r);

    for (int it = 0; it < n_iter; ++it) {
        Eigen::MatrixXd Z2 = op.apply_t(Q);  // J × r
        qr.compute(Z2);
        Eigen::MatrixXd Qz = qr.householderQ() * Eigen::MatrixXd::Identity(J, r);
        Y = op.apply(Qz);                    // N × r
        qr.compute(Y);
        Q = qr.householderQ() * Eigen::MatrixXd::Identity(N, r);
    }

    // Project to a small r × J problem and SVD it directly.
    Eigen::MatrixXd B = op.apply_t(Q).transpose();  // r × J
    Eigen::BDCSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    TruncatedSVD out;
    out.U = Q * svd.matrixU().leftCols(k);
    out.s = svd.singularValues().head(k);
    out.V = svd.matrixV().leftCols(k);
    return out;
}

}  // namespace mca_detail

inline MCAResult fit_mca(const std::vector<std::string>& seqs, int n_components = 2) {
    using namespace mca_detail;

    if (seqs.empty()) throw std::runtime_error("empty sequence set");
    const int N = static_cast<int>(seqs.size());
    const int L = static_cast<int>(seqs[0].size());
    if (L == 0) throw std::runtime_error("zero-length sequences");

    // ---- 1. Per-column alphabet via O(1) char lookup tables. ---------------
    std::vector<std::array<int, 256>> lookup(L);
    for (auto& tbl : lookup) tbl.fill(-1);
    std::vector<int> off(L + 1, 0);

    for (int col = 0; col < L; ++col) {
        // First pass: mark observed characters.
        std::array<bool, 256> seen{};
        for (const auto& s : seqs) seen[static_cast<unsigned char>(s[col])] = true;
        // Assign indices in sorted character order (stable, reproducible).
        int idx = 0;
        for (int ch = 0; ch < 256; ++ch)
            if (seen[ch]) lookup[col][ch] = idx++;
        off[col + 1] = off[col] + idx;
    }
    const int J = off[L];

    // ---- 2. Build sparse Z (N × J) — exactly N·L non-zeros. ---------------
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(N) * L);
    for (int i = 0; i < N; ++i) {
        const std::string& s = seqs[i];
        for (int col = 0; col < L; ++col) {
            int idx = lookup[col][static_cast<unsigned char>(s[col])];
            if (idx >= 0)
                trips.emplace_back(i, off[col] + idx, 1.0);
        }
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> Z(N, J);
    Z.setFromTriplets(trips.begin(), trips.end());
    Z.makeCompressed();
    Eigen::SparseMatrix<double, Eigen::ColMajor> Zt = Z.transpose();
    Zt.makeCompressed();

    // ---- 3. Masses and inverse-square-root scalings. -----------------------
    const double total = static_cast<double>(N) * L;
    Eigen::VectorXd row_sums = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd col_sums = Eigen::VectorXd::Zero(J);
    for (int k = 0; k < Z.outerSize(); ++k)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(Z, k); it; ++it) {
            row_sums(it.row()) += it.value();
            col_sums(it.col()) += it.value();
        }
    Eigen::VectorXd r      = row_sums / total;   // ≈ 1/N
    Eigen::VectorXd c_mass = col_sums / total;

    Eigen::VectorXd dr_inv_sqrt(N), dc_inv_sqrt(J), sqrt_r(N), sqrt_c(J);
    for (int i = 0; i < N; ++i) {
        sqrt_r(i)      = std::sqrt(r(i));
        dr_inv_sqrt(i) = (r(i) > 1e-14) ? 1.0 / sqrt_r(i) : 0.0;
    }
    for (int j = 0; j < J; ++j) {
        sqrt_c(j)      = std::sqrt(c_mass(j));
        dc_inv_sqrt(j) = (c_mass(j) > 1e-14) ? 1.0 / sqrt_c(j) : 0.0;
    }

    // ---- 3b. Cardinality normalisation. ------------------------------------
    // Scale column j's block by 1/sqrt(Q_j) so every column contributes
    // equally to the chi-square metric regardless of alphabet size.
    // Without this, high-cardinality columns (noise/marginal positions with
    // many rare characters) dominate over low-cardinality SDP columns.
    for (int col = 0; col < L; ++col) {
        const int q_j = off[col + 1] - off[col];
        if (q_j <= 1) continue;
        const double w = 1.0 / std::sqrt(static_cast<double>(q_j));
        for (int a = 0; a < q_j; ++a) {
            const int idx = off[col] + a;
            dc_inv_sqrt(idx) *= w;
            sqrt_c(idx)      *= w;
        }
    }

    // ---- 4. Implicit operator + randomized truncated SVD. ------------------
    SOperator S{Z, Zt, dr_inv_sqrt, dc_inv_sqrt, sqrt_r, sqrt_c, total, N, J};

    const int want = n_components + 1;  // +1 for the trivial component we'll drop
    const int rank_cap = std::min(N, J);
    if (want > rank_cap)
        throw std::runtime_error(
            "cannot extract " + std::to_string(n_components) +
            " MCA components from this data; try a smaller -k value");

    auto svd = randomized_svd(S, want);

    // ---- 5. Drop trivial leading component, build row principal coords. ----
    Eigen::MatrixXd U_k = svd.U.middleCols(1, n_components);
    Eigen::VectorXd s_k = svd.s.segment(1, n_components);

    MCAResult res;
    res.coords  = dr_inv_sqrt.asDiagonal() * U_k * s_k.asDiagonal();
    res.inertia = s_k.array().square();
    return res;
}
