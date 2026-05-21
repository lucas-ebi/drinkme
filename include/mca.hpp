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
//   1. Build per-column alphabets.  Henikoff & Henikoff (1994) sequence weights
//      wᵢ are computed from per-column non-gap character diversity; Z entries
//      are set to N·wᵢ (total = N·L unchanged, row masses = wᵢ) to down-weight
//      sequences from over-represented clusters.
//   2. Build sparse Z (N × J) — exactly N·L non-zeros (H&H weighted).
//   3. Compute masses; apply three-layer column normalisation absorbed into
//      dc_inv_sqrt and √c (all multiplicative):
//        (a) Cardinality  1/√Qⱼ  — equalises Qⱼ-proportional chi-square bias.
//        (b) Gap-aware entropy  exp(−Sⱼ),  Sⱼ = gⱼ·ln20 + (1−gⱼ)·H(p̃ⱼ)
//            — penalises high-entropy and gap-rich columns.
//        (c) CV  std(pⱼ over 20 standard AAs) / (1/20)
//            — rewards character concentration relative to random.
//      Combined SDP(Q=3) vs noise(Q=12) contrast: ~23× vs ~2× for (a) alone.
//   4. Weighted operator Sw is applied implicitly:
//        Sw     x  =  Dr^{-½} (Z/total) W̃Dc^{-½} x  −  √r (W̃√cᵀ x)
//        Swᵀ    y  =  W̃Dc^{-½} (Zᵀ/total) Dr^{-½} y  −  W̃√c (√rᵀ y)
//   5. Randomized SVD extracts top k+1 singular triples in O(N·L·k) time.
//   6. Trivial σ ≈ 1 component is dropped; row principal coords returned.
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

inline MCAResult fit_mca(const std::vector<std::string>& seqs,
                         int  n_components = 2,
                         char gap_char     = '-') {
    using namespace mca_detail;

    if (seqs.empty()) throw std::runtime_error("empty sequence set");
    const int N = static_cast<int>(seqs.size());
    const int L = static_cast<int>(seqs[0].size());
    if (L == 0) throw std::runtime_error("zero-length sequences");

    const int gap_uc = static_cast<unsigned char>(gap_char);

    // ---- 1. Per-column alphabet via O(1) char lookup tables. ---------------
    std::vector<std::array<int, 256>> lookup(L);
    for (auto& tbl : lookup) tbl.fill(-1);
    std::vector<int> off(L + 1, 0);

    for (int col = 0; col < L; ++col) {
        std::array<bool, 256> seen{};
        for (const auto& s : seqs) seen[static_cast<unsigned char>(s[col])] = true;
        int idx = 0;
        for (int ch = 0; ch < 256; ++ch)
            if (seen[ch]) lookup[col][ch] = idx++;
        off[col + 1] = off[col] + idx;
    }
    const int J = off[L];

    // ---- 1b. Henikoff & Henikoff (1994) sequence weights. ------------------
    // cnt[col][alpha_idx] = count of sequences with that char at col (all chars).
    // k_j = distinct non-gap chars per column.
    std::vector<std::vector<int>> cnt(L);
    std::vector<int> k_j(L, 0);
    for (int col = 0; col < L; ++col) {
        const int q_total = off[col + 1] - off[col];
        cnt[col].assign(q_total, 0);
        for (const auto& s : seqs)
            cnt[col][lookup[col][static_cast<unsigned char>(s[col])]]++;
        const int gap_idx = lookup[col][gap_uc];
        k_j[col] = q_total - (gap_idx >= 0 ? 1 : 0);
    }

    // w[i] = (1/L) Σ_{non-gap j} 1/(k_j · c_{ij});  Σ w_i = 1 exactly.
    std::vector<double> hh_w(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int col = 0; col < L; ++col) {
            const int ch = static_cast<unsigned char>(seqs[i][col]);
            if (ch == gap_uc || k_j[col] <= 0) continue;
            const int idx  = lookup[col][ch];
            const int c_ij = cnt[col][idx];
            if (c_ij > 0)
                hh_w[i] += 1.0 / (static_cast<double>(k_j[col]) * c_ij);
        }
        hh_w[i] /= static_cast<double>(L);
    }
    // Floating-point safeguard (analytically Σ=1, but guard all-gap sequences).
    {
        double sum_w = 0.0;
        for (double wi : hh_w) sum_w += wi;
        if (sum_w > 1e-14)
            for (double& wi : hh_w) wi /= sum_w;
        else
            std::fill(hh_w.begin(), hh_w.end(), 1.0 / N);
    }

    // ---- 2. Build sparse Z (N × J), values = N·wᵢ. -------------------------
    // total = N·L unchanged; row masses become wᵢ automatically.
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(N) * L);
    for (int i = 0; i < N; ++i) {
        const std::string& s  = seqs[i];
        const double        zi = static_cast<double>(N) * hh_w[i];
        for (int col = 0; col < L; ++col) {
            int idx = lookup[col][static_cast<unsigned char>(s[col])];
            if (idx >= 0)
                trips.emplace_back(i, off[col] + idx, zi);
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
    Eigen::VectorXd r      = row_sums / total;
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

    // ---- 3b. Three-layer column normalisation. -----------------------------
    // col_sums[α] × L = weighted fraction of character α at its column.
    // Layers: (a) cardinality · (b) gap-aware entropy · (c) CV around 1/20.
    static const char AAs20[20] = {
        'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'
    };
    for (int col = 0; col < L; ++col) {
        const int gap_idx  = lookup[col][gap_uc];
        const int q_all    = off[col + 1] - off[col];
        const int q_nongap = q_all - (gap_idx >= 0 ? 1 : 0);
        if (q_nongap <= 1) continue;

        // (a) Cardinality: 1/√Q_j
        const double w_card = 1.0 / std::sqrt(static_cast<double>(q_nongap));

        // (b) Gap-aware entropy: exp(−S_j), S_j = g·ln20 + (1−g)·H(p̃)
        // c_mass[α]*L = Σ_{i with α} hh_w[i] = weighted fraction of α at col j.
        const double g_j = (gap_idx >= 0)
                           ? c_mass[off[col] + gap_idx] * L
                           : 0.0;
        const double ng  = 1.0 - g_j;
        double H_cond = 0.0;
        if (ng > 1e-14) {
            for (int a = 0; a < q_all; ++a) {
                if (a == gap_idx) continue;
                const double p_tilde = c_mass[off[col] + a] * L / ng;
                if (p_tilde > 1e-14) H_cond -= p_tilde * std::log(p_tilde);
            }
        }
        const double w_entr = std::exp(-(g_j * std::log(20.0) + ng * H_cond));

        // (c) CV of residue frequencies around mean 1/20
        constexpr double mean_aa = 1.0 / 20.0;
        double sum_sq = 0.0;
        for (int a = 0; a < 20; ++a) {
            const int aa_idx = lookup[col][static_cast<unsigned char>(AAs20[a])];
            const double p_aa = (aa_idx >= 0) ? c_mass[off[col] + aa_idx] * L : 0.0;
            const double dev  = p_aa - mean_aa;
            sum_sq += dev * dev;
        }
        const double w_cv = std::sqrt(sum_sq / 20.0) / mean_aa;

        const double w_j = w_card * w_entr * w_cv;
        if (w_j < 1e-15) continue;

        for (int a = 0; a < q_all; ++a) {
            const int idx = off[col] + a;
            dc_inv_sqrt(idx) *= w_j;
            sqrt_c(idx)      *= w_j;
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
