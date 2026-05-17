#pragma once
// Multiple Correspondence Analysis on a categorical MSA.
//
// Algorithm: standard CA-style SVD on the row-mass-standardised indicator matrix.
//   1.  Build binary indicator matrix Z (N × J) — one column per unique
//       residue per alignment column; each row sums to L (alignment length).
//   2.  Form the correspondence matrix P = Z / (N·L).
//   3.  Compute row masses r_i = 1/N (uniform) and column masses c_j.
//   4.  Standardised residual matrix S = Dr^{-½} (P − r cᵀ) Dc^{-½}.
//   5.  Thin SVD of S.  The first singular value is trivially ≈ 1 and is
//       discarded; the next n_components columns give the principal row coords:
//         F = Dr^{-½} U[·,1:k] Σ[1:k].

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct MCAResult {
    Eigen::MatrixXd coords;   // N × n_components row principal coordinates
    Eigen::VectorXd inertia;  // squared singular values (explained inertia) per component
};

inline MCAResult fit_mca(const std::vector<std::string>& seqs, int n_components = 2) {
    if (seqs.empty()) throw std::runtime_error("empty sequence set");
    const int N = static_cast<int>(seqs.size());
    const int L = static_cast<int>(seqs[0].size());
    if (L == 0) throw std::runtime_error("zero-length sequences");

    // Build per-column alphabet (sorted for binary search)
    std::vector<std::vector<char>> alpha(L);
    for (int col = 0; col < L; ++col) {
        std::map<char, int> freq;
        for (const auto& s : seqs) ++freq[s[col]];
        for (auto& [c, _] : freq) alpha[col].push_back(c);
        std::sort(alpha[col].begin(), alpha[col].end());
    }

    // Column offsets in indicator matrix
    std::vector<int> off(L + 1, 0);
    for (int col = 0; col < L; ++col)
        off[col + 1] = off[col] + static_cast<int>(alpha[col].size());
    const int J = off[L];

    // Build Z (N × J)
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(N, J);
    for (int i = 0; i < N; ++i) {
        for (int col = 0; col < L; ++col) {
            char c = seqs[i][col];
            const auto& a = alpha[col];
            auto it = std::lower_bound(a.begin(), a.end(), c);
            if (it != a.end() && *it == c)
                Z(i, off[col] + static_cast<int>(it - a.begin())) = 1.0;
        }
    }

    // Correspondence matrix and masses
    const double total = static_cast<double>(N) * L;
    Eigen::MatrixXd P = Z / total;
    Eigen::VectorXd r      = P.rowwise().sum();  // ≈ 1/N for all rows
    Eigen::VectorXd c_mass = P.colwise().sum();  // J

    // Inverse-sqrt mass diagonals (guard against zero-mass columns)
    Eigen::VectorXd dr_inv_sqrt(N), dc_inv_sqrt(J);
    for (int i = 0; i < N; ++i)
        dr_inv_sqrt(i) = (r(i) > 1e-14) ? 1.0 / std::sqrt(r(i)) : 0.0;
    for (int j = 0; j < J; ++j)
        dc_inv_sqrt(j) = (c_mass(j) > 1e-14) ? 1.0 / std::sqrt(c_mass(j)) : 0.0;

    // Standardised residual matrix S = Dr^{-½} (P − r cᵀ) Dc^{-½}
    Eigen::MatrixXd S = P;
    for (int j = 0; j < J; ++j) S.col(j) -= r * c_mass(j);
    for (int i = 0; i < N; ++i) S.row(i) *= dr_inv_sqrt(i);
    for (int j = 0; j < J; ++j) S.col(j) *= dc_inv_sqrt(j);

    // Thin SVD — only U is needed for row coordinates
    Eigen::BDCSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeThinU);
    const int rank = static_cast<int>(svd.singularValues().size());

    // Skip index 0 (trivial component, σ ≈ 1.0)
    const int k = std::min(n_components, rank - 1);
    if (k < 1)
        throw std::runtime_error(
            "cannot extract " + std::to_string(n_components) +
            " MCA components from this data; try a smaller -k value");

    Eigen::MatrixXd U_k = svd.matrixU().middleCols(1, k);      // N × k
    Eigen::VectorXd s_k = svd.singularValues().segment(1, k);  // k

    MCAResult res;
    res.coords  = dr_inv_sqrt.asDiagonal() * U_k * s_k.asDiagonal();
    res.inertia = s_k.array().square();
    return res;
}
