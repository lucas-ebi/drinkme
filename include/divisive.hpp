#pragma once
// Divisive (top-down) hierarchical clustering for MSA Newick generation.
//
// Recursively bisects the sequence set: at each level, cleanses columns for the
// local subset, runs MCA, applies Ward clustering, cuts at the root split into
// two groups, and recurses on each until it has <= stop_size sequences.
//
// Branch lengths represent the Ward root-merge distance in the local MCA space
// and are not globally comparable across recursion levels.

#include "cleanse.hpp"
#include "linkage.hpp"
#include "mca.hpp"
#include "newick.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace divisive_detail {

inline void collect_leaves(const std::vector<LinkageRow>& Z,
                            int node, int N,
                            std::vector<int>& out)
{
    if (node < N) { out.push_back(node); return; }
    const int idx = node - N;
    collect_leaves(Z, Z[idx].left,  N, out);
    collect_leaves(Z, Z[idx].right, N, out);
}

} // namespace divisive_detail

// Returns a Newick sub-string (no trailing ';', no branch annotation on root).
// The top-level caller appends ';' to produce a valid Newick string.
inline std::string divisive_newick(
    const std::vector<std::string>& raw_seqs,
    const std::vector<std::string>& names,
    char   gap_char,
    bool   lowercase_as_gap,
    double threshold,
    int    n_components,
    int    stop_size,
    bool   verbose = false)
{
    const int N = static_cast<int>(raw_seqs.size());

    if (N == 1) return newick_quote(names[0]);

    // Cleanse columns for this subset; if all are filtered, sequences are
    // indistinguishable here — return a star topology with zero branch lengths.
    CleanseResult cl;
    try {
        cl = cleanse_columns(raw_seqs, gap_char, lowercase_as_gap, threshold);
    } catch (const std::runtime_error&) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << '(';
        for (int i = 0; i < N; ++i) {
            if (i > 0) oss << ',';
            oss << newick_quote(names[i]) << ":0.000000";
        }
        return oss.str() + ')';
    }

    // MCA requires n_components + 1 <= min(N, J); N - 1 is a safe upper bound.
    const int eff_k = std::min(n_components, N - 1);

    auto mca_res = fit_mca(cl.seqs, eff_k);
    auto Z       = ward_linkage(mca_res.coords);

    // Base case: return the local Ward Newick for this small cluster.
    if (N <= stop_size) {
        std::string nwk = to_newick(Z, names);
        return nwk.substr(0, nwk.size() - 1);  // strip trailing ';'
    }

    // Split at root merge and collect leaf indices for each subtree.
    const double split_h = Z.back().height;
    std::vector<int> left_idx, right_idx;
    divisive_detail::collect_leaves(Z, Z.back().left,  N, left_idx);
    divisive_detail::collect_leaves(Z, Z.back().right, N, right_idx);

    // Build subsets preserving original (uncleansed) sequences so each
    // recursive level can re-cleanse for its own subset.
    std::vector<std::string> left_seqs, left_names, right_seqs, right_names;
    left_seqs.reserve(left_idx.size());   left_names.reserve(left_idx.size());
    right_seqs.reserve(right_idx.size()); right_names.reserve(right_idx.size());
    for (int i : left_idx)  { left_seqs.push_back(raw_seqs[i]);  left_names.push_back(names[i]); }
    for (int i : right_idx) { right_seqs.push_back(raw_seqs[i]); right_names.push_back(names[i]); }

    if (verbose)
        std::cerr << "[drinkme] divisive split " << N
                  << " → " << left_idx.size() << " + " << right_idx.size() << '\n';

    const std::string left_str  = divisive_newick(left_seqs,  left_names,
                                                   gap_char, lowercase_as_gap, threshold,
                                                   n_components, stop_size, verbose);
    const std::string right_str = divisive_newick(right_seqs, right_names,
                                                   gap_char, lowercase_as_gap, threshold,
                                                   n_components, stop_size, verbose);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << '(' << left_str  << ':' << split_h
        << ',' << right_str << ':' << split_h << ')';
    return oss.str();
}
