#pragma once
// Agglomerative (bottom-up) hierarchical clustering for MSA Newick generation.
//
// Pipeline: column cleansing → H&H sequence weighting → three-layer column weighting
//           → MCA (global) → Ward dendrogram → Newick tree.

#include "cleanse.hpp"
#include "linkage.hpp"
#include "mca.hpp"
#include "newick.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

inline std::string agglomerative_newick(
    const std::vector<std::string>& raw_seqs,
    const std::vector<std::string>& names,
    char   gap_char,
    bool   lowercase_as_gap,
    double threshold,
    int    n_components,
    bool   verbose = false)
{
    auto cl = cleanse_columns(raw_seqs, gap_char, lowercase_as_gap, threshold);
    if (verbose)
        std::cerr << "[drinkme] kept " << cl.kept_columns.size()
                  << "/" << raw_seqs[0].size() << " columns (threshold=" << threshold << ")\n";

    if (verbose) std::cerr << "[drinkme] MCA (k=" << n_components << ")...\n";
    auto mca_res = fit_mca(cl.seqs, n_components, gap_char);
    if (verbose) {
        double total = mca_res.inertia.sum();
        std::cerr << "[drinkme] inertia per component:";
        for (int i = 0; i < mca_res.inertia.size(); ++i) {
            double pct = 100.0 * mca_res.inertia(i) / total;
            std::cerr << "  dim" << (i + 1) << "=" << mca_res.inertia(i)
                      << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                      << std::defaultfloat;
        }
        std::cerr << '\n';
    }

    if (verbose) std::cerr << "[drinkme] Ward clustering...\n";
    auto Z = ward_linkage(mca_res.coords);

    if (verbose) std::cerr << "[drinkme] writing Newick tree...\n";
    return to_newick(Z, names);
}
