// drinkme — shrink your MSA into a phylogenetic tree.
//
// Pipeline:
//   FASTA MSA  →  MCA (reduce to k dimensions)
//   →  Ward hierarchical clustering  →  Newick tree (stdout)
//
// Usage:
//   drinkme <alignment.fasta> [options]
//   Options:
//     -k, --components N    MCA dimensions to keep  (default: 2)
//     -v, --verbose         Print diagnostics to stderr

#include "fasta.hpp"
#include "linkage.hpp"
#include "mca.hpp"
#include "newick.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <alignment.fasta> [options]\n"
        << "Options:\n"
        << "  -k, --components N   MCA dimensions to keep (default: 2)\n"
        << "  -v, --verbose        Print diagnostics to stderr\n";
}

int main(int argc, char* argv[]) {
    std::string fasta_path;
    int  n_components = 2;
    bool verbose      = false;

    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* a, const char* b) {
            return std::strcmp(argv[i], a) == 0 || std::strcmp(argv[i], b) == 0;
        };
        if (eq("-k", "--components")) {
            if (++i >= argc) { std::cerr << "missing argument for -k\n"; return 1; }
            n_components = std::stoi(argv[i]);
            if (n_components < 1) { std::cerr << "-k must be >= 1\n"; return 1; }
        } else if (eq("-v", "--verbose")) {
            verbose = true;
        } else if (argv[i][0] != '-') {
            fasta_path = argv[i];
        } else {
            std::cerr << "unknown option: " << argv[i] << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (fasta_path.empty()) { usage(argv[0]); return 1; }

    try {
        // --- 1. Parse alignment ---
        auto seqs = parse_fasta(fasta_path);
        std::vector<std::string> names, raw;
        names.reserve(seqs.size());
        raw.reserve(seqs.size());
        for (auto& s : seqs) { names.push_back(s.name); raw.push_back(s.seq); }

        if (verbose)
            std::cerr << "[drinkme] loaded " << seqs.size()
                      << " sequences, length " << raw[0].size() << '\n';

        // --- 2. MCA ---
        if (verbose)
            std::cerr << "[drinkme] MCA (k=" << n_components << ")...\n";

        auto mca_res = fit_mca(raw, n_components);

        if (verbose) {
            double total = mca_res.inertia.sum();
            std::cerr << "[drinkme] inertia per component:";
            for (int i = 0; i < mca_res.inertia.size(); ++i) {
                double pct = 100.0 * mca_res.inertia(i) / total;
                std::cerr << "  dim" << (i + 1) << "=" << mca_res.inertia(i)
                          << " (" << std::fixed << std::setprecision(1) << pct << "%)";
            }
            std::cerr << '\n';
        }

        // --- 3. Ward hierarchical clustering ---
        if (verbose) std::cerr << "[drinkme] Ward clustering...\n";
        auto Z = ward_linkage(mca_res.coords);

        // --- 4. Newick output ---
        if (verbose) std::cerr << "[drinkme] writing Newick tree...\n";
        std::cout << to_newick(Z, names) << '\n';

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
