// DrinkMe – phylogenetic tree from MSA using MCA and Ward hierarchical clustering
//
// Pipeline:
//   FASTA MSA  →  cleanse columns  →  MCA (reduce to k dimensions)
//   →  Ward hierarchical clustering  →  Newick tree (stdout)
//
// Usage:
//   drinkme <alignment.fasta> [options]
//   Options:
//     -k, --components N    MCA dimensions to keep  (default: 2)
//     -t, --threshold F     Min non-gap fraction to keep a column (default: 5)
//     --keep-lowercase      Treat lowercase residues as valid (not as gaps)
//     -v, --verbose         Print diagnostics to stderr

#include "cleanse.hpp"
#include "fasta.hpp"
#include "linkage.hpp"
#include "mca.hpp"
#include "newick.hpp"

#include <chrono>
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
        << "  -t, --threshold F    Min non-gap fraction to keep a column (default: 0.5)\n"
        << "  --keep-lowercase     Treat lowercase residues as valid (not as gaps)\n"
        << "  -v, --verbose        Print diagnostics to stderr\n";
}

int main(int argc, char* argv[]) {
    std::string fasta_path;
    int    n_components      = 2;
    double threshold         = 0.5;
    bool   lowercase_as_gap  = true;
    bool   verbose           = false;

    for (int i = 1; i < argc; ++i) {
        auto eq = [&](const char* a, const char* b) {
            return std::strcmp(argv[i], a) == 0 || std::strcmp(argv[i], b) == 0;
        };
        if (eq("-k", "--components")) {
            if (++i >= argc) { std::cerr << "missing argument for -k\n"; return 1; }
            n_components = std::stoi(argv[i]);
            if (n_components < 1) { std::cerr << "-k must be >= 1\n"; return 1; }
        } else if (eq("-t", "--threshold")) {
            if (++i >= argc) { std::cerr << "missing argument for -t\n"; return 1; }
            threshold = std::stod(argv[i]);
            if (threshold <= 0.0 || threshold > 1.0) {
                std::cerr << "-t must be in (0, 1]\n"; return 1;
            }
        } else if (std::strcmp(argv[i], "--keep-lowercase") == 0) {
            lowercase_as_gap = false;
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
        using clk = std::chrono::steady_clock;
        using ms  = std::chrono::milliseconds;
        auto elapsed = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration_cast<ms>(b - a).count();
        };

        // --- 1. Parse alignment ---
        auto t0   = clk::now();
        auto seqs = parse_fasta(fasta_path);
        std::vector<std::string> names, raw;
        names.reserve(seqs.size());
        raw.reserve(seqs.size());
        for (auto& s : seqs) { names.push_back(s.name); raw.push_back(s.seq); }

        if (verbose)
            std::cerr << "[drinkme] loaded " << seqs.size()
                      << " sequences, length " << raw[0].size() << '\n';

        // --- 2. Cleanse columns ---
        auto t1 = clk::now();
        auto cl = cleanse_columns(raw, '-', lowercase_as_gap, threshold);
        if (verbose)
            std::cerr << "[drinkme] kept " << cl.kept_columns.size()
                      << "/" << raw[0].size() << " columns (threshold="
                      << threshold << ")\n";

        // --- 3. MCA ---
        auto t2 = clk::now();
        if (verbose)
            std::cerr << "[drinkme] MCA (k=" << n_components << ")...\n";

        auto mca_res = fit_mca(cl.seqs, n_components);

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

        // --- 4. Ward hierarchical clustering ---
        auto t3 = clk::now();
        if (verbose) std::cerr << "[drinkme] Ward clustering...\n";
        auto Z = ward_linkage(mca_res.coords);

        // --- 5. Newick output ---
        auto t4 = clk::now();
        if (verbose) std::cerr << "[drinkme] writing Newick tree...\n";
        std::cout << to_newick(Z, names) << '\n';

        if (verbose) {
            auto t5 = clk::now();
            std::cerr << "[drinkme] timings (ms):"
                      << "  parse="   << elapsed(t0, t1)
                      << "  cleanse=" << elapsed(t1, t2)
                      << "  mca="     << elapsed(t2, t3)
                      << "  ward="    << elapsed(t3, t4)
                      << "  newick="  << elapsed(t4, t5)
                      << "  total="   << elapsed(t0, t5) << '\n';
        }

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
