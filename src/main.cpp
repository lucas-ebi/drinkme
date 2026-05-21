// DrinkMe – hierarchical clustering of MSAs using MCA + Ward linkage.
//
// Pipeline: FASTA MSA → column cleansing → cardinality-normalised MCA
//           → Ward dendrogram → Newick tree (stdout)
//
// Usage:
//   drinkme <alignment.fasta> [options]

#include "agglomerative.hpp"
#include "fasta.hpp"

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
    int         n_components     = 2;
    double      threshold        = 0.5;
    bool        lowercase_as_gap = true;
    bool        verbose          = false;

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

        auto t0   = clk::now();
        auto seqs = parse_fasta(fasta_path);
        std::vector<std::string> names, raw;
        names.reserve(seqs.size());
        raw.reserve(seqs.size());
        for (auto& s : seqs) { names.push_back(s.name); raw.push_back(s.seq); }

        if (verbose)
            std::cerr << "[drinkme] loaded " << seqs.size()
                      << " sequences, length " << raw[0].size() << '\n';

        auto t1 = clk::now();
        std::cout << agglomerative_newick(raw, names, '-', lowercase_as_gap,
                                          threshold, n_components, verbose) << '\n';

        if (verbose)
            std::cerr << "[drinkme] total=" << elapsed(t0, clk::now()) << "ms"
                      << "  parse=" << elapsed(t0, t1) << "ms\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
