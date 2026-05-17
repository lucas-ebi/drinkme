#pragma once
// Column-level MSA cleansing: drops alignment columns whose gap fraction
// exceeds (1 - threshold).  Lowercase residues (HMMER insert states) can
// optionally be counted as gaps.  Sequences are never removed.

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

struct CleanseResult {
    std::vector<std::string> seqs;          // same order as input
    std::vector<int>         kept_columns;  // 0-based indices of surviving columns
};

inline CleanseResult cleanse_columns(
    const std::vector<std::string>& seqs,
    char   indel           = '-',
    bool   lowercase_as_gap = true,
    double threshold        = 0.9)
{
    if (seqs.empty()) throw std::runtime_error("empty sequence set");
    const int N = static_cast<int>(seqs.size());
    const int L = static_cast<int>(seqs[0].size());

    auto is_gap = [&](char c) -> bool {
        if (c == indel) return true;
        if (lowercase_as_gap && std::islower(static_cast<unsigned char>(c))) return true;
        return false;
    };

    const int min_non_gap = static_cast<int>(std::ceil(threshold * N));

    std::vector<int> kept;
    kept.reserve(L);
    for (int col = 0; col < L; ++col) {
        int non_gap = 0;
        for (const auto& s : seqs)
            if (!is_gap(s[col])) ++non_gap;
        if (non_gap >= min_non_gap)
            kept.push_back(col);
    }

    if (kept.empty())
        throw std::runtime_error(
            "all columns were filtered out; consider lowering --threshold");

    std::vector<std::string> out(N);
    for (int i = 0; i < N; ++i) {
        out[i].reserve(kept.size());
        for (int col : kept)
            out[i] += static_cast<char>(std::toupper(
                static_cast<unsigned char>(seqs[i][col])));
    }

    return {std::move(out), std::move(kept)};
}
