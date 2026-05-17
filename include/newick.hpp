#pragma once
// Converts a Ward linkage matrix into a Newick-format phylogenetic tree.
//
// Branch lengths: for each node the length to its parent is
//   branch = height(parent) − height(node)
// Leaves have height 0; internal node N+i has height Z[i].height.
// The root (node 2N-2) carries no branch annotation.

#include "linkage.hpp"
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// Quote a label if it contains Newick-reserved characters.
inline std::string newick_quote(const std::string& s) {
    for (char c : s)
        if (c == ' ' || c == '\t' || c == '(' || c == ')' ||
            c == ',' || c == ':' || c == ';' || c == '[' || c == ']')
            return '\'' + s + '\'';
    return s;
}

inline std::string to_newick(
    const std::vector<LinkageRow>& Z,
    const std::vector<std::string>& labels)
{
    const int N     = static_cast<int>(labels.size());
    const int total = 2 * N - 1;

    // Node heights: 0 for leaves, Z[i].height for internal node N+i
    std::vector<double> h(total, 0.0);
    for (int i = 0; i < N - 1; ++i)
        h[N + i] = Z[i].height;

    // Recursive Newick builder.
    // parent_h: height of this node's parent (used to compute branch length).
    // is_root:  if true, no branch annotation is appended.
    std::function<std::string(int, double, bool)> build =
        [&](int node, double parent_h, bool is_root) -> std::string {
        const double branch = parent_h - h[node];
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);

        if (node < N) {
            // Leaf
            oss << newick_quote(labels[node]) << ':' << branch;
            return oss.str();
        }

        const int    idx    = node - N;
        const double node_h = h[node];
        const std::string left  = build(Z[idx].left,  node_h, false);
        const std::string right = build(Z[idx].right, node_h, false);

        oss << '(' << left << ',' << right << ')';
        if (!is_root) oss << ':' << branch;
        return oss.str();
    };

    // Root = node 2N-2; pass its own height so branch = 0 and is_root suppresses it.
    return build(2 * N - 2, h[2 * N - 2], true) + ';';
}
