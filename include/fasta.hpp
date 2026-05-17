#pragma once
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Sequence {
    std::string name;
    std::string seq;
};

inline std::vector<Sequence> parse_fasta(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open: " + path);

    std::vector<Sequence> result;
    std::string line, current_name, current_seq;

    auto flush = [&]() {
        if (!current_name.empty())
            result.push_back({std::move(current_name), std::move(current_seq)});
    };

    while (std::getline(in, line)) {
        if (line.empty() || (line.size() == 1 && line[0] == '\r')) continue;
        if (line.back() == '\r') line.pop_back();

        if (line[0] == '>') {
            flush();
            current_name = line.substr(1);
            // Keep only up to the first whitespace (standard FASTA id)
            auto ws = current_name.find_first_of(" \t");
            if (ws != std::string::npos) current_name.resize(ws);
        } else {
            for (char c : line) current_seq += static_cast<char>(std::toupper(c));
        }
    }
    flush();

    if (result.empty()) throw std::runtime_error("no sequences found in " + path);

    const std::size_t L = result[0].seq.size();
    for (const auto& s : result)
        if (s.seq.size() != L)
            throw std::runtime_error(
                "sequence '" + s.name + "' has length " +
                std::to_string(s.seq.size()) + " but expected " +
                std::to_string(L) + " — not a valid MSA");

    return result;
}
