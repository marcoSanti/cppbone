/// @file
/// @brief Graph implementation: storage, parallel copy, CSV input and output.

#include "cppbone/graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace cppbone {

std::uint64_t Graph::key(int u, int v) {
    if (u > v) {
        std::swap(u, v);
    }
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32) |
           static_cast<std::uint32_t>(v);
}

/// @brief Element count below which copying or freeing a graph stays single-threaded (threads would
/// cost more).
static constexpr int PARALLEL_MIN = 1 << 16;

Graph::Graph(const Graph &other) { *this = other; }

Graph &Graph::operator=(const Graph &other) {
    if (this == &other) {
        return *this;
    }
    const int n = other.number_of_nodes();
    const int m = static_cast<int>(other.attrs_.size());
    adj_.assign(n, {});
    node_attrs_.assign(n, {});
    attrs_.assign(m, {});
    edge_index_.resize(EDGE_SHARDS); // a moved-from graph has none
    n_edges_ = other.n_edges_;
#pragma omp parallel if (m + n > PARALLEL_MIN)
    {
        // one thread copies the containers that cannot be split while the others copy the
        // per-element ones
#pragma omp single nowait
        {
            labels_ = other.labels_;
            index_  = other.index_;
            ends_   = other.ends_;
        }
#pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < EDGE_SHARDS; ++i) {
            edge_index_[i] = other.edge_index_[i];
        }
#pragma omp for schedule(dynamic, 4096) nowait
        for (int i = 0; i < m; ++i) {
            attrs_[i] = other.attrs_[i];
        }
#pragma omp for schedule(dynamic, 1024)
        for (int u = 0; u < n; ++u) {
            adj_[u] = other.adj_[u], node_attrs_[u] = other.node_attrs_[u];
        }
    }
    return *this;
}

Graph::~Graph() {
    const int m = static_cast<int>(attrs_.size());
#pragma omp parallel for schedule(dynamic, 4096) if (m > PARALLEL_MIN)
    for (int i = 0; i < m; ++i) {
        Attrs().swap(attrs_[i]);
    }
}

int Graph::add_node(const std::string &label) {
    if (const auto it = index_.find(label); it != index_.end()) {
        return it->second;
    }
    int id = static_cast<int>(labels_.size());
    labels_.push_back(label);
    index_.emplace(label, id);
    adj_.emplace_back();
    node_attrs_.emplace_back();
    return id;
}

int Graph::add_edge(const std::string &u, const std::string &v, const Attrs &attrs) {
    const int a = add_node(u);
    const int b = add_node(v);
    return add_edge(a, b, attrs);
}

int Graph::add_edge(int u, int v, const Attrs &attrs) {
    int id = find_edge(u, v);
    if (id < 0) {
        id = static_cast<int>(ends_.size());
        ends_.emplace_back(u, v);
        attrs_.emplace_back();
        std::uint64_t k = key(u, v);
        if (edge_index_.empty()) {
            edge_index_.resize(EDGE_SHARDS); // moved-from graph
        }
        edge_index_[shard(k)].emplace(k, id);
        adj_[u].emplace_back(v, id);
        if (u != v) {
            adj_[v].emplace_back(u, id);
        }
        ++n_edges_;
    }
    for (const auto &[k, val] : attrs) {
        attrs_[id][k] = val;
    }
    return id;
}

void Graph::remove_edge(int id) {
    auto [u, v] = ends_[id];
    auto drop   = [id](std::vector<std::pair<int, int>> &a) {
        a.erase(std::find_if(a.begin(), a.end(), [id](const auto &p) { return p.second == id; }));
    };
    drop(adj_[u]);
    if (u != v) {
        drop(adj_[v]);
    }
    const std::uint64_t k = key(u, v);
    edge_index_[shard(k)].erase(k);
    --n_edges_;
}

void Graph::remove_edges(const std::vector<char> &drop) {
    const int n      = number_of_nodes();
    const int shards = static_cast<int>(edge_index_.size());
    int removed      = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : removed)
    for (int i = 0; i < shards; ++i) {
        for (auto it = edge_index_[i].begin(); it != edge_index_[i].end();) {
            if (drop[it->second]) {
                it = edge_index_[i].erase(it), ++removed;
            } else {
                ++it;
            }
        }
    }
    n_edges_ -= removed;
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        auto &a = adj_[u];
        a.erase(
            std::remove_if(a.begin(), a.end(), [&](const auto &p) { return drop[p.second] != 0; }),
            a.end());
    }
}

int Graph::node(const std::string &label) const {
    const auto it = index_.find(label);
    return it == index_.end() ? -1 : it->second;
}

int Graph::find_edge(const int u, const int v) const {
    if (edge_index_.empty()) {
        return -1; // moved-from graph
    }
    const std::uint64_t k = key(u, v);
    const auto &index     = edge_index_[shard(k)];
    const auto it         = index.find(k);
    return it == index.end() ? -1 : it->second;
}

std::vector<EdgeRef> Graph::edges() const {
    std::vector<EdgeRef> out;
    out.reserve(n_edges_);
    std::vector seen(ends_.size(), false);
    for (int u = 0; u < number_of_nodes(); ++u) {
        for (auto [v, id] : adj_[u]) {
            if (!seen[id]) {
                seen[id] = true;
                out.push_back({u, v, id});
            }
        }
    }
    return out;
}

double Graph::degree(int u, const std::string &weight) const {
    double s = 0;
    for (auto [v, id] : adj_[u]) {
        auto it        = attrs_[id].find(weight);
        const double w = it == attrs_[id].end() ? 1.0 : it->second;
        s += v == u ? 2 * w : w; // networkx counts self-loops twice
    }
    return s;
}

double Graph::get(const int id, const std::string &key) const {
    const auto it = attrs_[id].find(key);
    return it == attrs_[id].end() ? NaN : it->second;
}

std::vector<double> Graph::edge_values(const std::string &key, double missing) const {
    std::vector out(ends_.size(), missing);
    const int m = static_cast<int>(ends_.size());
#pragma omp parallel for schedule(static)
    for (int id = 0; id < m; ++id) {
        if (auto it = attrs_[id].find(key); it != attrs_[id].end()) {
            out[id] = it->second;
        }
    }
    return out;
}

double Graph::node_get(int u, const std::string &key) const {
    const auto it = node_attrs_[u].find(key);
    return it == node_attrs_[u].end() ? NaN : it->second;
}

Graph Graph::subgraph(const std::vector<bool> &keep) const {
    Graph g;
    for (int u = 0; u < number_of_nodes(); ++u) {
        if (keep[u]) {
            g.add_node(labels_[u]);
            g.node_attrs_.back() = node_attrs_[u];
        }
    }
    for (const auto &[u, v, id] : edges()) {
        if (keep[u] && keep[v]) {
            g.add_edge(labels_[u], labels_[v], attrs_[id]);
        }
    }
    return g;
}

/// @brief Splits a CSV line on every comma; quotes are not interpreted.
/// @param line Line without its terminator.
/// @return The fields: "a," gives {"a", ""} and an empty line gives no fields.
static std::vector<std::string_view> split(const std::string_view line) {
    std::vector<std::string_view> out;
    if (line.empty()) {
        return out;
    }
    size_t start = 0;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            out.push_back(line.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

/// @brief Converts a CSV field to an attribute value.
/// @param s Field text.
/// @return NaN for an empty field, 1 / 0 for "True" / "False" (either case of the first letter),
/// else the number.
/// @throws std::invalid_argument if the field is not numeric.
static double parse_value(const std::string &s) {
    if (s.empty()) {
        return NaN;
    }
    if (s == "True" || s == "true") {
        return 1.0;
    }
    if (s == "False" || s == "false") {
        return 0.0;
    }
    size_t pos     = 0;
    const double v = std::stod(s, &pos);
    if (pos != s.size()) {
        throw std::invalid_argument("non-numeric attribute value: " + s);
    }
    return v;
}

/// @brief Reads a whole file and splits it into lines.
/// @param path File to read.
/// @param text Receives the file content; the returned views point into it.
/// @return The lines without their "\n" or "\r\n" terminators.
/// @throws std::runtime_error if the file cannot be opened.
static std::vector<std::string_view> read_lines(const std::string &path, std::string &text) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    text = ss.str();
    std::vector<std::string_view> lines;
    std::string_view all(text);
    size_t start = 0;
    while (start < all.size()) {
        size_t end = all.find('\n', start);
        if (end == std::string_view::npos) {
            end = all.size();
        }
        std::string_view line = all.substr(start, end - start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        lines.push_back(line);
        start = end + 1;
    }
    return lines;
}

/// @brief One parsed CSV line, waiting to be inserted into the graph.
struct CsvRow {
    std::string_view source; ///< Source label (points into the file text).
    std::string_view target; ///< Target label (points into the file text).
    Attrs attrs;             ///< Edge attributes parsed from the other columns.
    bool skip = false;       ///< The line was blank.
};

/// @brief Parses CSV lines in parallel.
/// @tparam Parse Callable (std::string_view line, CsvRow& row) that fills the row or throws.
/// @param lines All lines of the file.
/// @param first Index of the first data line (1 to skip a header).
/// @param parse Parser for one non-blank line.
/// @return One row per line from `first` on; blank lines are marked skip.
/// @throws The exception of the earliest bad line, whatever order the threads ran in.
template <typename Parse>
static std::vector<CsvRow> parse_rows(const std::vector<std::string_view> &lines, size_t first,
                                      Parse parse) {
    const int count = static_cast<int>(lines.size() - std::min(first, lines.size()));
    std::vector<CsvRow> rows(count);
    int bad = count;
    std::exception_ptr error;
#pragma omp parallel for schedule(dynamic, 4096)
    for (int i = 0; i < count; ++i) {
        std::string_view line = lines[first + i];
        if (line.empty()) {
            rows[i].skip = true;
            continue;
        }
        try {
            parse(line, rows[i]);
        } catch (...) {
#pragma omp critical(cppbone_read_csv)
            if (i < bad) {
                bad = i, error = std::current_exception();
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
    return rows;
}

void Graph::add_rows(std::vector<CsvRow> &rows) {
    ends_.reserve(rows.size());
    attrs_.reserve(rows.size());
    for (auto &index : edge_index_) {
        index.reserve(rows.size() / EDGE_SHARDS + 1);
    }
    for (auto &[source, target, attrs, skip] : rows) {
        if (skip) {
            continue;
        }
        const int u =
            add_node(std::string(source)); // separate statements: source first fixes the node order
        const int v = add_node(std::string(target));
        if (const int id = add_edge(u, v); attrs_[id].empty()) {
            attrs_[id] = std::move(attrs);
        } else {
            for (const auto &[k, val] : attrs) {
                attrs_[id][k] = val;
            }
        }
        Attrs().swap(attrs);
    }
}

Graph Graph::read_csv(const std::string &path) {
    std::string text;
    const auto lines = read_lines(path, text);
    if (lines.empty()) {
        throw std::runtime_error("empty csv: " + path);
    }
    auto header_views = split(lines[0]);
    const std::vector<std::string> header(header_views.begin(), header_views.end());
    int src = -1, dst = -1;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        if (header[i] == "source") {
            src = i;
        }
        if (header[i] == "target") {
            dst = i;
        }
    }
    if (src < 0 || dst < 0) {
        throw std::runtime_error("csv needs 'source' and 'target' columns");
    }
    auto rows = parse_rows(lines, 1, [&](const std::string_view line, CsvRow &row) {
        const auto cells = split(line);
        if (cells.size() != header.size()) {
            throw std::runtime_error("bad csv row: " + std::string(line));
        }
        for (int i = 0; i < static_cast<int>(cells.size()); ++i) {
            if (i != src && i != dst) {
                row.attrs[header[i]] = parse_value(std::string(cells[i]));
            }
        }
        row.source = cells[src];
        row.target = cells[dst];
    });
    Graph g;
    g.add_rows(rows);
    return g;
}

Graph Graph::read_csv(const std::string &path, int source_col, int target_col, int weight_col) {
    if (source_col < 0 || target_col < 0 || source_col == target_col || weight_col == source_col ||
        weight_col == target_col) {
        throw std::invalid_argument(
            "read_csv: source, target and weight columns must be distinct and non-negative");
    }
    std::string text;
    const auto lines = read_lines(path, text);
    const int need   = std::max({source_col, target_col, weight_col}) + 1;
    auto rows        = parse_rows(lines, 0, [&](std::string_view line, CsvRow &row) {
        const auto cells = split(line);
        if (static_cast<int>(cells.size()) < need) {
            throw std::runtime_error("bad csv row: " + std::string(line));
        }
        if (weight_col >= 0) {
            row.attrs["weight"] = parse_value(std::string(cells[weight_col]));
        }
        row.source = cells[source_col];
        row.target = cells[target_col];
    });
    Graph g;
    g.add_rows(rows);
    return g;
}

std::string format_double(const double x) {
    if (std::isnan(x)) {
        return "";
    }
    // shortest round-trip digits; std::to_chars(double) needs macOS 13.3+ libc++
    char buf[64];
    for (int p = 15; p <= 17; ++p) {
        std::snprintf(buf, sizeof buf, "%.*g", p, x);
        if (std::strtod(buf, nullptr) == x) {
            break;
        }
    }
    return buf;
}

void Graph::to_csv(std::ostream &out) const {
    std::set<std::string> cols;
    for (const auto &e : edges()) {
        for (const auto &[fst, _] : attrs_[e.id]) {
            cols.insert(fst);
        }
    }
    out << "source,target";
    for (const auto &c : cols) {
        out << ',' << c;
    }
    out << '\n';
    for (const auto &[u, v, id] : edges()) {
        out << labels_[u] << ',' << labels_[v];
        for (const auto &c : cols) {
            out << ',' << format_double(get(id, c));
        }
        out << '\n';
    }
}

void Graph::nodes_to_csv(std::ostream &out) const {
    std::set<std::string> cols;
    for (const auto &a : node_attrs_) {
        for (const auto &[fst, _] : a) {
            cols.insert(fst);
        }
    }
    out << "node";
    for (const auto &c : cols) {
        out << ',' << c;
    }
    out << '\n';
    for (int u = 0; u < number_of_nodes(); ++u) {
        out << labels_[u];
        for (const auto &c : cols) {
            out << ',' << format_double(node_get(u, c));
        }
        out << '\n';
    }
}

bool label_less(const std::string &a, const std::string &b) {
    char *ea;
    char *eb;
    const double x = std::strtod(a.c_str(), &ea);
    const double y = std::strtod(b.c_str(), &eb);
    if (!a.empty() && !b.empty() && *ea == '\0' && *eb == '\0') {
        return x < y;
    }
    return a < b;
}

} // namespace cppbone
