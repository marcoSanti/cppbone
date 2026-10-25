#include "netbone/graph.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace netbone {

std::uint64_t Graph::key(int u, int v) {
    if (u > v) std::swap(u, v);
    return (std::uint64_t(std::uint32_t(u)) << 32) | std::uint32_t(v);
}

int Graph::add_node(const std::string& label) {
    auto it = index_.find(label);
    if (it != index_.end()) return it->second;
    int id = (int)labels_.size();
    labels_.push_back(label);
    index_.emplace(label, id);
    adj_.emplace_back();
    node_attrs_.emplace_back();
    return id;
}

int Graph::add_edge(const std::string& u, const std::string& v, const Attrs& attrs) {
    int a = add_node(u);
    int b = add_node(v);
    return add_edge(a, b, attrs);
}

int Graph::add_edge(int u, int v, const Attrs& attrs) {
    int id = find_edge(u, v);
    if (id < 0) {
        id = (int)ends_.size();
        ends_.emplace_back(u, v);
        attrs_.emplace_back();
        edge_index_.emplace(key(u, v), id);
        adj_[u].emplace_back(v, id);
        if (u != v) adj_[v].emplace_back(u, id);
        ++n_edges_;
    }
    for (const auto& [k, val] : attrs) attrs_[id][k] = val;
    return id;
}

void Graph::remove_edge(int id) {
    auto [u, v] = ends_[id];
    auto drop = [id](std::vector<std::pair<int, int>>& a) {
        a.erase(std::find_if(a.begin(), a.end(), [id](const auto& p) { return p.second == id; }));
    };
    drop(adj_[u]);
    if (u != v) drop(adj_[v]);
    edge_index_.erase(key(u, v));
    --n_edges_;
}

int Graph::node(const std::string& label) const {
    auto it = index_.find(label);
    return it == index_.end() ? -1 : it->second;
}

int Graph::find_edge(int u, int v) const {
    auto it = edge_index_.find(key(u, v));
    return it == edge_index_.end() ? -1 : it->second;
}

std::vector<EdgeRef> Graph::edges() const {
    std::vector<EdgeRef> out;
    out.reserve(n_edges_);
    std::vector<bool> seen(ends_.size(), false);
    for (int u = 0; u < number_of_nodes(); ++u)
        for (auto [v, id] : adj_[u])
            if (!seen[id]) {
                seen[id] = true;
                out.push_back({u, v, id});
            }
    return out;
}

double Graph::degree(int u, const std::string& weight) const {
    double s = 0;
    for (auto [v, id] : adj_[u]) {
        auto it = attrs_[id].find(weight);
        double w = it == attrs_[id].end() ? 1.0 : it->second;
        s += v == u ? 2 * w : w;  // networkx counts self-loops twice
    }
    return s;
}

double Graph::get(int id, const std::string& key) const {
    auto it = attrs_[id].find(key);
    return it == attrs_[id].end() ? NaN : it->second;
}

std::vector<double> Graph::edge_values(const std::string& key, double missing) const {
    std::vector<double> out(ends_.size(), missing);
    for (size_t id = 0; id < ends_.size(); ++id) {
        auto it = attrs_[id].find(key);
        if (it != attrs_[id].end()) out[id] = it->second;
    }
    return out;
}

double Graph::node_get(int u, const std::string& key) const {
    auto it = node_attrs_[u].find(key);
    return it == node_attrs_[u].end() ? NaN : it->second;
}

Graph Graph::subgraph(const std::vector<bool>& keep) const {
    Graph g;
    for (int u = 0; u < number_of_nodes(); ++u)
        if (keep[u]) {
            g.add_node(labels_[u]);
            g.node_attrs_.back() = node_attrs_[u];
        }
    for (const auto& e : edges())
        if (keep[e.u] && keep[e.v]) g.add_edge(labels_[e.u], labels_[e.v], attrs_[e.id]);
    return g;
}

static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    if (!line.empty() && line.back() == ',') out.emplace_back();
    return out;
}

static double parse_value(const std::string& s) {
    if (s.empty()) return NaN;
    if (s == "True" || s == "true") return 1.0;
    if (s == "False" || s == "false") return 0.0;
    size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) throw std::invalid_argument("non-numeric attribute value: " + s);
    return v;
}

Graph Graph::read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("empty csv: " + path);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    auto header = split(line);
    int src = -1, dst = -1;
    for (int i = 0; i < (int)header.size(); ++i) {
        if (header[i] == "source") src = i;
        if (header[i] == "target") dst = i;
    }
    if (src < 0 || dst < 0) throw std::runtime_error("csv needs 'source' and 'target' columns");
    Graph g;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto cells = split(line);
        if (cells.size() != header.size()) throw std::runtime_error("bad csv row: " + line);
        Attrs attrs;
        for (int i = 0; i < (int)cells.size(); ++i)
            if (i != src && i != dst) attrs[header[i]] = parse_value(cells[i]);
        g.add_edge(cells[src], cells[dst], attrs);
    }
    return g;
}

std::string format_double(double x) {
    if (std::isnan(x)) return "";
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof buf, x);
    return std::string(buf, res.ptr);
}

void Graph::to_csv(std::ostream& out) const {
    std::set<std::string> cols;
    for (const auto& e : edges())
        for (const auto& kv : attrs_[e.id]) cols.insert(kv.first);
    out << "source,target";
    for (const auto& c : cols) out << ',' << c;
    out << '\n';
    for (const auto& e : edges()) {
        out << labels_[e.u] << ',' << labels_[e.v];
        for (const auto& c : cols) out << ',' << format_double(get(e.id, c));
        out << '\n';
    }
}

void Graph::nodes_to_csv(std::ostream& out) const {
    std::set<std::string> cols;
    for (const auto& a : node_attrs_)
        for (const auto& kv : a) cols.insert(kv.first);
    out << "node";
    for (const auto& c : cols) out << ',' << c;
    out << '\n';
    for (int u = 0; u < number_of_nodes(); ++u) {
        out << labels_[u];
        for (const auto& c : cols) out << ',' << format_double(node_get(u, c));
        out << '\n';
    }
}

bool label_less(const std::string& a, const std::string& b) {
    char* ea;
    char* eb;
    double x = std::strtod(a.c_str(), &ea);
    double y = std::strtod(b.c_str(), &eb);
    if (!a.empty() && !b.empty() && *ea == '\0' && *eb == '\0') return x < y;
    return a < b;
}

}  // namespace netbone
