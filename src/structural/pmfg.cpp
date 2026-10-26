#include <algorithm>
#include <functional>

#include "cppbone/cppbone.hpp"

namespace cppbone {

namespace {

// Left-Right planarity test (Brandes' formulation), ported from the testing phase of networkx LRPlanarity.
// Edges are oriented once, so an undirected edge id doubles as its oriented-edge id; -1 plays Python's None.
// shortcut: recursive DFS, depth = number of nodes; switch to the iterative networkx variant for huge graphs.
struct LRPlanarity {
    struct Interval {
        int low = -1, high = -1;
        bool empty() const { return low < 0 && high < 0; }
    };
    struct ConflictPair {
        Interval left, right;
        void swap() { std::swap(left, right); }
    };

    int n;
    std::vector<std::vector<std::pair<int, int>>> adj;  // (neighbour, edge id)
    std::vector<int> head, tail, height, parent_edge, lowpt, lowpt2, nesting_depth, lowpt_edge, ref, stack_bottom;
    std::vector<bool> oriented;
    std::vector<std::vector<int>> out;  // oriented out-edges per node
    std::vector<ConflictPair> S;

    bool conflicting(const Interval& i, int b) const { return !i.empty() && lowpt[i.high] > lowpt[b]; }
    int lowest(const ConflictPair& p) const {
        if (p.left.empty()) return lowpt[p.right.low];
        if (p.right.empty()) return lowpt[p.left.low];
        return std::min(lowpt[p.left.low], lowpt[p.right.low]);
    }

    void dfs_orientation(int v) {
        int e = parent_edge[v];
        for (auto [w, vw] : adj[v]) {
            if (oriented[vw]) continue;  // the edge was already oriented
            oriented[vw] = true;
            tail[vw] = v;
            head[vw] = w;
            out[v].push_back(vw);
            lowpt[vw] = lowpt2[vw] = height[v];
            if (height[w] < 0) {  // tree edge
                parent_edge[w] = vw;
                height[w] = height[v] + 1;
                dfs_orientation(w);
            } else {  // back edge
                lowpt[vw] = height[w];
            }
            // determine nesting graph
            nesting_depth[vw] = 2 * lowpt[vw] + (lowpt2[vw] < height[v] ? 1 : 0);  // +1 when chordal
            // update lowpoints of parent edge e
            if (e >= 0) {
                if (lowpt[vw] < lowpt[e]) {
                    lowpt2[e] = std::min(lowpt[e], lowpt2[vw]);
                    lowpt[e] = lowpt[vw];
                } else if (lowpt[vw] > lowpt[e]) {
                    lowpt2[e] = std::min(lowpt2[e], lowpt[vw]);
                } else {
                    lowpt2[e] = std::min(lowpt2[e], lowpt2[vw]);
                }
            }
        }
    }

    bool add_constraints(int ei, int e) {
        ConflictPair P;
        // merge return edges of e_i into P.right
        do {
            ConflictPair Q = S.back();
            S.pop_back();
            if (!Q.left.empty()) Q.swap();
            if (!Q.left.empty()) return false;  // not planar
            if (lowpt[Q.right.low] > lowpt[e]) {  // merge intervals
                if (P.right.empty())
                    P.right = Q.right;
                else
                    ref[P.right.low] = Q.right.high;
                P.right.low = Q.right.low;
            } else {  // align
                ref[Q.right.low] = lowpt_edge[e];
            }
        } while ((int)S.size() != stack_bottom[ei]);
        // merge conflicting return edges of e_1,...,e_i-1 into P.left
        while (!S.empty() && (conflicting(S.back().left, ei) || conflicting(S.back().right, ei))) {
            ConflictPair Q = S.back();
            S.pop_back();
            if (conflicting(Q.right, ei)) Q.swap();
            if (conflicting(Q.right, ei)) return false;  // not planar
            // merge interval below lowpt(e_i) into P.right
            if (P.right.low >= 0) ref[P.right.low] = Q.right.high;
            if (Q.right.low >= 0) P.right.low = Q.right.low;
            if (P.left.empty())
                P.left = Q.left;
            else
                ref[P.left.low] = Q.left.high;
            P.left.low = Q.left.low;
        }
        if (!(P.left.empty() && P.right.empty())) S.push_back(P);
        return true;
    }

    void remove_back_edges(int e) {
        int u = tail[e];
        // trim back edges ending at parent u: drop entire conflict pairs
        while (!S.empty() && lowest(S.back()) == height[u]) S.pop_back();
        if (!S.empty()) {  // one more conflict pair to consider
            ConflictPair& P = S.back();
            // trim left interval
            while (P.left.high >= 0 && head[P.left.high] == u) P.left.high = ref[P.left.high];
            if (P.left.high < 0 && P.left.low >= 0) {  // just emptied
                ref[P.left.low] = P.right.low;
                P.left.low = -1;
            }
            // trim right interval
            while (P.right.high >= 0 && head[P.right.high] == u) P.right.high = ref[P.right.high];
            if (P.right.high < 0 && P.right.low >= 0) {  // just emptied
                ref[P.right.low] = P.left.low;
                P.right.low = -1;
            }
        }
        // side of e is side of a highest return edge
        if (lowpt[e] < height[u] && !S.empty()) {  // e has return edge
            int hl = S.back().left.high, hr = S.back().right.high;
            ref[e] = (hl >= 0 && (hr < 0 || lowpt[hl] > lowpt[hr])) ? hl : hr;
        }
    }

    bool dfs_testing(int v) {
        int e = parent_edge[v];
        for (int ei : out[v]) {
            int w = head[ei];
            stack_bottom[ei] = (int)S.size();
            if (ei == parent_edge[w]) {  // tree edge
                if (!dfs_testing(w)) return false;
            } else {  // back edge
                lowpt_edge[ei] = ei;
                S.push_back({{}, {ei, ei}});
            }
            // integrate new return edges
            if (lowpt[ei] < height[v]) {
                if (ei == out[v][0])  // e_i has return edge
                    lowpt_edge[e] = lowpt_edge[ei];
                else if (!add_constraints(ei, e))
                    return false;
            }
        }
        if (e >= 0) remove_back_edges(e);  // v isn't root
        return true;
    }

    bool run(const Graph& g) {
        n = g.number_of_nodes();
        int m = g.edge_capacity();
        adj.assign(n, {});
        int size = 0;
        for (int u = 0; u < n; ++u)
            for (auto [v, id] : g.adj(u))
                if (v != u) adj[u].emplace_back(v, id), size += u < v;
        if (n > 2 && size > 3 * n - 6) return false;
        head.assign(m, -1), tail.assign(m, -1), lowpt.assign(m, 0), lowpt2.assign(m, 0), nesting_depth.assign(m, 0);
        lowpt_edge.assign(m, -1), ref.assign(m, -1), stack_bottom.assign(m, 0), oriented.assign(m, false);
        height.assign(n, -1), parent_edge.assign(n, -1), out.assign(n, {});
        std::vector<int> roots;
        for (int v = 0; v < n; ++v)
            if (height[v] < 0) {
                height[v] = 0;
                roots.push_back(v);
                dfs_orientation(v);
            }
        // sort the adjacency lists by nesting depth
        for (int v = 0; v < n; ++v)
            std::stable_sort(out[v].begin(), out[v].end(),
                             [&](int a, int b) { return nesting_depth[a] < nesting_depth[b]; });
        for (int v : roots)
            if (!dfs_testing(v)) return false;
        return true;
    }
};

}  // namespace

bool is_planar(const Graph& g) { return LRPlanarity().run(g); }

Backbone pmfg(const Graph& data) {
    Graph g = data;
    auto rows = g.edges();
    for (const auto& e : rows) g.set(e.id, "in_backbone", false);
    auto w = g.edge_values("weight", NaN);
    std::stable_sort(rows.begin(), rows.end(), [&](const EdgeRef& a, const EdgeRef& b) { return w[a.id] > w[b.id]; });
    Graph backbone;
    for (const auto& row : rows) {
        int id = backbone.add_edge(g.label(row.u), g.label(row.v));
        if (!is_planar(backbone))
            backbone.remove_edge(id);
        else
            g.set(row.id, "in_backbone", true);
        if (backbone.number_of_edges() == 3 * (g.number_of_nodes() - 2)) break;
    }
    return {std::move(g), "Planar Maximally Filtered Graph", "weight", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace cppbone
