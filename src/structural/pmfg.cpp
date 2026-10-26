/// @file
/// @brief Planar Maximally Filtered Graph and the Left-Right planarity test (structural/pmfg.py).

#include <algorithm>
#include <functional>

#include "cppbone/cppbone.hpp"

namespace cppbone {

namespace {

/// @brief Left-Right planarity test (Brandes' formulation), ported from the testing phase of networkx LRPlanarity.
///
/// Edges are oriented once, so an undirected edge id doubles as its oriented-edge id; -1 plays Python's None.
/// shortcut: recursive DFS, depth = number of nodes; switch to the iterative networkx variant for huge graphs.
struct LRPlanarity {
    /// @brief Interval of return edges, given by its lowest and highest edge (-1 for none).
    struct Interval {
        int low = -1;   ///< Lowest return edge.
        int high = -1;  ///< Highest return edge.

        /// @brief Tells whether the interval holds no edge.
        /// @return true if both ends are unset.
        bool empty() const { return low < 0 && high < 0; }
    };

    /// @brief Pair of intervals whose edges must go on opposite sides.
    struct ConflictPair {
        Interval left;   ///< Edges on the left side.
        Interval right;  ///< Edges on the right side.

        /// @brief Exchanges the two sides.
        void swap() { std::swap(left, right); }
    };

    int n;                                              ///< Number of nodes.
    std::vector<std::vector<std::pair<int, int>>> adj;  ///< (neighbour, edge id) per node, self-loops removed.
    std::vector<int> head;           ///< Head node of every oriented edge.
    std::vector<int> tail;           ///< Tail node of every oriented edge.
    std::vector<int> height;         ///< DFS depth of every node, -1 before it is visited.
    std::vector<int> parent_edge;    ///< DFS tree edge into every node, -1 for roots.
    std::vector<int> lowpt;          ///< Lowest height reachable through an edge's subtree and one back edge.
    std::vector<int> lowpt2;         ///< Second lowest such height.
    std::vector<int> nesting_depth;  ///< Order key of the out-edges.
    std::vector<int> lowpt_edge;     ///< Return edge that reaches lowpt.
    std::vector<int> ref;            ///< Edge whose side decides this edge's side.
    std::vector<int> stack_bottom;   ///< Size of S when an edge was entered.
    std::vector<bool> oriented;      ///< Whether an edge has been oriented by the DFS.
    std::vector<std::vector<int>> out;  ///< Oriented out-edges per node.
    std::vector<ConflictPair> S;        ///< Stack of conflict pairs.

    /// @brief Tells whether an interval conflicts with an edge.
    /// @param i Interval.
    /// @param b Edge.
    /// @return true if the interval is non-empty and its highest edge returns above b's lowpoint.
    bool conflicting(const Interval& i, int b) const { return !i.empty() && lowpt[i.high] > lowpt[b]; }

    /// @brief Lowest lowpoint among the return edges of a conflict pair.
    /// @param p Conflict pair.
    /// @return The smallest lowpt of its two lowest edges.
    int lowest(const ConflictPair& p) const {
        if (p.left.empty()) return lowpt[p.right.low];
        if (p.right.empty()) return lowpt[p.left.low];
        return std::min(lowpt[p.left.low], lowpt[p.right.low]);
    }

    /// @brief First DFS: orients the edges and computes heights, lowpoints and nesting depths.
    /// @param v Node being visited.
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

    /// @brief Merges the return edges of out-edge ei into the constraints of its parent edge e.
    /// @param ei Out-edge just processed.
    /// @param e Parent edge of ei's tail.
    /// @return false if the constraints cannot be satisfied (the graph is not planar).
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

    /// @brief Drops the return edges that end at e's tail and records which side e is on.
    /// @param e Tree edge being left.
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

    /// @brief Second DFS: checks the side constraints of every edge.
    /// @param v Node being visited.
    /// @return false as soon as a constraint fails (the graph is not planar).
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

    /// @brief Runs the whole test on a graph.
    /// @param g Graph; self-loops are ignored.
    /// @return true if g is planar.
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
