#pragma once

/// @file
/// @brief The Graph type (stand-in for networkx.Graph and the pandas edge list) and its attribute
/// helpers.

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cppbone {

/// @brief Quiet NaN: the value of a missing attribute.
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
/// @brief Positive infinity: the distance to an unreachable node.
constexpr double INF = std::numeric_limits<double>::infinity();

/// @brief Attributes of a node or edge, by name.
///
/// Values are doubles; booleans are stored as 1.0 / 0.0 and a missing value reads as NaN (like
/// pandas).
using Attrs = std::map<std::string, double>;

/// @brief An edge as listed by Graph::edges(): its endpoints, in iteration order, and its id.
struct EdgeRef {
    int u;  ///< First endpoint (the node from whose adjacency the edge was listed).
    int v;  ///< Second endpoint.
    int id; ///< Edge id, the index of its attributes.
};

/// @brief One parsed CSV line (defined in graph.cpp, used by Graph::read_csv).
struct CsvRow;

/// @brief Undirected simple graph with string node labels, standing in for networkx.Graph and the
/// pandas edge list.
///
/// Nodes are numbered 0..n-1 in insertion order and edges get increasing ids. Node order, adjacency
/// order and edges() order follow networkx iteration order. Removed edges leave their id unused, so
/// per-edge arrays are sized with edge_capacity().
class Graph {
public:
    /// @brief Creates an empty graph.
    Graph() = default;
    /// @brief Copies a graph; large graphs are copied in parallel.
    /// @param other Graph to copy.
    Graph(const Graph &other);
    /// @brief Moves a graph; the source is left empty.
    Graph(Graph &&) noexcept = default;
    /// @brief Replaces this graph with a copy of another; large graphs are copied in parallel.
    /// @param other Graph to copy.
    /// @return This graph.
    Graph &operator=(const Graph &other);
    /// @brief Replaces this graph by moving another into it.
    /// @return This graph.
    Graph &operator=(Graph &&) noexcept = default;
    /// @brief Frees the graph; the per-edge attribute maps of a large graph are released in
    /// parallel.
    ~Graph();

    /// @brief Adds a node, or finds it if the label exists.
    /// @param label Node label.
    /// @return The node's index.
    int add_node(const std::string &label);

    /// @brief Adds an edge between two labels, creating the nodes as needed.
    ///
    /// Adding an existing edge updates its attributes, like networkx.
    /// @param u Label of the first endpoint.
    /// @param v Label of the second endpoint.
    /// @param attrs Attributes to set on the edge.
    /// @return The edge id.
    int add_edge(const std::string &u, const std::string &v, const Attrs &attrs = {});

    /// @brief Adds an edge between two existing nodes, or updates the attributes of the existing
    /// edge.
    /// @param u First endpoint.
    /// @param v Second endpoint (equal to u for a self-loop).
    /// @param attrs Attributes to set on the edge.
    /// @return The edge id.
    int add_edge(int u, int v, const Attrs &attrs = {});

    /// @brief Removes an edge; its id is not reused.
    /// @param id Id of an edge in the graph.
    void remove_edge(int id);

    /// @brief Removes many edges in one pass, with the same result as calling remove_edge on each.
    /// @param drop Indexed by edge id; the edges with a non-zero entry are removed.
    void remove_edges(const std::vector<char> &drop);

    /// @brief Returns the number of nodes.
    /// @return Node count.
    int number_of_nodes() const { return (int) labels_.size(); }

    /// @brief Returns the number of edges.
    /// @return Edge count (removed edges excluded).
    int number_of_edges() const { return n_edges_; }

    /// @brief Looks up a node by label.
    /// @param label Node label.
    /// @return The node's index, -1 if absent.
    int node(const std::string &label) const;

    /// @brief Returns a node's label.
    /// @param u Node index.
    /// @return The label.
    const std::string &label(int u) const { return labels_[u]; }

    /// @brief Looks up the edge between two nodes.
    /// @param u One endpoint.
    /// @param v The other endpoint.
    /// @return The edge id, -1 if absent.
    int find_edge(int u, int v) const;

    /// @brief Tells whether two nodes are adjacent.
    /// @param u One endpoint.
    /// @param v The other endpoint.
    /// @return true if the edge exists.
    bool has_edge(int u, int v) const { return find_edge(u, v) >= 0; }

    /// @brief Lists every edge once, in networkx order (by first endpoint, then adjacency order).
    /// @return The edges.
    std::vector<EdgeRef> edges() const;

    /// @brief Returns a node's adjacency list.
    /// @param u Node index.
    /// @return (neighbour, edge id) pairs in insertion order; a self-loop appears once.
    const std::vector<std::pair<int, int>> &adj(int u) const { return adj_[u]; }

    /// @brief Returns a node's degree.
    /// @param u Node index.
    /// @return Number of incident edges.
    int degree(int u) const { return (int) adj_[u].size(); }

    /// @brief Returns a node's strength.
    /// @param u Node index.
    /// @param weight Attribute to sum; a missing value counts as 1 and a self-loop twice, like
    /// networkx.
    /// @return Sum of the attribute over the incident edges.
    double degree(int u, const std::string &weight) const;

    /// @brief Returns an edge's endpoints.
    /// @param id Edge id.
    /// @return The endpoints in the order the edge was added.
    std::pair<int, int> ends(int id) const { return ends_[id]; }

    /// @brief Reads an edge attribute.
    /// @param id Edge id.
    /// @param key Attribute name.
    /// @return The value, NaN when missing.
    double get(int id, const std::string &key) const;

    /// @brief Writes an edge attribute.
    ///
    /// Safe to call concurrently for different edge ids.
    /// @param id Edge id.
    /// @param key Attribute name.
    /// @param value New value.
    void set(int id, const std::string &key, double value) { attrs_[id][key] = value; }

    /// @brief Returns all attributes of an edge.
    /// @param id Edge id.
    /// @return The attribute map.
    const Attrs &edge_attrs(int id) const { return attrs_[id]; }

    /// @brief Reads one attribute of every edge.
    /// @param key Attribute name.
    /// @param missing Value used where the attribute is absent.
    /// @return Values indexed by edge id (size edge_capacity()).
    std::vector<double> edge_values(const std::string &key, double missing = 1.0) const;

    /// @brief Returns the size of per-edge arrays: one past the largest edge id ever used.
    /// @return The capacity.
    int edge_capacity() const { return (int) ends_.size(); }

    /// @brief Reads a node attribute.
    /// @param u Node index.
    /// @param key Attribute name.
    /// @return The value, NaN when missing.
    double node_get(int u, const std::string &key) const;

    /// @brief Writes a node attribute.
    ///
    /// Safe to call concurrently for different nodes.
    /// @param u Node index.
    /// @param key Attribute name.
    /// @param value New value.
    void node_set(int u, const std::string &key, double value) { node_attrs_[u][key] = value; }

    /// @brief Returns all attributes of a node.
    /// @param u Node index.
    /// @return The attribute map.
    const Attrs &node_attrs(int u) const { return node_attrs_[u]; }

    /// @brief Induced subgraph on the kept nodes, preserving node order, edge order and attributes.
    /// @param keep Indexed by node; true keeps the node.
    /// @return The subgraph, with nodes renumbered.
    Graph subgraph(const std::vector<bool> &keep) const;

    /// @brief Reads an edge list with a header that has "source" and "target" columns.
    ///
    /// Every other column becomes a numeric edge attribute ("True"/"False" read as 1/0, empty as
    /// NaN). Fields are split on plain commas: quoted fields are not supported, so labels must not
    /// contain commas. Rows are parsed in parallel and inserted in file order.
    /// @param path CSV file.
    /// @return The graph.
    /// @throws std::runtime_error if the file cannot be read, lacks the columns or has a malformed
    /// row;
    ///         std::invalid_argument for a non-numeric attribute value.
    static Graph read_csv(const std::string &path);

    /// @brief Reads a header-less edge list, picking the columns by index.
    /// @param path CSV file.
    /// @param source_col 0-based column of the source labels.
    /// @param target_col 0-based column of the target labels.
    /// @param weight_col 0-based column stored as "weight", or -1 for an unweighted list. Other
    /// columns are ignored.
    /// @return The graph.
    /// @throws std::invalid_argument if the columns are negative or not distinct, or a weight is
    /// not numeric;
    ///         std::runtime_error if the file cannot be read or a row is too short.
    static Graph read_csv(const std::string &path, int source_col, int target_col,
                          int weight_col = 2);

    /// @brief Writes the edge list as CSV (networkx to_pandas_edgelist).
    ///
    /// Columns: source, target, then every edge attribute in alphabetical order; missing values are
    /// empty.
    /// @param out Destination stream.
    void to_csv(std::ostream &out) const;

    /// @brief Writes the node attribute table as CSV: a "node" column, then every attribute
    /// alphabetically.
    /// @param out Destination stream.
    void nodes_to_csv(std::ostream &out) const;

private:
    /// @brief Key of an unordered node pair in the edge index.
    /// @param u One endpoint.
    /// @param v The other endpoint.
    /// @return The smaller index in the high 32 bits, the larger in the low 32 bits.
    static std::uint64_t key(int u, int v);

    /// @brief Inserts the rows parsed by read_csv in file order, moving their attributes into the
    /// graph.
    /// @param rows Parsed rows; their attributes are consumed.
    void add_rows(std::vector<CsvRow> &rows);

    /// @brief Picks the edge-index shard of a key.
    /// @param k Key from key().
    /// @return Shard number in [0, EDGE_SHARDS).
    static int shard(std::uint64_t k) { return int((k * 0x9E3779B97F4A7C15ull) >> 56); }

    std::vector<std::string> labels_;                   ///< Label of every node.
    std::unordered_map<std::string, int> index_;        ///< Label to node index.
    std::vector<std::vector<std::pair<int, int>>> adj_; ///< (neighbour, edge id) lists per node.
    std::vector<Attrs> node_attrs_;                     ///< Attributes of every node.
    std::vector<std::pair<int, int>> ends_; ///< Endpoints of every edge id, removed ones included.
    std::vector<Attrs> attrs_;              ///< Attributes of every edge id.
    static constexpr int EDGE_SHARDS = 256; ///< Number of edge-index shards.
    /// @brief Node-pair key to edge id, sharded so large graphs can copy the shards in parallel.
    std::vector<std::unordered_map<std::uint64_t, int>> edge_index_ =
        std::vector<std::unordered_map<std::uint64_t, int>>(EDGE_SHARDS);
    int n_edges_ = 0; ///< Number of edges, removed ones excluded.
};

/// @brief Orders node labels like pandas: numerically when both parse as numbers, lexicographically
/// otherwise.
/// @param a First label.
/// @param b Second label.
/// @return true if a sorts before b.
bool label_less(const std::string &a, const std::string &b);

/// @brief Formats a double with the fewest digits (15 to 17) that read back to the same value.
/// @param x Value.
/// @return The text; empty for NaN.
std::string format_double(double x);

} // namespace cppbone
