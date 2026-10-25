#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "netbone/graph.hpp"

namespace netbone {

enum class FilterOn { Edges, Nodes };

struct Backbone;

// All filters share one signature so they can be compared and stored, like the Python function objects.
using Filter = Graph (*)(const Backbone& backbone, double value, bool narrate, const std::string& secondary_property,
                         bool secondary_property_ascending);

// filters.py. `value` is ignored by boolean_filter. An incompatible filter prints the accepted ones and returns an
// empty graph (Python returns None).
Graph boolean_filter(const Backbone& backbone, double value = 0, bool narrate = true,
                     const std::string& secondary_property = "weight", bool secondary_property_ascending = false);
Graph threshold_filter(const Backbone& backbone, double value, bool narrate = true,
                       const std::string& secondary_property = "weight", bool secondary_property_ascending = false);
Graph fraction_filter(const Backbone& backbone, double value, bool narrate = true,
                      const std::string& secondary_property = "weight", bool secondary_property_ascending = false);
std::string filter_name(Filter f);

struct Backbone {
    Graph graph;
    std::string method_name;
    std::string property_name;
    bool ascending = false;
    std::vector<Filter> filters;
    FilterOn filter_on = FilterOn::Edges;

    void to_csv(std::ostream& out) const;  // to_dataframe
    void narrate() const;
    const std::vector<Filter>& compatible_filters() const { return filters; }
};

}  // namespace netbone
