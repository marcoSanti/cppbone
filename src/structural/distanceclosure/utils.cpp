#include <cmath>

#include "cppbone/cppbone.hpp"

namespace cppbone {

// Non-negative [0,1] proximity to [0,inf] distance: d = 1/p - 1.
double prox2dist(double p) { return p == 0 ? INF : 1 / p - 1; }

// Non-negative distance to [0,1] proximity: p = 1/(d+1).
double dist2prox(double d) { return d == INF ? 0 : 1 / (d + 1); }

}  // namespace cppbone
