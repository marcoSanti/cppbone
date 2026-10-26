/// @file
/// @brief Proximity / distance conversions (distanceclosure/utils.py).

#include <cmath>

#include "cppbone/cppbone.hpp"

namespace cppbone {

double prox2dist(double p) { return p == 0 ? INF : 1 / p - 1; }

double dist2prox(double d) { return d == INF ? 0 : 1 / (d + 1); }

}  // namespace cppbone
