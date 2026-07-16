// components.hpp — weakly connected components + odgi's component packing.
//
// Ported (not vendored) from odgi src/algorithms/weakly_connected_components.cpp
// and the packing loop in src/subcommand/layout_main.cpp (MIT, Erik Garrison
// et al.). The algorithm is odgi's; the data structures are ours.
//
// Why ported rather than copied verbatim:
//
//   - odgi accumulates ska::flat_hash_set<nid_t> per component and a
//     flat_hash_set<handle_t> of everything traversed. We already have XP's
//     dense rank space (0..N-1), so a flat vector<int32_t> of component ids is
//     both simpler and far leaner: chr1 is 10.6M nodes, which is ~42 MB of
//     vector against hash sets holding every node id twice over.
//
//   - odgi's coord_range_2d_t initialises max_x/max_y to
//     std::numeric_limits<double>::min(), which is the smallest POSITIVE double
//     (~2.2e-308), not the most negative. include() then never updates max for
//     an all-negative component. That is a bug, and it is live for us -- chr1's
//     layout has min_x = -17,696,844. We use lowest().
//
// What the packing does (odgi layout_main.cpp:400-432): each weak component gets
// its bounding box measured, then is offset so the components stack vertically
// separated by a border, instead of piling on top of each other at the origin.

#pragma once

#include "xp.hpp"

#include <gbwtgraph/gbwtgraph.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace gbz2layout {

// Fills `component[rank]` with a component index in discovery order (XP rank
// order, which follows GBWTGraph's for_each_handle). Returns the count.
// Components are weak: edges are followed in both directions, orientation
// ignored, exactly as odgi does.
std::uint64_t weakly_connected_components(const gbwtgraph::GBWTGraph& graph,
                                          const XP& xp,
                                          std::vector<std::int32_t>& component);

// odgi's packing: measure each component's bbox and shift it so components form
// a vertical stack separated by `border`. Mutates X/Y in place. No-op for a
// single component beyond normalising it against the border.
void pack_components(std::uint64_t ncomp,
                     const std::vector<std::int32_t>& component,
                     std::vector<std::atomic<double>>& X,
                     std::vector<std::atomic<double>>& Y,
                     double border = 1000.0);

} // namespace gbz2layout
