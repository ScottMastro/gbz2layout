// compartment.hpp — balanced pinch-bounded decomposition for parallel layout.
//
// Cuts the backbone at a *subset* of path-pinches (nodes where all locally
// present haplotypes converge) chosen so each compartment holds ~equal work.
// Each compartment is a whole backbone stretch plus its bubbles, laid out
// jointly (full SGD, so it de-collides internally); only the chosen boundary
// pinches are frozen anchors, shared between neighbouring compartments.
//
// Produces the region[] / freeze[] arrays consumed by path_linear_sgd_layout:
//   region[rank] = compartment id, or -1 for a boundary-pinch anchor
//   freeze[rank] = true for boundary-pinch anchors (never move)

#pragma once

#include "xp.hpp"

#include <gbwtgraph/gbz.h>

#include <cstdint>
#include <vector>

namespace gbz2layout {

struct CompartmentResult {
    std::vector<std::int32_t> region;   // per rank: compartment id, -1 = anchor
    std::vector<bool>         freeze;    // per rank: boundary-pinch anchor
    std::vector<std::uint32_t> boundary_ranks;   // the frozen anchors, in ref order
    std::uint64_t n_compartments = 0;
    std::uint64_t n_pinches = 0;        // candidate pinches found
    // balance stats (compartment node counts)
    std::uint64_t min_nodes = 0, max_nodes = 0, median_nodes = 0;
};

// target_tasks: desired number of compartments (balance target).
// pinch_window: local-envelope half-window in reference nodes.
CompartmentResult build_compartments(const gbwtgraph::GBZ& gbz,
                                     const gbwtgraph::GBWTGraph& graph,
                                     const XP& xp,
                                     std::uint64_t target_tasks,
                                     std::uint64_t pinch_window,
                                     std::int64_t genome_min_id,
                                     const std::vector<bool>* node_mask,
                                     bool verbose);

} // namespace gbz2layout
