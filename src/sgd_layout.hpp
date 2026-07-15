// sgd_layout.hpp — port of odgi's path_linear_sgd_layout (CPU), adapted to run
// over a GBWT-backed XP and GBWTGraph. Snapshots/progress-meter/GPU stripped.
//
// The SGD math is unchanged from odgi. The only structural adaptations:
//   - path index is gbz2layout::XP (GBWT-backed) instead of odgi's xp::XP
//   - node->dense-index uses XP::rank_of_handle (ids are sparse), replacing
//     odgi's number_bool_packing::unpack_number(handle)
//   - per-path step counts come from the XP (the capped/retained set), not the
//     GBWTGraph (which only exposes the reference path)

#pragma once

#include "xp.hpp"

#include <gbwtgraph/gbwtgraph.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace gbz2layout {

struct SgdParams {
    std::uint64_t iter_max = 30;
    std::uint64_t iter_with_max_learning_rate = 0;
    std::uint64_t min_term_updates = 0;      // caller sets (default 10 * sum steps)
    double        delta = 0.0;               // early-stop threshold (0 = run all iters)
    double        eps = 0.01;
    double        eta_max = 0.0;             // caller sets (default max_steps^2)
    double        theta = 0.99;
    std::uint64_t space = 0;                 // caller sets (default max_steps)
    std::uint64_t space_max = 1000;
    std::uint64_t space_quantization_step = 100;
    double        cooling_start = 0.5;
    std::uint64_t nthreads = 1;
    bool          progress = true;
    std::uint64_t seed = 9399220;
    // optional: node ranks with pin_y[rank]==true are pulled toward the line
    // Y=0. pin_strength in [0,1]: 1 = hard pin (Y stays 0, X free), 0 = no pull
    // (free), in between = soft spring (reference tends toward the center line
    // but bubbles can still bend it). Used to keep the backbone ~linear.
    const std::vector<bool>* pin_y = nullptr;
    double pin_strength = 1.0;
};

// X, Y are each sized 2 * node_count (two endpoints per node), indexed by
// 2 * node_rank + endpoint. Caller initializes them (reference-anchored init).
void path_linear_sgd_layout(const gbwtgraph::GBWTGraph& graph,
                            const XP& path_index,
                            const SgdParams& params,
                            std::vector<std::atomic<double>>& X,
                            std::vector<std::atomic<double>>& Y);

} // namespace gbz2layout
