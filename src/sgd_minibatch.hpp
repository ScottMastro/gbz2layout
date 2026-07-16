// sgd_minibatch.hpp — PG-SGD layout with whole-path minibatching.
//
// Instead of a fixed per-node-capped step index, each SGD iteration streams a
// group of K WHOLE paths from the GBWT (intact local adjacency), runs its
// updates at that iteration's learning rate, then discards the group. Paths are
// consumed without replacement within a pass and reshuffled between passes, so
// every path (hence every node) is covered while peak memory stays bounded to
// ~K paths. This targets the cap-induced "icy" spikes: the per-node cap breaks
// path adjacency, minibatching preserves it at low peak memory.

#pragma once

#include "xp.hpp"

#include <gbwtgraph/gbwtgraph.h>
#include <gbwt/gbwt.h>

#include <atomic>
#include <cstdint>
#include <vector>

namespace gbz2layout {

struct MinibatchParams {
    std::uint64_t iter_max = 30;
    std::uint64_t batch_paths = 64;          // K whole paths resident per iteration
    std::uint64_t updates_mult = 10;         // updates/iter = mult * batch_step_count
    std::uint64_t iter_with_max_learning_rate = 0;
    double        eps = 0.01;
    double        eta_max = 0.0;             // caller sets (max_steps^2)
    double        theta = 0.99;
    std::uint64_t space = 0;                 // caller sets (max_steps)
    std::uint64_t space_max = 1000;
    std::uint64_t space_quantization_step = 100;
    double        cooling_start = 0.5;
    std::uint64_t nthreads = 1;
    bool          progress = true;
    std::uint64_t seed = 9399220;
    // sub-path sampling: if >0, the paired sample is drawn within this many
    // steps of the anchor along its path (overlapping length-L windows). Keeps
    // local adjacency, drops long-range pairs. 0 = whole-path (unbounded).
    std::uint64_t window_len = 0;
    bool          use_gpu = false;           // run the update on the GPU (odgi-derived kernel)
};

// X, Y sized 2*node_count (two endpoints/node), indexed 2*rank + endpoint.
// Caller initializes them (reference-anchored init).
void path_linear_sgd_layout_minibatch(const gbwtgraph::GBWTGraph& graph,
                                      const gbwt::GBWT& gbwt_index,
                                      const XP& xp,
                                      const MinibatchParams& params,
                                      std::vector<std::atomic<double>>& X,
                                      std::vector<std::atomic<double>>& Y);

} // namespace gbz2layout
