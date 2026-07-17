// sgd_minibatch_gpu.hpp — GPU backend for the whole-path minibatch update.
//
// CUDA is isolated behind a plain-pointer interface so the heavy templated
// headers (sdsl/gbwt) never go through nvcc. The C++ orchestrator (epoch loop +
// GBWT path extraction) stays in sgd_minibatch.cpp; per group it hands the
// extracted batch to run_group(), which launches the update kernel over the
// coordinate arrays resident in VRAM.

#pragma once

#include <cstdint>
#include <vector>

namespace gbz2layout {

// Opaque GPU context. Coordinates (X, Y, each length 2*node_count) live in VRAM
// for the whole run; only the small per-group batch arrays are streamed in.
struct GpuLayout {
    void* impl = nullptr;

    // allocate VRAM state; upload node seq lengths + the Zipf zeta table.
    void init(std::uint64_t node_count, std::uint64_t seed, std::uint64_t nthreads,
              const std::uint32_t* seq_lengths,
              const std::vector<double>& zetas, std::uint64_t space, std::uint64_t space_max,
              std::uint64_t space_quantization_step, double theta);
    void set_coords(const double* X, const double* Y);          // host -> VRAM (2*N each)
    void get_coords(double* X, double* Y) const;                // VRAM -> host

    // one group of whole paths (flat arrays): run `updates` Hogwild term-updates
    // at learning rate `eta`.
    void run_group(const std::uint32_t* rank, const std::uint64_t* pos, const std::uint8_t* rev,
                   const std::uint64_t* start, std::uint32_t npaths, std::uint64_t total,
                   double eta, bool cooling, std::uint64_t updates);

    void destroy();
    static bool available();                                    // a usable CUDA device exists
};

} // namespace gbz2layout
