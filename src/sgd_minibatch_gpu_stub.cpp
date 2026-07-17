// sgd_minibatch_gpu_stub.cpp — CUDA-free stand-in for GpuLayout.
//
// Links in place of sgd_minibatch_gpu.o for hosts without CUDA (e.g. cluster
// compute nodes). available() reports no device, so the minibatch stays on the
// CPU path; the other methods are never reached unless --gpu is forced, in
// which case they abort loudly rather than silently mislead. The GBZ export
// path (--export-gbz / --export-all-gbz) touches none of this.

#include "sgd_minibatch_gpu.hpp"

#include <cstdlib>
#include <iostream>

namespace gbz2layout {

static void nope(const char* fn) {
    std::cerr << "[gpu-stub] " << fn << " called but this build has no CUDA support; "
                 "rebuild with `make tool` on a CUDA host to use --gpu\n";
    std::abort();
}

void GpuLayout::init(std::uint64_t, std::uint64_t, std::uint64_t, const std::uint32_t*,
                     const std::vector<double>&, std::uint64_t, std::uint64_t,
                     std::uint64_t, double) { nope("init"); }
void GpuLayout::set_coords(const double*, const double*) { nope("set_coords"); }
void GpuLayout::get_coords(double*, double*) const { nope("get_coords"); }
void GpuLayout::run_group(const std::uint32_t*, const std::uint64_t*, const std::uint8_t*,
                          const std::uint64_t*, std::uint32_t, std::uint64_t,
                          double, bool, std::uint64_t) { nope("run_group"); }
void GpuLayout::destroy() { /* nothing allocated */ }
bool GpuLayout::available() { return false; }

} // namespace gbz2layout
