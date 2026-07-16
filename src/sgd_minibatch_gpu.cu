// sgd_minibatch_gpu.cu — GPU backend for the whole-path minibatch.
//
// The device kernel, coalesced curand RNG, and device Zipf sampler are adapted
// from odgi's src/cuda/layout.cu (Li et al., "Rapid GPU-Based Pangenome Graph
// Layout"), which implements the identical PG-SGD update our CPU port already
// runs. The difference is the FEEDING: odgi holds all paths resident in VRAM;
// here the node coordinates stay resident but each group's path_data is streamed
// in per launch, so peak VRAM scales with K paths, not haplotype depth. A
// window_len knob (our sub-path sampling) bounds the paired-sample distance.

#include "sgd_minibatch_gpu.hpp"

#include "third_party/dirty_zipfian_int_distribution.h"   // host-side zeta build only

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <curand_kernel.h>

namespace gbz2layout {

#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define CUDACHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "[gpu] CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); } } while (0)

struct __align__(8) node_t { float coords[4]; int32_t seq_length; };
struct __align__(8) path_element_t { uint32_t pidx; uint32_t node_id; int64_t pos; };
struct path_t { uint32_t step_count; uint64_t first_step_in_path; path_element_t* elements; };

struct curandStateCoalesced_t {
    unsigned int d[BLOCK_SIZE], w0[BLOCK_SIZE], w1[BLOCK_SIZE],
                 w2[BLOCK_SIZE], w3[BLOCK_SIZE], w4[BLOCK_SIZE];
};

// ---- device RNG (coalesced XORWOW), verbatim from odgi ----------------------
__global__ void cuda_device_init(curandState_t* tmp, curandStateCoalesced_t* rs) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(42 + tid, tid, 0, &tmp[tid]);
    rs[blockIdx.x].d[threadIdx.x]  = tmp[tid].d;
    rs[blockIdx.x].w0[threadIdx.x] = tmp[tid].v[0];
    rs[blockIdx.x].w1[threadIdx.x] = tmp[tid].v[1];
    rs[blockIdx.x].w2[threadIdx.x] = tmp[tid].v[2];
    rs[blockIdx.x].w3[threadIdx.x] = tmp[tid].v[3];
    rs[blockIdx.x].w4[threadIdx.x] = tmp[tid].v[4];
}
__device__ unsigned int curand_coalesced(curandStateCoalesced_t* s, uint32_t t) {
    uint32_t x = (s->w0[t] ^ (s->w0[t] >> 2));
    s->w0[t] = s->w1[t]; s->w1[t] = s->w2[t]; s->w2[t] = s->w3[t]; s->w3[t] = s->w4[t];
    s->w4[t] = (s->w4[t] ^ (s->w4[t] << 4)) ^ (x ^ (x << 1));
    s->d[t] += 362437;
    return s->w4[t] + s->d[t];
}
__device__ float curand_uniform_coalesced(curandStateCoalesced_t* s, uint32_t t) {
    uint32_t x = s->w0[t] ^ (s->w0[t] >> 2);
    s->w0[t] = s->w1[t]; s->w1[t] = s->w2[t]; s->w2[t] = s->w3[t]; s->w3[t] = s->w4[t];
    s->w4[t] = (s->w4[t] ^ (s->w4[t] << 4)) ^ (x ^ (x << 1));
    s->d[t] += 362437;
    return _curand_uniform(s->d[t] + s->w4[t]);
}
__device__ uint32_t cuda_rnd_zipf(curandStateCoalesced_t* rs, uint32_t n, double theta, double zeta2, double zetan) {
    double alpha = 1.0 / (1.0 - theta);
    double denom = 1.0 - zeta2 / zetan; if (denom == 0.0) denom = 1e-9;
    double eta = (1.0 - __powf(2.0 / double(n), 1.0 - theta)) / denom;
    double u = 1.0 - curand_uniform_coalesced(rs, threadIdx.x);
    double uz = u * zetan;
    int64_t v;
    if (uz < 1.0) v = 1;
    else if (uz < 1.0 + __powf(0.5, theta)) v = 2;
    else v = 1 + int64_t(double(n) * __powf(eta * u - eta + 1.0, alpha));
    if (v > n) v--;
    if (v < 1) v = 1;
    return uint32_t(v);
}
static __device__ __inline__ uint32_t mysmid() {
    uint32_t smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); return smid;
}

// ---- update (odgi update_pos_gpu, atomicExch Hogwild) -----------------------
__device__ void update_pos(int64_t p1, uint32_t id1, int off1,
                           int64_t p2, uint32_t id2, int off2, double eta, node_t* nodes) {
    double term = fabs(double(p1) - double(p2)); if (term < 1e-9) term = 1e-9;
    double mu = eta / term; if (mu > 1.0) mu = 1.0;
    float* x1 = &nodes[id1].coords[off1]; float* y1 = &nodes[id1].coords[off1 + 1];
    float* x2 = &nodes[id2].coords[off2]; float* y2 = &nodes[id2].coords[off2 + 1];
    double x1v = *x1, y1v = *y1, x2v = *x2, y2v = *y2;
    double dx = x1v - x2v, dy = y1v - y2v; if (dx == 0.0) dx = 1e-9;
    double mag = sqrt(dx * dx + dy * dy);
    double r = (mu * (mag - term) / 2.0) / mag, rx = r * dx, ry = r * dy;
    atomicExch(x1, float(x1v - rx)); atomicExch(x2, float(x2v + rx));
    atomicExch(y1, float(y1v - ry)); atomicExch(y2, float(y2v + ry));
}

// ---- the streaming kernel (one group's path_data resident this launch) ------
__global__ void group_kernel(curandStateCoalesced_t* rnd, double eta, double* zetas,
                             node_t* nodes, path_element_t* elems, path_t* paths,
                             uint64_t total_steps, double theta, uint32_t space,
                             uint32_t space_max, uint32_t sqs, uint32_t window,
                             int cooling_iter, uint64_t updates, int sm_count) {
    uint64_t gid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= updates) return;
    uint32_t smid = mysmid();
    curandStateCoalesced_t* rs = &rnd[smid];

    __shared__ bool cooling[BLOCK_SIZE / WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 1)
        cooling[threadIdx.x / WARP_SIZE] = cooling_iter || (curand_coalesced(rs, threadIdx.x) % 2 == 0);
    __syncthreads();

    uint32_t step_idx = curand_coalesced(rs, threadIdx.x) % total_steps;
    uint32_t pidx = elems[step_idx].pidx;
    path_t p = paths[pidx];
    if (p.step_count < 2) return;

    uint32_t s1 = curand_coalesced(rs, threadIdx.x) % p.step_count;
    uint32_t s2;
    if (cooling[threadIdx.x / WARP_SIZE]) {
        bool backward; uint32_t js;
        if ((s1 > 0 && (curand_coalesced(rs, threadIdx.x) % 2 == 0)) || s1 == p.step_count - 1) {
            backward = true;  js = min(min(space, s1), window ? window : space);
        } else {
            backward = false; js = min(min(space, p.step_count - s1 - 1), window ? window : space);
        }
        uint32_t sp = js;
        if (js > space_max) sp = space_max + (js - space_max) / sqs + 1;
        uint32_t z = (js >= 1) ? cuda_rnd_zipf(rs, js, theta, zetas[2], zetas[sp]) : 1;
        s2 = backward ? s1 - z : s1 + z;
    } else {
        uint32_t lo = 0, hi = p.step_count - 1;
        if (window) { lo = (s1 > window) ? s1 - window : 0; hi = min(p.step_count - 1, s1 + window); }
        do { s2 = lo + curand_coalesced(rs, threadIdx.x) % (hi - lo + 1); } while (s2 == s1);
    }
    if (s2 >= p.step_count || s2 == s1) return;

    uint32_t id1 = p.elements[s1].node_id; int64_t pp1 = p.elements[s1].pos;
    bool rev1 = pp1 < 0; pp1 = llabs(pp1);
    uint32_t id2 = p.elements[s2].node_id; int64_t pp2 = p.elements[s2].pos;
    bool rev2 = pp2 < 0; pp2 = llabs(pp2);

    bool oe1 = (curand_coalesced(rs, threadIdx.x) % 2 == 0);
    if (oe1) { pp1 += nodes[id1].seq_length; oe1 = !rev1; } else oe1 = rev1;
    bool oe2 = (curand_coalesced(rs, threadIdx.x) % 2 == 0);
    if (oe2) { pp2 += nodes[id2].seq_length; oe2 = !rev2; } else oe2 = rev2;

    update_pos(pp1, id1, oe1 ? 2 : 0, pp2, id2, oe2 ? 2 : 0, eta, nodes);
}

// ---- host context -----------------------------------------------------------
struct GpuImpl {
    uint64_t N = 0;
    int sm_count = 0;
    node_t* nodes = nullptr;
    double* zetas = nullptr;
    curandStateCoalesced_t* rnd = nullptr;
    double theta = 0.99;
    uint32_t space = 0, space_max = 1000, sqs = 100;
    // reusable per-group buffers
    path_element_t* elems = nullptr; uint64_t elems_cap = 0;
    path_t* paths = nullptr; uint32_t paths_cap = 0;
};

bool GpuLayout::available() {
    int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void GpuLayout::init(std::uint64_t node_count, std::uint64_t /*seed*/, std::uint64_t /*nthreads*/,
                     const std::uint32_t* seq_lengths, const std::vector<double>& zetas_host,
                     std::uint64_t space, std::uint64_t space_max, std::uint64_t sqs, double theta) {
    GpuImpl* g = new GpuImpl();
    impl = g;
    g->N = node_count; g->theta = theta; g->space = space; g->space_max = space_max; g->sqs = sqs;
    cudaDeviceProp prop; CUDACHECK(cudaGetDeviceProperties(&prop, 0)); g->sm_count = prop.multiProcessorCount;

    CUDACHECK(cudaMallocManaged(&g->nodes, node_count * sizeof(node_t)));
    for (uint64_t i = 0; i < node_count; ++i) g->nodes[i].seq_length = (int32_t)seq_lengths[i];

    CUDACHECK(cudaMallocManaged(&g->zetas, zetas_host.size() * sizeof(double)));
    for (size_t i = 0; i < zetas_host.size(); ++i) g->zetas[i] = zetas_host[i];

    curandState_t* tmp;
    CUDACHECK(cudaMallocManaged(&tmp, (uint64_t)g->sm_count * BLOCK_SIZE * sizeof(curandState_t)));
    CUDACHECK(cudaMallocManaged(&g->rnd, (uint64_t)g->sm_count * sizeof(curandStateCoalesced_t)));
    cuda_device_init<<<g->sm_count, BLOCK_SIZE>>>(tmp, g->rnd);
    CUDACHECK(cudaDeviceSynchronize());
    cudaFree(tmp);
}

void GpuLayout::set_coords(const double* X, const double* Y) {
    GpuImpl* g = (GpuImpl*)impl;
    for (uint64_t r = 0; r < g->N; ++r) {
        g->nodes[r].coords[0] = (float)X[2 * r];     g->nodes[r].coords[1] = (float)Y[2 * r];
        g->nodes[r].coords[2] = (float)X[2 * r + 1]; g->nodes[r].coords[3] = (float)Y[2 * r + 1];
    }
}
void GpuLayout::get_coords(double* X, double* Y) const {
    GpuImpl* g = (GpuImpl*)impl;
    CUDACHECK(cudaDeviceSynchronize());
    for (uint64_t r = 0; r < g->N; ++r) {
        X[2 * r] = g->nodes[r].coords[0];     Y[2 * r] = g->nodes[r].coords[1];
        X[2 * r + 1] = g->nodes[r].coords[2]; Y[2 * r + 1] = g->nodes[r].coords[3];
    }
}

void GpuLayout::run_group(const std::uint32_t* rank, const std::uint64_t* pos, const std::uint8_t* rev,
                          const std::uint64_t* start, std::uint32_t npaths, std::uint64_t total,
                          double eta, bool cooling, std::uint64_t updates, std::uint64_t window) {
    GpuImpl* g = (GpuImpl*)impl;
    if (total < 2 || updates == 0) return;
    if (total > g->elems_cap) { if (g->elems) cudaFree(g->elems); CUDACHECK(cudaMallocManaged(&g->elems, total * sizeof(path_element_t))); g->elems_cap = total; }
    if (npaths > g->paths_cap) { if (g->paths) cudaFree(g->paths); CUDACHECK(cudaMallocManaged(&g->paths, npaths * sizeof(path_t))); g->paths_cap = npaths; }
    // build path_data for this group
    for (uint32_t p = 0; p < npaths; ++p) {
        g->paths[p].step_count = (uint32_t)(start[p + 1] - start[p]);
        g->paths[p].first_step_in_path = start[p];
        g->paths[p].elements = &g->elems[start[p]];
    }
    // fill pidx + node_id + signed pos (pos+1 so orientation of the 0-position node survives)
    uint32_t p = 0;
    for (uint64_t s = 0; s < total; ++s) {
        while (p + 1 < npaths && s >= start[p + 1]) ++p;
        g->elems[s].pidx = p;
        g->elems[s].node_id = rank[s];
        int64_t pv = (int64_t)pos[s] + 1;
        g->elems[s].pos = rev[s] ? -pv : pv;
    }
    uint64_t block_nbr = (updates + BLOCK_SIZE - 1) / BLOCK_SIZE;
    group_kernel<<<block_nbr, BLOCK_SIZE>>>(g->rnd, eta, g->zetas, g->nodes, g->elems, g->paths,
                                            total, g->theta, g->space, g->space_max, g->sqs,
                                            (uint32_t)window, cooling ? 1 : 0, updates, g->sm_count);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());
}

void GpuLayout::destroy() {
    GpuImpl* g = (GpuImpl*)impl; if (!g) return;
    if (g->nodes) cudaFree(g->nodes); if (g->zetas) cudaFree(g->zetas); if (g->rnd) cudaFree(g->rnd);
    if (g->elems) cudaFree(g->elems); if (g->paths) cudaFree(g->paths);
    delete g; impl = nullptr;
}

} // namespace gbz2layout
