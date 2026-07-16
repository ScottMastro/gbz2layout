#include "sgd_minibatch.hpp"
#include "sgd_minibatch_gpu.hpp"

#include "third_party/XoshiroCpp.hpp"
#include "third_party/dirty_zipfian_int_distribution.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

namespace gbz2layout {

using handlegraph::handle_t;

// odgi's exponential learning-rate schedule (unchanged).
static std::vector<double> layout_schedule(double w_min, double w_max,
                                           std::uint64_t iter_max,
                                           std::uint64_t iter_with_max_learning_rate,
                                           double eps) {
    double eta_max = 1.0 / w_min;
    double eta_min = eps / w_max;
    double lambda = std::log(eta_max / eta_min) / ((double)iter_max - 1);
    std::vector<double> etas;
    etas.reserve(iter_max + 1);
    for (std::int64_t t = 0; t <= (std::int64_t)iter_max; t++)
        etas.push_back(eta_max * std::exp(-lambda * std::abs(t - (std::int64_t)iter_with_max_learning_rate)));
    return etas;
}

// A resident group of whole paths: flat per-step (rank, bp position, orientation),
// with per-path offsets into the flat arrays.
struct PathBatch {
    std::vector<std::uint32_t> rank;
    std::vector<std::uint64_t> pos;
    std::vector<std::uint8_t>  rev;
    std::vector<std::uint64_t> start;   // size npaths+1
    std::uint64_t total = 0;
};

static PathBatch build_batch(const gbwt::GBWT& idx, const XP& xp,
                             const std::uint32_t* seqs, std::uint64_t n) {
    PathBatch b;
    b.start.push_back(0);
    for (std::uint64_t k = 0; k < n; ++k) {
        gbwt::vector_type path = idx.extract(seqs[k]);
        std::uint64_t bp = 0;
        for (gbwt::node_type node : path) {
            std::uint64_t r = xp.rank_of_id(gbwt::Node::id(node));
            b.rank.push_back((std::uint32_t)r);
            b.pos.push_back(bp);
            b.rev.push_back(gbwt::Node::is_reverse(node) ? 1 : 0);
            bp += xp.node_length_of_rank(r);
        }
        b.start.push_back(b.rank.size());
    }
    b.total = b.rank.size();
    return b;
}

void path_linear_sgd_layout_minibatch(const gbwtgraph::GBWTGraph& /*graph*/,
                                      const gbwt::GBWT& gbwt_index,
                                      const XP& xp,
                                      const MinibatchParams& p,
                                      std::vector<std::atomic<double>>& X,
                                      std::vector<std::atomic<double>>& Y) {
    const std::uint64_t iter_max = p.iter_max;
    const std::uint64_t nthreads = std::max<std::uint64_t>(1, p.nthreads);
    const std::uint64_t space = p.space;
    const std::uint64_t space_max = p.space_max;
    const std::uint64_t space_quantization_step = p.space_quantization_step;
    const std::uint64_t first_cooling_iteration = std::floor(p.cooling_start * (double)iter_max);

    // learning-rate schedule
    double w_min = 1.0 / p.eta_max, w_max = 1.0;
    std::vector<double> etas = layout_schedule(w_min, w_max, iter_max, p.iter_with_max_learning_rate, p.eps);

    // Zipf zetas over the (quantized) path space — computed once
    std::vector<double> zetas((space <= space_max ? space : space_max + (space - space_max) / space_quantization_step + 1) + 1);
    { double z = 0.0;
      for (std::uint64_t i = 1; i < space + 1; i++) {
          z += dirtyzipf::fast_precise_pow(1.0 / i, p.theta);
          if (i <= space_max) zetas[i] = z;
          else if ((i - space_max) % space_quantization_step == 0)
              zetas[space_max + 1 + (i - space_max) / space_quantization_step] = z; } }

    // path list. Each EPOCH is a full pass over all paths: the paths are
    // shuffled and consumed in K-sized groups (without replacement), so every
    // path is seen once per epoch. updates/group = mult * group_steps, so an
    // epoch does mult * total_full_steps updates at that epoch's eta — matching
    // a non-batched run's per-iteration budget, at one-group peak memory.
    std::vector<std::uint32_t> seqs = p.paths ? *p.paths : xp.path_seq_ids();
    if (seqs.empty()) { std::cerr << "[minibatch] no paths\n"; return; }
    std::vector<std::uint32_t> order(seqs.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 shuf(p.seed);

    const std::uint64_t K = std::max<std::uint64_t>(1, p.batch_paths);
    std::uint64_t groups_per_pass = (seqs.size() + K - 1) / K;
    if (p.progress)
        std::cerr << "[minibatch] " << seqs.size() << " paths, K=" << K
                  << " -> " << groups_per_pass << " groups/epoch, "
                  << iter_max << " epochs (full pass each)"
                  << (p.use_gpu ? " [GPU]" : "") << "\n";

    // ---- GPU path: same epoch/group loop, update kernel on device ----
    if (p.use_gpu) {
        const std::uint64_t N = xp.node_count();
        std::vector<std::uint32_t> seqlen(N);
        for (std::uint64_t r = 0; r < N; ++r) seqlen[r] = (std::uint32_t)xp.node_length_of_rank(r);
        std::vector<double> Xd(2 * N), Yd(2 * N);
        for (std::uint64_t k = 0; k < 2 * N; ++k) { Xd[k] = X[k].load(); Yd[k] = Y[k].load(); }

        GpuLayout gpu;
        gpu.init(N, p.seed, p.nthreads, seqlen.data(), zetas, space, space_max, space_quantization_step, p.theta);
        gpu.set_coords(Xd.data(), Yd.data());

        for (std::uint64_t iter = 0; iter < iter_max; ++iter) {
            std::shuffle(order.begin(), order.end(), shuf);
            const double eta = etas[iter];
            const bool cooling = (iter >= first_cooling_iteration);
            std::uint64_t epoch_updates = 0;
            for (std::uint64_t gstart = 0; gstart < order.size(); gstart += K) {
                std::uint64_t gend = std::min(order.size(), gstart + K);
                std::vector<std::uint32_t> group;
                for (std::uint64_t c = gstart; c < gend; ++c) group.push_back(seqs[order[c]]);
                PathBatch b = build_batch(gbwt_index, xp, group.data(), group.size());
                if (b.total < 2) continue;
                std::uint64_t updates = p.updates_mult * b.total;
                epoch_updates += updates;
                gpu.run_group(b.rank.data(), b.pos.data(), b.rev.data(), b.start.data(),
                              (std::uint32_t)(b.start.size() - 1), b.total, eta, cooling,
                              updates, p.window_len);
            }
            if (p.progress)
                std::cerr << "[minibatch] epoch " << iter << "/" << iter_max
                          << " eta=" << eta << " updates=" << epoch_updates
                          << (cooling ? " (cooling)" : "") << " [GPU]\n";
        }
        gpu.get_coords(Xd.data(), Yd.data());
        for (std::uint64_t k = 0; k < 2 * N; ++k) { X[k].store(Xd[k]); Y[k].store(Yd[k]); }
        gpu.destroy();
        if (p.progress) std::cerr << "[minibatch] done (" << iter_max << " full-pass epochs, GPU)\n";
        return;
    }

    for (std::uint64_t iter = 0; iter < iter_max; ++iter) {
        std::shuffle(order.begin(), order.end(), shuf);     // fresh pass each epoch
        const double eta = etas[iter];
        const bool cooling = (iter >= first_cooling_iteration);
        std::atomic<double> Delta_max(0);
        std::uint64_t epoch_updates = 0;

      for (std::uint64_t gstart = 0; gstart < order.size(); gstart += K) {
        std::uint64_t gend = std::min(order.size(), gstart + K);
        std::vector<std::uint32_t> group;
        group.reserve(gend - gstart);
        for (std::uint64_t c = gstart; c < gend; ++c) group.push_back(seqs[order[c]]);
        PathBatch b = build_batch(gbwt_index, xp, group.data(), group.size());
        if (b.total < 2) continue;

        const std::uint64_t updates = p.updates_mult * b.total;
        epoch_updates += updates;

        auto worker = [&](std::uint64_t tid) {
            XoshiroCpp::Xoshiro256Plus gen(p.seed + iter * 1000003ull + gstart * 131ull + tid);
            std::uniform_int_distribution<std::uint64_t> dis_step(0, b.total - 1);
            std::uniform_int_distribution<std::uint64_t> flip(0, 1);
            const std::uint64_t my = updates / nthreads;

            for (std::uint64_t u = 0; u < my; ++u) {
                std::uint64_t sA = dis_step(gen);
                // locate path of sA
                std::uint64_t pth = std::upper_bound(b.start.begin(), b.start.end(), sA) - b.start.begin() - 1;
                std::uint64_t s0 = b.start[pth], s1 = b.start[pth + 1];
                std::uint64_t plen = s1 - s0;
                if (plen < 2) continue;
                std::uint64_t iA = sA - s0;

                // sub-path window: cap how far B can be from A along the path
                const std::uint64_t W = p.window_len ? std::min(space, p.window_len) : space;
                std::uint64_t sB;
                if (cooling || flip(gen)) {
                    std::uint64_t jump, z_i;
                    if ((iA > 0 && flip(gen)) || iA == plen - 1) {
                        jump = std::min(W, iA);
                        std::uint64_t sl = jump > space_max ? space_max + (jump - space_max) / space_quantization_step + 1 : jump;
                        dirtyzipf::dirty_zipfian_int_distribution<std::uint64_t>::param_type zp(1, jump, p.theta, zetas[sl]);
                        dirtyzipf::dirty_zipfian_int_distribution<std::uint64_t> z(zp); z_i = z(gen);
                        sB = s0 + (iA - z_i);
                    } else {
                        jump = std::min(W, plen - iA - 1);
                        std::uint64_t sl = jump > space_max ? space_max + (jump - space_max) / space_quantization_step + 1 : jump;
                        dirtyzipf::dirty_zipfian_int_distribution<std::uint64_t>::param_type zp(1, jump, p.theta, zetas[sl]);
                        dirtyzipf::dirty_zipfian_int_distribution<std::uint64_t> z(zp); z_i = z(gen);
                        sB = s0 + (iA + z_i);
                    }
                } else {
                    // uniform B, but within +/- W of A when windowed
                    std::uint64_t lo = (p.window_len && iA > W) ? iA - W : 0;
                    std::uint64_t hi = p.window_len ? std::min(plen - 1, iA + W) : plen - 1;
                    std::uniform_int_distribution<std::uint64_t> r(lo, hi);
                    sB = s0 + r(gen);
                }
                if (sB == sA) continue;

                std::uint64_t i = b.rank[sA], j = b.rank[sB];
                double pa = (double)b.pos[sA], pb = (double)b.pos[sB];
                std::uint64_t la = xp.node_length_of_rank(i), lb = xp.node_length_of_rank(j);
                bool ra = b.rev[sA], rb = b.rev[sB];

                bool oea = flip(gen);
                if (oea) { pa += la; oea = !ra; } else { oea = ra; }
                bool oeb = flip(gen);
                if (oeb) { pb += lb; oeb = !rb; } else { oeb = rb; }

                double term_dist = std::abs(pa - pb);
                if (term_dist == 0) term_dist = 1e-9;
                double w_ij = 1.0 / term_dist;
                double mu = eta * w_ij; if (mu > 1) mu = 1;
                std::uint64_t oi = oea ? 1 : 0, oj = oeb ? 1 : 0;

                double dx = X[2*i+oi].load() - X[2*j+oj].load();
                double dy = Y[2*i+oi].load() - Y[2*j+oj].load();
                if (dx == 0) dx = 1e-9;
                double mag = std::sqrt(dx*dx + dy*dy);
                double Delta = mu * (mag - term_dist) / 2;
                double Da = std::abs(Delta);
                while (Da > Delta_max.load()) Delta_max.store(Da);
                double r = Delta / mag, rx = r * dx, ry = r * dy;
                X[2*i+oi].store(X[2*i+oi].load() - rx);
                Y[2*i+oi].store(Y[2*i+oi].load() - ry);
                X[2*j+oj].store(X[2*j+oj].load() + rx);
                Y[2*j+oj].store(Y[2*j+oj].load() + ry);
            }
        };

        std::vector<std::thread> ts;
        for (std::uint64_t t = 0; t < nthreads; ++t) ts.emplace_back(worker, t);
        for (auto& t : ts) t.join();
      } // group loop (one full pass)

        if (p.progress)
            std::cerr << "[minibatch] epoch " << iter << "/" << iter_max
                      << " eta=" << eta << " updates=" << epoch_updates
                      << (cooling ? " (cooling)" : "") << "\n";
    }
    if (p.progress) std::cerr << "[minibatch] done (" << iter_max << " full-pass epochs)\n";
}

} // namespace gbz2layout
