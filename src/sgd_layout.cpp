// sgd_layout.cpp — port of odgi's path_linear_sgd_layout (CPU path).
// Source: odgi v0.9.3 src/algorithms/path_sgd_layout.cpp (commit 405be8f6),
// stripped of snapshots / progress-meter / GPU and adapted to the GBWT-backed XP.

#include "sgd_layout.hpp"

#include "third_party/XoshiroCpp.hpp"
#include "third_party/dirty_zipfian_int_distribution.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

namespace gbz2layout {

using handlegraph::as_integer;
using handlegraph::as_integers;
using handlegraph::as_path_handle;
using handlegraph::handle_t;
using handlegraph::path_handle_t;
using handlegraph::step_handle_t;

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
    for (std::int64_t t = 0; t <= (std::int64_t)iter_max; t++) {
        etas.push_back(eta_max * std::exp(-lambda * std::abs(t - (std::int64_t)iter_with_max_learning_rate)));
    }
    return etas;
}

void path_linear_sgd_layout(const gbwtgraph::GBWTGraph& graph,
                            const XP& path_index,
                            const SgdParams& p,
                            std::vector<std::atomic<double>>& X,
                            std::vector<std::atomic<double>>& Y) {
    const std::uint64_t iter_max = p.iter_max;
    const std::uint64_t min_term_updates = p.min_term_updates;
    const double delta = p.delta;
    const double theta = p.theta;
    const std::uint64_t space = p.space;
    const std::uint64_t space_max = p.space_max;
    const std::uint64_t space_quantization_step = p.space_quantization_step;
    const std::uint64_t nthreads = std::max<std::uint64_t>(1, p.nthreads);

    const std::uint64_t first_cooling_iteration = std::floor(p.cooling_start * (double)iter_max);
    const std::uint64_t total_term_updates = iter_max * min_term_updates;

    // need at least one path with >1 step
    bool ok = false;
    for (std::uint64_t pid = 0; pid < path_index.path_count(); ++pid)
        if (path_index.get_path_step_count(as_path_handle(pid)) > 1) { ok = true; break; }
    if (!ok) { std::cerr << "[sgd] no path with >1 step; nothing to do\n"; return; }

    // learning-rate schedule
    double w_min = 1.0 / p.eta_max;
    double w_max = 1.0;
    std::vector<double> etas = layout_schedule(w_min, w_max, iter_max, p.iter_with_max_learning_rate, p.eps);

    // cache Zipf zetas over the (quantized) path space
    std::vector<double> zetas((space <= space_max ? space : space_max + (space - space_max) / space_quantization_step + 1) + 1);
    {
        double zeta_tmp = 0.0;
        for (std::uint64_t i = 1; i < space + 1; i++) {
            zeta_tmp += dirtyzipf::fast_precise_pow(1.0 / i, theta);
            if (i <= space_max) zetas[i] = zeta_tmp;
            else if ((i - space_max) % space_quantization_step == 0)
                zetas[space_max + 1 + (i - space_max) / space_quantization_step] = zeta_tmp;
        }
    }

    // shared control state
    std::atomic<std::uint64_t> term_updates(0);
    std::atomic<double> eta(etas.front());
    std::atomic<bool> cooling(false);
    std::atomic<double> Delta_max(0);
    std::atomic<bool> work_todo(true);
    std::uint64_t iteration = 0;
    std::atomic<std::uint64_t> updates_done(0);   // for progress only

    // per-iteration spring: pull reference (pin_y) nodes' Y toward 0 by
    // pin_strength. strength 1 => snap flat each iteration (hard pin); 0 => free;
    // between => the spine bows toward big bubbles during an iteration and is
    // relaxed back a fraction each iteration, reaching a soft equilibrium.
    const bool do_pin = p.pin_y && p.pin_strength > 0.0;
    auto apply_pin = [&]() {
        if (!do_pin) return;
        const double keep = 1.0 - p.pin_strength;
        const std::vector<bool>& m = *p.pin_y;
        for (std::uint64_t r = 0; r < m.size(); ++r) {
            if (!m[r]) continue;
            Y[2 * r].store(Y[2 * r].load() * keep);
            Y[2 * r + 1].store(Y[2 * r + 1].load() * keep);
        }
    };

    // checker thread: advances iterations, updates eta, decides stop
    auto checker_lambda = [&]() {
        while (work_todo.load()) {
            if (term_updates.load() > min_term_updates) {
                iteration++;
                if (iteration >= iter_max) {
                    work_todo.store(false);
                } else if (Delta_max.load() <= delta) {
                    if (p.progress)
                        std::cerr << "[sgd] delta_max " << Delta_max.load() << " <= " << delta
                                  << ", stopping early at iter " << iteration << "\n";
                    work_todo.store(false);
                } else {
                    eta.store(etas[iteration]);
                    Delta_max.store(delta);
                    if (iteration >= first_cooling_iteration) cooling.store(true);
                    if (p.progress)
                        std::cerr << "[sgd] iter " << iteration << "/" << iter_max
                                  << " eta=" << eta.load()
                                  << " updates=" << updates_done.load() << "/" << total_term_updates << "\n";
                }
                term_updates.store(0);
                apply_pin();                 // pull reference back toward Y=0
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };

    auto worker_lambda = [&](std::uint64_t tid) {
        XoshiroCpp::Xoshiro256Plus gen(p.seed + tid);
        const sdsl::bit_vector&   np_bv  = path_index.get_np_bv();
        const sdsl::int_vector<>& nr_iv  = path_index.get_nr_iv();
        const sdsl::int_vector<>& npi_iv = path_index.get_npi_iv();
        std::uniform_int_distribution<std::uint64_t> dis_step(0, np_bv.size() - 1);
        std::uniform_int_distribution<std::uint64_t> flip(0, 1);
        std::uint64_t local = 0;

        while (work_todo.load()) {
            // sample step A uniformly across all indexed steps
            std::uint64_t step_index = dis_step(gen);
            std::uint64_t path_i = npi_iv[step_index];
            path_handle_t path = as_path_handle(path_i);
            std::uint64_t path_step_count = path_index.get_path_step_count(path);
            if (path_step_count == 1) continue;

            step_handle_t step_a, step_b;
            as_integers(step_a)[0] = path_i;
            std::uint64_t s_rank = nr_iv[step_index] - 1;   // 0-based retained rank
            as_integers(step_a)[1] = s_rank;

            if (cooling.load() || flip(gen)) {
                // Zipf-sample step B near A along the path
                if ((s_rank > 0 && flip(gen)) || s_rank == path_step_count - 1) {
                    std::uint64_t jump_space = std::min(space, (std::uint64_t)s_rank);
                    std::uint64_t space_l = jump_space;
                    if (jump_space > space_max)
                        space_l = space_max + (jump_space - space_max) / space_quantization_step + 1;
                    dirtyzipf::dirty_zipfian_int_distribution<std::uint64_t>::param_type z_p(1, jump_space, theta, zetas[space_l]);
                    dirtyzipf::dirty_zipfian_int_distribution<std::uint64_t> z(z_p);
                    std::uint64_t z_i = z(gen);
                    as_integers(step_b)[0] = path_i;
                    as_integers(step_b)[1] = s_rank - z_i;
                } else {
                    std::uint64_t jump_space = std::min(space, (std::uint64_t)(path_step_count - s_rank - 1));
                    std::uint64_t space_l = jump_space;
                    if (jump_space > space_max)
                        space_l = space_max + (jump_space - space_max) / space_quantization_step + 1;
                    dirtyzipf::dirty_zipfian_int_distribution<std::uint64_t>::param_type z_p(1, jump_space, theta, zetas[space_l]);
                    dirtyzipf::dirty_zipfian_int_distribution<std::uint64_t> z(z_p);
                    std::uint64_t z_i = z(gen);
                    as_integers(step_b)[0] = path_i;
                    as_integers(step_b)[1] = s_rank + z_i;
                }
            } else {
                // sample B uniformly across the (retained) path
                std::uniform_int_distribution<std::uint64_t> rando(0, path_index.get_path_step_count(path) - 1);
                as_integers(step_b)[0] = path_i;
                as_integers(step_b)[1] = rando(gen);
            }

            // resolve handles + positions
            handle_t term_i = path_index.get_handle_of_step(step_a);
            handle_t term_j = path_index.get_handle_of_step(step_b);
            std::uint64_t term_i_length = graph.get_length(term_i);
            std::uint64_t term_j_length = graph.get_length(term_j);
            std::uint64_t pos_in_path_a = path_index.get_position_of_step(step_a);
            std::uint64_t pos_in_path_b = path_index.get_position_of_step(step_b);

            bool term_i_is_rev = graph.get_is_reverse(term_i);
            bool use_other_end_a = flip(gen);
            if (use_other_end_a) { pos_in_path_a += term_i_length; use_other_end_a = !term_i_is_rev; }
            else                 { use_other_end_a = term_i_is_rev; }
            bool term_j_is_rev = graph.get_is_reverse(term_j);
            bool use_other_end_b = flip(gen);
            if (use_other_end_b) { pos_in_path_b += term_j_length; use_other_end_b = !term_j_is_rev; }
            else                 { use_other_end_b = term_j_is_rev; }

            double term_dist = std::abs((double)pos_in_path_a - (double)pos_in_path_b);
            if (term_dist == 0) term_dist = 1e-9;
            double w_ij = 1.0 / term_dist;
            double mu = eta.load() * w_ij;
            if (mu > 1) mu = 1;
            double d_ij = term_dist;

            std::uint64_t i = path_index.rank_of_handle(term_i);   // <-- id->rank remap
            std::uint64_t j = path_index.rank_of_handle(term_j);
            std::uint64_t offset_i = use_other_end_a ? 1 : 0;
            std::uint64_t offset_j = use_other_end_b ? 1 : 0;

            double dx = X[2 * i + offset_i].load() - X[2 * j + offset_j].load();
            double dy = Y[2 * i + offset_i].load() - Y[2 * j + offset_j].load();
            if (dx == 0) dx = 1e-9;
            double mag = std::sqrt(dx * dx + dy * dy);
            double Delta = mu * (mag - d_ij) / 2;
            double Delta_abs = std::abs(Delta);
            while (Delta_abs > Delta_max.load()) Delta_max.store(Delta_abs);

            double r = Delta / mag;
            double r_x = r * dx;
            double r_y = r * dy;
            // all nodes move freely; reference (pin_y) nodes are pulled back
            // toward Y=0 once per iteration in the checker thread (see below),
            // so the pull strength is independent of sampling frequency.
            X[2 * i + offset_i].store(X[2 * i + offset_i].load() - r_x);
            Y[2 * i + offset_i].store(Y[2 * i + offset_i].load() - r_y);
            X[2 * j + offset_j].store(X[2 * j + offset_j].load() + r_x);
            Y[2 * j + offset_j].store(Y[2 * j + offset_j].load() + r_y);

            if (++local >= 1000) { term_updates += local; updates_done += local; local = 0; }
        }
    };

    std::thread checker(checker_lambda);
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (std::uint64_t t = 0; t < nthreads; ++t) workers.emplace_back(worker_lambda, t);
    for (auto& w : workers) w.join();
    checker.join();
    apply_pin();   // final clean pull (no concurrent workers now)
}

} // namespace gbz2layout
