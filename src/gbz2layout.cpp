// gbz2layout — compute an odgi-compatible 2D layout directly from a GBZ.
//
//   load GBZ -> build GBWT-backed lean XP (optional per-node cap)
//   -> reference-anchored init -> PG-SGD (odgi port) -> emit .lay.tsv
//
// Output idx column matches odgi: 2 rows per node (2*rank, 2*rank+1) in
// GBWTGraph for_each_handle order. The GFA fed to `pangyplot add` must share
// that node order (integration contract, validated at M3).

#include "xp.hpp"
#include "sgd_layout.hpp"
#include "compartment.hpp"

#include <gbwtgraph/gbz.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <thread>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <vector>

using namespace gbz2layout;

static void usage() {
    std::cerr <<
    "usage: gbz2layout <graph.gbz> -o <out_prefix> [options]\n"
    "  -o PREFIX          output prefix (writes PREFIX.lay.tsv)\n"
    "  --cap N            per-node coverage cap (0 = uncapped) [30]\n"
    "  --threads N        worker threads [hw]\n"
    "  --iter N           max SGD iterations [30]\n"
    "  --updates-per-node M   min_term_updates = M * node_count (else 10*steps)\n"
    "  --init ref|random  initialization [ref]\n"
    "  --hierarchical     freeze backbone, lay out each bubble independently (prototype)\n"
    "  --compartments N   balanced pinch-bounded compartments, N target tasks (parallel-ready)\n"
    "  --pinch-window W   local-envelope half-window for pinch detection [50]\n"
    "  --seed N           RNG seed [9399220]\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string gbz_path = argv[1];
    std::string out_prefix;
    std::uint64_t cap = 30;
    std::uint64_t threads = std::max(1u, std::thread::hardware_concurrency());
    std::uint64_t iter_max = 30;
    std::uint64_t updates_per_node = 0;
    std::string init_mode = "ref";
    std::uint64_t seed = 9399220;
    std::string chromosome;   // if set, extract this chromosome from a whole-genome GBZ
    bool pin_reference = false;
    double pin_strength = 1.0; // 1 = hard pin, 0 = free, between = soft spring
    bool emit_links = false;   // also write PREFIX.links.tsv (edges) for rendering
    bool hierarchical = false; // divide-and-conquer: freeze backbone, lay out each bubble independently
    std::uint64_t compartments = 0;   // >0: balanced pinch-bounded compartments (target task count)
    std::uint64_t pinch_window = 50;  // local-envelope half-window (reference nodes)

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "-o") out_prefix = next();
        else if (a == "--cap") cap = std::stoull(next());
        else if (a == "--threads") threads = std::stoull(next());
        else if (a == "--iter") iter_max = std::stoull(next());
        else if (a == "--updates-per-node") updates_per_node = std::stoull(next());
        else if (a == "--init") init_mode = next();
        else if (a == "--seed") seed = std::stoull(next());
        else if (a == "--chromosome") chromosome = next();
        else if (a == "--pin-reference") pin_reference = true;
        else if (a == "--pin-strength") { pin_reference = true; pin_strength = std::stod(next()); }
        else if (a == "--emit-links") emit_links = true;
        else if (a == "--hierarchical") hierarchical = true;
        else if (a == "--compartments") compartments = std::stoull(next());
        else if (a == "--pinch-window") pinch_window = std::stoull(next());
        else { std::cerr << "unknown arg: " << a << "\n"; usage(); return 1; }
    }
    if (out_prefix.empty()) { std::cerr << "error: -o required\n"; return 1; }

    // load GBZ
    std::cerr << "[gbz2layout] loading " << gbz_path << "\n";
    gbwtgraph::GBZ gbz;
    { std::ifstream in(gbz_path, std::ios::binary);
      if (!in) { std::cerr << "cannot open " << gbz_path << "\n"; return 1; }
      gbz.simple_sds_load(in); }
    const gbwtgraph::GBWTGraph& graph = gbz.graph;

    // optional: extract a single chromosome from a whole-genome GBZ by BFS'ing
    // its connected component (MC chromosomes are disjoint components).
    std::vector<bool> mask;
    std::int64_t genome_min_id = graph.min_node_id();
    if (!chromosome.empty()) {
        std::int64_t gmax = graph.max_node_id();
        std::uint64_t span = std::uint64_t(gmax - genome_min_id + 1);
        mask.assign(span, false);
        // find the chromosome's reference path: PanSN name sample#hap#<contig>,
        // match contig (after last '#') == chromosome
        bool found = false;
        std::queue<std::uint64_t> bfs;      // node ids
        graph.for_each_path_handle([&](const handlegraph::path_handle_t& path) {
            if (found) return;
            std::string nm = graph.get_path_name(path);
            std::string contig = nm.substr(nm.find_last_of('#') + 1);
            if (contig != chromosome) return;
            found = true;
            graph.for_each_step_in_path(path, [&](const handlegraph::step_handle_t& s) {
                std::uint64_t id = graph.get_id(graph.get_handle_of_step(s));
                if (!mask[id - genome_min_id]) { mask[id - genome_min_id] = true; bfs.push(id); }
            });
        });
        if (!found) { std::cerr << "error: no reference path for contig '" << chromosome << "'\n"; return 1; }
        // BFS the component to capture off-reference (bubble) nodes too
        std::uint64_t marked = bfs.size();
        while (!bfs.empty()) {
            std::uint64_t id = bfs.front(); bfs.pop();
            handle_t h = graph.get_handle(id, false);
            for (bool go_left : {false, true})
                graph.follow_edges(h, go_left, [&](const handle_t& nb) {
                    std::uint64_t nid = graph.get_id(nb);
                    if (!mask[nid - genome_min_id]) { mask[nid - genome_min_id] = true; bfs.push(nid); ++marked; }
                    return true;
                });
        }
        std::cerr << "[gbz2layout] chromosome " << chromosome << ": " << marked
                  << " nodes in component\n";
    }

    // build XP (filtered to the chromosome if a mask was computed)
    XP xp;
    xp.build(gbz, cap, true, chromosome.empty() ? nullptr : &mask, genome_min_id);
    const std::uint64_t N = xp.node_count();

    // ---- initialization ----
    std::vector<std::atomic<double>> X(2 * N), Y(2 * N);
    std::mt19937_64 rng(seed);
    double sigma = std::sqrt((double)N * 2.0);
    std::normal_distribution<double> gauss(0.0, sigma);
    for (std::uint64_t k = 0; k < 2 * N; ++k) Y[k].store(gauss(rng));

    bool use_ref = (init_mode == "ref") && xp.has_reference();
    if (init_mode == "ref" && !xp.has_reference())
        std::cerr << "[gbz2layout] no reference path exposed; falling back to random init\n";

    if (use_ref) {
        // place reference nodes at their bp coordinate, then BFS-propagate X to
        // off-reference nodes from placed neighbors (doc §4 #2).
        std::vector<char> placed(N, 0);
        std::queue<std::uint64_t> q;
        const auto& rr = xp.ref_ranks();
        const auto& rp = xp.ref_positions();
        for (std::uint64_t s = 0; s < rr.size(); ++s) {
            std::uint64_t r = rr[s];
            if (placed[r]) continue;
            double x0 = (double)rp[s];
            X[2 * r].store(x0);
            X[2 * r + 1].store(x0 + xp.node_length_of_rank(r));
            placed[r] = 1;
            q.push(r);
        }
        std::uint64_t ref_placed = q.size();
        while (!q.empty()) {
            std::uint64_t r = q.front(); q.pop();
            double xr_end = X[2 * r + 1].load();
            handle_t h = graph.get_handle(xp.node_id_of_rank(r), false);
            for (bool go_left : {false, true}) {
                graph.follow_edges(h, go_left, [&](const handle_t& nb) {
                    std::uint64_t nr = xp.rank_of_handle(nb);
                    if (!placed[nr]) {
                        double x0 = xr_end;
                        X[2 * nr].store(x0);
                        X[2 * nr + 1].store(x0 + xp.node_length_of_rank(nr));
                        placed[nr] = 1;
                        q.push(nr);
                    }
                    return true;
                });
            }
        }
        // any disconnected leftovers: random along reference length
        double ref_len = rp.empty() ? 1.0 : (double)rp.back() + xp.node_length_of_rank(rr.back());
        std::uniform_real_distribution<double> uni(0.0, ref_len);
        std::uint64_t leftover = 0;
        for (std::uint64_t r = 0; r < N; ++r) if (!placed[r]) {
            double x0 = uni(rng);
            X[2 * r].store(x0); X[2 * r + 1].store(x0 + xp.node_length_of_rank(r));
            ++leftover;
        }
        std::cerr << "[gbz2layout] ref-init: " << ref_placed << " ref nodes, "
                  << (N - ref_placed - leftover) << " propagated, " << leftover << " random leftover\n";
    } else {
        // odgi 'r' init: uniform random in the order of the graph length
        double total_len = 0;
        for (std::uint64_t r = 0; r < N; ++r) total_len += xp.node_length_of_rank(r);
        std::uniform_real_distribution<double> uni(0.0, total_len);
        for (std::uint64_t k = 0; k < 2 * N; ++k) X[k].store(uni(rng));
    }

    // ---- optional: pin reference nodes to the line Y=0 (free in X) ----
    std::vector<bool> ref_mask;
    if (pin_reference) {
        if (!xp.has_reference()) {
            std::cerr << "[gbz2layout] --pin-reference: no reference path; ignoring\n";
        } else {
            ref_mask.assign(N, false);
            for (std::uint32_t r : xp.ref_ranks()) {
                ref_mask[r] = true;
                Y[2 * r].store(0.0);        // lock both endpoints onto Y=0
                Y[2 * r + 1].store(0.0);
            }
            std::uint64_t pinned = 0;
            for (bool b : ref_mask) pinned += b;
            std::cerr << "[gbz2layout] pin-reference: " << pinned
                      << " reference nodes locked to Y=0 (free in X)\n";
        }
    }

    // ---- compartment mode: balanced pinch-bounded chunks laid out jointly ----
    // Each compartment = a backbone stretch + its bubbles (full SGD inside, so
    // it de-collides); only the chosen boundary pinches are frozen anchors.
    std::vector<bool>         comp_freeze;
    std::vector<std::int32_t> comp_region;
    if (compartments > 0) {
        if (!xp.has_reference()) {
            std::cerr << "[gbz2layout] --compartments: no reference path; ignoring\n";
            compartments = 0;
        } else {
            CompartmentResult cr = build_compartments(
                gbz, graph, xp, compartments, pinch_window, genome_min_id,
                chromosome.empty() ? nullptr : &mask, true);
            comp_region = std::move(cr.region);
            comp_freeze = std::move(cr.freeze);
            for (std::uint64_t r = 0; r < N; ++r)
                if (comp_freeze[r]) { Y[2 * r].store(0.0); Y[2 * r + 1].store(0.0); }  // anchors on Y=0
        }
    }

    // ---- hierarchical (divide-and-conquer): freeze the reference backbone flat
    // on Y=0 and partition off-reference nodes into independent bubble regions
    // (connected components of the non-reference subgraph). Each bubble then
    // settles on its own against the fixed backbone; cross-bubble forces are cut.
    std::vector<bool>         freeze_mask;   // reference/anchor nodes (never move)
    std::vector<std::int32_t> region;        // interior region id; -1 = anchor
    if (hierarchical) {
        if (!xp.has_reference()) {
            std::cerr << "[gbz2layout] --hierarchical: no reference path; ignoring\n";
            hierarchical = false;
        } else {
            freeze_mask.assign(N, false);
            for (std::uint32_t r : xp.ref_ranks()) {
                freeze_mask[r] = true;
                Y[2 * r].store(0.0);        // flat skeleton on Y=0 (X already = bp)
                Y[2 * r + 1].store(0.0);
            }
            // union-find over non-reference nodes: connect interior neighbours.
            std::vector<std::uint64_t> uf(N);
            for (std::uint64_t r = 0; r < N; ++r) uf[r] = r;
            std::function<std::uint64_t(std::uint64_t)> find =
                [&](std::uint64_t x){ while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };
            auto uni = [&](std::uint64_t a, std::uint64_t b){ uf[find(a)] = find(b); };
            for (std::uint64_t r = 0; r < N; ++r) {
                if (freeze_mask[r]) continue;
                handle_t h = graph.get_handle(xp.node_id_of_rank(r), false);
                for (bool go_left : {false, true})
                    graph.follow_edges(h, go_left, [&](const handle_t& nb) {
                        std::uint64_t nr = xp.rank_of_handle(nb);
                        if (nr < N && !freeze_mask[nr]) uni(r, nr);
                        return true;
                    });
            }
            // densify component roots into region ids; anchors get -1
            region.assign(N, -1);
            std::vector<std::int32_t> root2id(N, -1);
            std::int32_t next_id = 0;
            std::uint64_t interior = 0;
            for (std::uint64_t r = 0; r < N; ++r) {
                if (freeze_mask[r]) continue;
                std::uint64_t root = find(r);
                if (root2id[root] < 0) root2id[root] = next_id++;
                region[r] = root2id[root];
                ++interior;
            }
            std::cerr << "[gbz2layout] hierarchical: " << (N - interior) << " frozen anchors, "
                      << interior << " interior nodes in " << next_id << " bubble regions\n";
        }
    }

    // ---- SGD params (odgi defaults) ----
    SgdParams p;
    p.pin_y = ref_mask.empty() ? nullptr : &ref_mask;
    p.pin_strength = pin_strength;
    if (compartments > 0)      { p.region = &comp_region; p.freeze = &comp_freeze; }
    else if (hierarchical)     { p.region = &region;      p.freeze = &freeze_mask; }
    else                       { p.region = nullptr;      p.freeze = nullptr; }
    p.iter_max = iter_max;
    p.nthreads = threads;
    p.seed = seed;
    std::uint64_t max_steps = xp.max_path_step_count();
    p.space = max_steps;
    p.space_max = std::min<std::uint64_t>(p.space, 1000);
    p.space_quantization_step = 100;
    p.eta_max = (double)max_steps * (double)max_steps;
    p.theta = 0.99;
    p.eps = 0.01;
    p.delta = 0.0;
    p.cooling_start = 0.5;
    p.min_term_updates = updates_per_node ? updates_per_node * N : 10 * xp.total_steps();

    std::cerr << "[gbz2layout] SGD: iter=" << p.iter_max << " threads=" << p.nthreads
              << " max_steps=" << max_steps << " eta_max=" << p.eta_max
              << " min_term_updates=" << p.min_term_updates
              << " (total " << (p.iter_max * p.min_term_updates) << ")\n";

    auto t0 = std::chrono::steady_clock::now();
    path_linear_sgd_layout(graph, xp, p, X, Y);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << "[gbz2layout] SGD done in " << secs << " s\n";

    // ---- emit .lay.tsv (odgi-compatible) ----
    std::string tsv = out_prefix + ".lay.tsv";
    std::ofstream out(tsv);
    out << "idx\tX\tY\tcomponent\n";
    out.precision(9);
    // flush denormals / negligible coordinates to 0: a node pulled onto the
    // frozen backbone can underflow to a subnormal, which is meaningless at
    // layout scale and trips strtod-based parsers.
    auto fz = [](double v){ return std::fabs(v) < 1e-9 ? 0.0 : v; };
    for (std::uint64_t r = 0; r < N; ++r) {
        out << (2 * r)     << '\t' << fz(X[2 * r].load())     << '\t' << fz(Y[2 * r].load())     << "\t0\n";
        out << (2 * r + 1) << '\t' << fz(X[2 * r + 1].load()) << '\t' << fz(Y[2 * r + 1].load()) << "\t0\n";
    }
    out.close();
    std::cerr << "[gbz2layout] wrote " << tsv << " (" << (2 * N) << " rows)\n";

    // ---- optional: emit links (edges) for rendering connectivity ----
    // node-rank pairs a<b (deduped). Lets the renderer draw actual edges so
    // bubbles read as loops threaded on the spine, not floating segments.
    if (emit_links) {
        std::string lp = out_prefix + ".links.tsv";
        std::ofstream lo(lp);
        lo << "a\tb\n";
        std::uint64_t nlinks = 0;
        for (std::uint64_t r = 0; r < N; ++r) {
            handle_t h = graph.get_handle(xp.node_id_of_rank(r), false);
            graph.follow_edges(h, false, [&](const handle_t& nb) {   // outgoing only
                std::uint64_t nr = xp.rank_of_handle(nb);
                if (chromosome.empty() || nr < N) {                  // stay within the chromosome
                    lo << r << '\t' << nr << '\n';
                    ++nlinks;
                }
                return true;
            });
        }
        lo.close();
        std::cerr << "[gbz2layout] wrote " << lp << " (" << nlinks << " links)\n";
    }
    return 0;
}
