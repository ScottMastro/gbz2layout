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
#include "sgd_minibatch.hpp"
#include "export_gbz.hpp"

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
#include <set>
#include <algorithm>
#include <string>
#include <vector>

using namespace gbz2layout;

static void usage() {
    std::cerr <<
    "usage: gbz2layout <graph.gbz> -o <out_prefix> [options]\n"
    "  -o PREFIX          output prefix (writes PREFIX.lay.tsv)\n"
    "  --chromosome NAME  restrict to NAME's connected component (BFS from its reference path)\n"
    "  --threads N        worker threads [hw]\n"
    "  --iter N           SGD iterations; with --path-batch, one full pass over the paths [30]\n"
    "  --init ref|random  initialization [ref]\n"
    "  --seed N           RNG seed [9399220]\n"
    "\n"
    " term sampling (pick one; default is the capped full path index)\n"
    "  --cap N            per-node coverage cap (0 = uncapped) [30]\n"
    "  --updates-per-node M   min_term_updates = M * node_count (else 10*steps)\n"
    "  --path-batch K     whole-path minibatch: K whole paths resident per group, no per-node cap.\n"
    "                     Each --iter is a full pass over every path in K-sized groups, so K bounds\n"
    "                     peak RAM without changing coverage or total updates [64]\n"
    "  --updates-mult M   minibatch updates/group = M * group steps [10]\n"
    "  --window-len L     minibatch: draw the paired sample within L steps of the anchor\n"
    "                     (0 = whole-path, unbounded) [0]\n"
    "  --gpu              run the minibatch update on the GPU; requires --path-batch and a CUDA\n"
    "                     build (`make tool`). `make tool-nocuda` aborts on this flag.\n"
    "\n"
    " extra outputs / export\n"
    "  --emit-links       also write PREFIX.links.tsv\n"
    "  --emit-meta        also write PREFIX.meta.tsv (reference nodes)\n"
    "  --export-gbz FILE  with --chromosome: write a standalone per-chromosome GBZ to FILE and exit (no layout)\n"
    "  --export-all-gbz DIR  write DIR/<contig>.v2.gbz for every reference contig (one load) and exit\n";
}

// BFS a chromosome's connected component from its reference path; fills `mask`
// (indexed by node_id - genome_min_id) with the component's nodes. Returns the
// number of nodes marked, or 0 if no reference path matches `chromosome`.
// (MC chromosomes are disjoint components, so the BFS stays within one.)
static std::uint64_t compute_chromosome_mask(const gbwtgraph::GBWTGraph& graph,
                                             const std::string& chromosome,
                                             std::int64_t genome_min_id,
                                             std::vector<bool>& mask) {
    std::int64_t gmax = graph.max_node_id();
    std::uint64_t span = std::uint64_t(gmax - genome_min_id + 1);
    mask.assign(span, false);
    bool found = false;
    std::queue<std::uint64_t> bfs;
    graph.for_each_path_handle([&](const handlegraph::path_handle_t& path) {
        if (found) return;
        std::string nm = graph.get_path_name(path);
        std::string contig = nm.substr(nm.find_last_of('#') + 1);
        auto br = contig.find('[');            // strip GBWTGraph fragment suffix contig[offset]
        if (br != std::string::npos) contig = contig.substr(0, br);
        if (contig != chromosome) return;
        found = true;
        graph.for_each_step_in_path(path, [&](const handlegraph::step_handle_t& s) {
            std::uint64_t id = graph.get_id(graph.get_handle_of_step(s));
            if (!mask[id - genome_min_id]) { mask[id - genome_min_id] = true; bfs.push(id); }
        });
    });
    if (!found) return 0;
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
    return marked;
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
    bool emit_links = false;   // also write PREFIX.links.tsv (edges) for rendering
    bool emit_meta = false;    // also write PREFIX.meta.tsv (rank, is_ref) for analysis
    std::uint64_t path_batch = 0;     // >0: whole-path minibatch SGD, K whole paths per group
    std::uint64_t updates_mult = 10;  // minibatch updates/group = mult * group steps
    std::uint64_t window_len = 0;     // sub-path sampling: pair within L steps of anchor (0 = whole path)
    bool gpu = false;                 // run the minibatch update on the GPU
    std::string export_gbz;           // if set (+ --chromosome): write a standalone per-chr GBZ and exit
    std::string export_all;           // if set: write a per-chr GBZ for every reference contig into this dir and exit

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
        else if (a == "--emit-links") emit_links = true;
        else if (a == "--emit-meta") emit_meta = true;
        else if (a == "--path-batch") path_batch = std::stoull(next());
        else if (a == "--updates-mult") updates_mult = std::stoull(next());
        else if (a == "--window-len") window_len = std::stoull(next());
        else if (a == "--gpu") gpu = true;
        else if (a == "--export-gbz") export_gbz = next();
        else if (a == "--export-all-gbz") export_all = next();
        else { std::cerr << "unknown arg: " << a << "\n"; usage(); return 1; }
    }
    if (out_prefix.empty() && export_gbz.empty() && export_all.empty()) { std::cerr << "error: -o required\n"; return 1; }

    // load GBZ
    std::cerr << "[gbz2layout] loading " << gbz_path << "\n";
    gbwtgraph::GBZ gbz;
    { std::ifstream in(gbz_path, std::ios::binary);
      if (!in) { std::cerr << "cannot open " << gbz_path << "\n"; return 1; }
      gbz.simple_sds_load(in); }
    const gbwtgraph::GBWTGraph& graph = gbz.graph;

    std::int64_t genome_min_id = graph.min_node_id();

    // ---- batch: write a per-chromosome GBZ for every reference contig, then exit.
    // Load the whole-genome GBZ once and stream each chromosome out (vs reloading
    // 5 GB per chromosome). Largest components go last, so an OOM on chr1 only
    // costs chr1 — everything else is already written (skip-if-exists = resumable).
    if (!export_all.empty()) {
        std::vector<std::string> contigs; std::set<std::string> seen;
        graph.for_each_path_handle([&](const handlegraph::path_handle_t& p) {
            std::string nm = graph.get_path_name(p);
            std::string c = nm.substr(nm.find_last_of('#') + 1);
            auto br = c.find('[');             // fragmented reference paths: contig[offset]
            if (br != std::string::npos) c = c.substr(0, br);
            if (seen.insert(c).second) contigs.push_back(c);
        });
        std::cerr << "[export-all] " << contigs.size() << " reference contigs; sizing components\n";
        struct ChrMask { std::string name; std::vector<bool> mask; std::uint64_t marked; };
        std::vector<ChrMask> chrs;
        for (const std::string& c : contigs) {
            std::vector<bool> m;
            std::uint64_t marked = compute_chromosome_mask(graph, c, genome_min_id, m);
            if (marked == 0) { std::cerr << "[export-all] " << c << ": no ref path, skip\n"; continue; }
            chrs.push_back({c, std::move(m), marked});
        }
        std::sort(chrs.begin(), chrs.end(),
                  [](const ChrMask& a, const ChrMask& b){ return a.marked < b.marked; });
        std::uint64_t ok_count = 0;
        for (const ChrMask& cm : chrs) {
            std::string out = export_all + "/" + cm.name + ".v2.gbz";
            { std::ifstream probe(out, std::ios::binary);
              if (probe.good()) { std::cerr << "[export-all] " << cm.name << ": exists, skip\n"; ++ok_count; continue; } }
            std::cerr << "[export-all] " << cm.name << ": " << cm.marked << " nodes\n";
            XP cxp;
            cxp.build(gbz, 1, false, &cm.mask, genome_min_id);
            if (export_chromosome_gbz(gbz, cxp, cm.mask, genome_min_id, cm.name, out, true)) ++ok_count;
        }
        std::cerr << "[export-all] done: " << ok_count << "/" << chrs.size() << " written\n";
        return 0;
    }

    // optional: extract a single chromosome from a whole-genome GBZ by BFS'ing
    // its connected component (MC chromosomes are disjoint components).
    std::vector<bool> mask;
    if (!chromosome.empty()) {
        std::uint64_t marked = compute_chromosome_mask(graph, chromosome, genome_min_id, mask);
        if (marked == 0) { std::cerr << "error: no reference path for contig '" << chromosome << "'\n"; return 1; }
        std::cerr << "[gbz2layout] chromosome " << chromosome << ": " << marked
                  << " nodes in component\n";
    }

    // build XP (filtered to the chromosome if a mask was computed). In minibatch
    // mode we only need node metadata + the path list, not the full step index,
    // so build a cap-1 (tiny) XP and stream whole paths per iteration instead.
    XP xp;
    std::uint64_t xp_cap = (path_batch > 0 || !export_gbz.empty()) ? 1 : cap;
    xp.build(gbz, xp_cap, true, chromosome.empty() ? nullptr : &mask, genome_min_id);
    const std::uint64_t N = xp.node_count();

    // ---- optional: write a standalone single-chromosome GBZ and exit ----
    // (extracts this chromosome's nodes+threads into a compact GBZ so downstream
    // layout / pangyplot ingest loads at chromosome scale, not whole-genome.)
    if (!export_gbz.empty()) {
        if (chromosome.empty()) {
            std::cerr << "error: --export-gbz requires --chromosome\n"; return 1;
        }
        bool ok = export_chromosome_gbz(gbz, xp, mask, genome_min_id,
                                        chromosome, export_gbz, true);
        return ok ? 0 : 1;
    }

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

    // ---- SGD params (odgi defaults) ----
    SgdParams p;
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
    if (path_batch > 0) {
        // whole-path minibatch: stream K paths/iteration from the GBWT
        std::uint64_t full_max = xp.max_full_path_step_count();
        MinibatchParams mp;
        mp.iter_max = iter_max;
        mp.batch_paths = path_batch > 0 ? path_batch : 64;
        mp.updates_mult = updates_mult;
        mp.nthreads = threads;
        mp.seed = seed;
        mp.space = full_max;
        mp.space_max = std::min<std::uint64_t>(mp.space, 1000);
        mp.space_quantization_step = 100;
        mp.eta_max = (double)full_max * (double)full_max;
        mp.theta = 0.99; mp.eps = 0.01; mp.cooling_start = 0.5;
        mp.window_len = window_len;
        mp.use_gpu = gpu;

        std::cerr << "[gbz2layout] minibatch: iter=" << mp.iter_max << " K=" << mp.batch_paths
                  << " updates_mult=" << mp.updates_mult << " threads=" << mp.nthreads
                  << " full_max_steps=" << full_max << "\n";
        path_linear_sgd_layout_minibatch(graph, gbz.index, xp, mp, X, Y);
    } else {
        path_linear_sgd_layout(graph, xp, p, X, Y);
    }
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

    // ---- optional: emit per-node meta (rank, is_ref) for external analysis ----
    if (emit_meta) {
        std::vector<char> is_ref(N, 0);
        if (xp.has_reference()) for (std::uint32_t r : xp.ref_ranks()) is_ref[r] = 1;
        bool have_seg = graph.has_segment_names();
        std::string mp = out_prefix + ".meta.tsv";
        std::ofstream mo(mp);
        mo << "rank\tis_ref\tsegment\n";      // segment = original GFA segment name (pre-chop)
        for (std::uint64_t r = 0; r < N; ++r) {
            mo << r << '\t' << (int)is_ref[r] << '\t';
            if (have_seg) mo << graph.get_segment_name(graph.get_handle(xp.node_id_of_rank(r), false));
            else          mo << r;
            mo << '\n';
        }
        mo.close();
        std::uint64_t nref = 0; for (char c : is_ref) nref += c;
        std::cerr << "[gbz2layout] wrote " << mp << " (" << nref << " reference nodes)\n";
    }

    return 0;
}
