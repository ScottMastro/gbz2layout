// xp_probe — M1 load-bearing experiment (step 1: path exposure + step count).
//
// Loads a GBZ, reports graph/GBWT stats, and answers the §10 open question:
// does GBWTGraph::for_each_path_handle expose ALL haplotype threads, or only
// named/reference paths? Counts total exposed path-steps (the quantity XP
// memory scales with). Structure-building + RSS measurement added next.

#include <gbwtgraph/gbz.h>
#include <gbwtgraph/gbwtgraph.h>
#include <gbwt/gbwt.h>

#include <sdsl/int_vector.hpp>
#include <sdsl/enc_vector.hpp>
#include <sdsl/bit_vectors.hpp>
#include <sdsl/util.hpp>

#include <sys/resource.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static long peak_rss_kb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;   // kbytes on Linux
}

static double mib(std::uint64_t bytes) { return bytes / (1024.0 * 1024.0); }

using clock_t_ = std::chrono::steady_clock;

static double secs_since(clock_t_::time_point t0) {
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: xp_probe <graph.gbz> [--stats-only]\n";
        std::cerr << "  --stats-only: load + report graph/GBWT stats + peak RSS,\n";
        std::cerr << "                skip path cache / XP build (safe on whole-genome GBZ)\n";
        return 1;
    }
    const std::string gbz_path = argv[1];
    bool stats_only = false;
    for (int i = 2; i < argc; ++i)
        if (std::string(argv[i]) == "--stats-only") stats_only = true;

    auto t0 = clock_t_::now();
    std::cerr << "[xp_probe] loading " << gbz_path << " ...\n";
    gbwtgraph::GBZ gbz;
    gbwt::GBWT& index = gbz.index;          // reference into the GBZ
    {
        std::ifstream in(gbz_path, std::ios::binary);
        if (!in) { std::cerr << "error: cannot open " << gbz_path << "\n"; return 1; }
        gbz.simple_sds_load(in);
    }
    const gbwtgraph::GBWTGraph& graph = gbz.graph;
    std::cerr << "[xp_probe] loaded in " << secs_since(t0) << " s\n\n";

    // --- graph-level stats -------------------------------------------------
    const std::uint64_t node_count = graph.get_node_count();
    const std::int64_t  min_id     = graph.min_node_id();
    const std::int64_t  max_id     = graph.max_node_id();
    const std::uint64_t path_count = graph.get_path_count();

    std::cout << "=== graph ===\n";
    std::cout << "node_count        " << node_count << "\n";
    std::cout << "min_node_id       " << min_id << "\n";
    std::cout << "max_node_id       " << max_id << "\n";
    std::cout << "id_span           " << (max_id - min_id + 1)
              << (std::uint64_t(max_id - min_id + 1) == node_count ? "  (DENSE)" : "  (SPARSE!)")
              << "\n";
    std::cout << "get_path_count()  " << path_count << "\n";

    // --- GBWT-level stats: how many haplotype threads actually exist? -------
    std::cout << "\n=== gbwt ===\n";
    std::cout << "sequences()       " << index.sequences()
              << "   (2 per haplotype: fwd+rev)\n";
    if (index.hasMetadata()) {
        const gbwt::Metadata& md = index.metadata;
        std::cout << "metadata.haplotypes() " << md.haplotypes() << "\n";
        std::cout << "metadata.paths()      " << md.paths() << "\n";
        std::cout << "metadata.samples()    " << md.samples() << "\n";
        std::cout << "metadata.contigs()    " << md.contigs() << "\n";
    } else {
        std::cout << "metadata          <none>\n";
    }

    if (stats_only) {
        // whole-genome-safe: no path cache, no XP build. Just load footprint +
        // per-contig path histogram (proves per-chromosome slicing is feasible
        // from the single whole-genome file via metadata alone).
        std::cout << "\n=== stats-only ===\n";
        std::cout << "load peak RSS     " << (peak_rss_kb() / 1024.0) << " MiB"
                  << "   (GBWTGraph resident; file was on disk)\n";

        if (index.hasMetadata() && index.metadata.hasPathNames()
            && index.metadata.hasContigNames()) {
            const gbwt::Metadata& md = index.metadata;
            std::vector<std::uint64_t> paths_per_contig(md.contigs(), 0);
            for (std::uint64_t i = 0; i < md.paths(); ++i)
                paths_per_contig[md.path(i).contig]++;
            std::cout << "\n=== per-contig path histogram (chromosome slicing) ===\n";
            std::cout << "contigs           " << md.contigs() << "\n";
            std::cout << "contig                     paths\n";
            std::uint64_t shown = 0;
            for (std::uint64_t c = 0; c < md.contigs() && shown < 30; ++c) {
                if (paths_per_contig[c] == 0) continue;
                std::cout.width(24); std::cout << std::left << md.contig(c) << "  ";
                std::cout.width(8);  std::cout << std::right << paths_per_contig[c] << "\n";
                ++shown;
            }
            if (md.contigs() > 30) std::cout << "... (" << md.contigs() << " contigs total)\n";
            std::cout << "\n=> select one chromosome by filtering forward sequences whose\n"
                         "   path contig == target, then extract only those. No subgraph op.\n";
        }
        std::cout << "\n[xp_probe] done in " << secs_since(t0) << " s\n";
        return 0;
    }

    // --- the exposure question: iterate paths, count steps -----------------
    std::cout << "\n=== path iteration (for_each_path_handle) ===\n";
    auto t1 = clock_t_::now();
    std::uint64_t exposed_paths = 0;
    std::uint64_t total_steps   = 0;
    std::uint64_t total_bp      = 0;
    std::uint64_t max_path_steps = 0;

    graph.for_each_path_handle([&](const handlegraph::path_handle_t& path) {
        std::uint64_t steps = 0;
        graph.for_each_step_in_path(path, [&](const handlegraph::step_handle_t& step) {
            handlegraph::handle_t h = graph.get_handle_of_step(step);
            total_bp += graph.get_length(h);
            ++steps;
        });
        ++exposed_paths;
        total_steps += steps;
        if (steps > max_path_steps) max_path_steps = steps;
    });

    std::cout << "exposed_paths     " << exposed_paths << "\n";
    std::cout << "total_steps       " << total_steps << "\n";
    std::cout << "total_bp          " << total_bp << "\n";
    std::cout << "max_path_steps    " << max_path_steps << "\n";
    std::cout << "avg_steps/path    "
              << (exposed_paths ? total_steps / exposed_paths : 0) << "\n";
    std::cout << "iterated in       " << secs_since(t1) << " s\n";

    // --- GBWT-direct iteration + path cache --------------------------------
    // for_each_path_handle only exposed reference/generic paths. The haplotype
    // threads live in the GBWT as sequences (2 per haplotype path: fwd + rev).
    // Forward sequences are the even ids. Cache each path as node ranks so the
    // cap sweep below can rebuild XP many times without re-extracting.
    std::cout << "\n=== gbwt-direct iteration + path cache ===\n";
    auto t2 = clock_t_::now();
    const std::uint64_t n_seq   = index.sequences();
    const std::uint64_t id_span = std::uint64_t(max_id - min_id + 1);

    // id -> dense rank remap (ids are sparse). rank[id-min] = dense+1; 0=absent.
    std::vector<std::uint32_t> id2rank(id_span, 0);
    std::vector<std::uint32_t> node_len(node_count, 0);
    std::uint32_t next_rank = 0;
    graph.for_each_handle([&](const handlegraph::handle_t& h) {
        std::uint64_t id = graph.get_id(h);
        id2rank[id - min_id] = ++next_rank;              // 1-based
        node_len[next_rank - 1] = graph.get_length(h);
    });
    auto rank_of = [&](std::uint64_t id) -> std::uint32_t {
        return id2rank[id - min_id] - 1;                 // dense 0-based
    };
    const std::uint64_t id2rank_bytes = id2rank.size() * sizeof(std::uint32_t);

    // cache paths as rank sequences; count full steps per node & coverage
    std::vector<std::vector<std::uint32_t>> paths_ranks;
    std::vector<std::uint64_t> full_spn(node_count, 0);   // full steps per node
    std::uint64_t gbwt_paths = 0, gbwt_steps = 0, gbwt_bp = 0, empty_paths = 0;
    for (std::uint64_t seq = 0; seq < n_seq; seq += 2) {   // even = forward
        gbwt::vector_type path = index.extract(seq);
        if (path.empty()) { ++empty_paths; continue; }
        std::vector<std::uint32_t> ranks;
        ranks.reserve(path.size());
        for (gbwt::node_type node : path) {
            std::uint32_t r = rank_of(gbwt::Node::id(node));
            ranks.push_back(r);
            full_spn[r]++;
            gbwt_bp += node_len[r];
        }
        gbwt_steps += ranks.size();
        paths_ranks.push_back(std::move(ranks));
        ++gbwt_paths;
    }
    std::uint64_t covered_nodes = 0;
    for (std::uint64_t r = 0; r < node_count; ++r) covered_nodes += (full_spn[r] > 0);

    std::cout << "gbwt_paths        " << gbwt_paths << "\n";
    std::cout << "empty_paths       " << empty_paths << "\n";
    std::cout << "gbwt_steps        " << gbwt_steps
              << "   (vs " << total_steps << " via handlegraph = "
              << (total_steps ? double(gbwt_steps) / total_steps : 0) << "x)\n";
    std::cout << "gbwt_bp           " << gbwt_bp << "\n";
    std::cout << "avg steps/node    " << (node_count ? double(gbwt_steps) / node_count : 0) << "\n";
    std::cout << "covered_nodes     " << covered_nodes << " / " << node_count
              << "   (" << (node_count ? 100.0 * covered_nodes / node_count : 0) << "%)\n";
    std::cout << "cached in         " << secs_since(t2) << " s\n";

    // --- reference-order check ---------------------------------------------
    // odgi's default init ('d') lays nodes along X in for_each_handle (storage)
    // order. If storage order already tracks the reference path, that init is
    // fine even unsorted; if it's scrambled, reference-anchored init matters.
    // Measure: walk the reference path, collect node ranks, and correlate path
    // position vs node rank (Pearson r) + fraction of adjacent steps increasing.
    std::cout << "\n=== reference-order check ===\n";
    std::vector<double> ref_ranks;
    graph.for_each_path_handle([&](const handlegraph::path_handle_t& path) {
        graph.for_each_step_in_path(path, [&](const handlegraph::step_handle_t& step) {
            handlegraph::handle_t h = graph.get_handle_of_step(step);
            ref_ranks.push_back(double(rank_of(graph.get_id(h))));
        });
    });
    if (ref_ranks.size() >= 2) {
        const std::uint64_t L = ref_ranks.size();
        auto pearson = [&](const std::vector<double>& y) {
            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, n = double(L);
            for (std::uint64_t i = 0; i < L; ++i) {
                double x = double(i);
                sx += x; sy += y[i]; sxx += x * x; syy += y[i] * y[i]; sxy += x * y[i];
            }
            double cov = sxy - sx * sy / n, vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
            return (vx > 0 && vy > 0) ? cov / std::sqrt(vx * vy) : 0.0;
        };
        double r_pearson = pearson(ref_ranks);

        // Spearman = Pearson on value-ranks (robust to nonlinearity)
        std::vector<std::uint64_t> ord(L);
        for (std::uint64_t i = 0; i < L; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(),
                  [&](std::uint64_t a, std::uint64_t b) { return ref_ranks[a] < ref_ranks[b]; });
        std::vector<double> vrank(L);
        for (std::uint64_t k = 0; k < L; ++k) vrank[ord[k]] = double(k);
        double r_spearman = pearson(vrank);

        // adjacent jump distribution (how near are consecutive reference nodes in storage?)
        std::uint64_t inc = 0;
        std::vector<std::uint64_t> absd(L - 1);
        for (std::uint64_t i = 0; i + 1 < L; ++i) {
            double d = ref_ranks[i + 1] - ref_ranks[i];
            if (d > 0) ++inc;
            absd[i] = (std::uint64_t)std::llabs((long long)d);
        }
        std::sort(absd.begin(), absd.end());
        std::uint64_t med = absd[absd.size() / 2];
        std::uint64_t p90 = absd[(absd.size() * 9) / 10];

        std::cout << "reference steps   " << L << "\n";
        std::cout << "Pearson(pos,rank) " << r_pearson << "   (linear; low if node density uneven)\n";
        std::cout << "Spearman          " << r_spearman
                  << "   (monotonic; near +1 => storage order tracks reference)\n";
        std::cout << "adj-increasing    " << (100.0 * inc / (L - 1)) << "%\n";
        std::cout << "|adj rank jump|   median " << med << ", p90 " << p90
                  << "   (small => consecutive ref nodes are near in storage)\n";
        std::cout << "verdict           "
                  << (r_spearman > 0.95 ? "storage order strongly tracks reference"
                     : r_spearman > 0.7  ? "storage order mostly tracks reference"
                     : "storage order weakly tracks reference")
                  << "; reference-anchored init recommended either way\n";
    }
    { std::vector<double>().swap(ref_ranks); }

    // pos_map_iv: cumulative bp per node (rank order) -> enc_vector (cap-independent)
    sdsl::int_vector<> position_map(node_count + 1, 0);
    { std::uint64_t acc = 0;
      for (std::uint64_t r = 0; r < node_count; ++r) { position_map[r] = acc; acc += node_len[r]; }
      position_map[node_count] = acc; }
    sdsl::enc_vector<> pos_map_iv(position_map);
    { sdsl::int_vector<> tmp; std::swap(tmp, position_map); }
    const std::uint64_t b_pos = sdsl::size_in_bytes(pos_map_iv);

    // --- per-node coverage cap sweep ---------------------------------------
    // Cap C: index at most C occurrences of each node (keep the first C seen).
    // Rare-variant nodes (<=C occurrences) are fully preserved -> coverage stays
    // 100%; the redundant backbone is thinned. C=0 means uncapped (full).
    //
    // We report two totals per cap:
    //   full  = every XP array incl. positions + offsets (odgi-faithful)
    //   lean  = drop positions & offsets (derive them from handles at query time)
    // 'lean' is the design we're targeting; 'full' shows the odgi-equivalent cost.
    // pre-size int_vectors at their target bit-widths so the transient
    // pre-compress peak matches the compressed size (avoids 64-bit blowup:
    // 578M steps x 8B x 4 arrays = 18 GB otherwise).
    std::uint64_t max_steps_path = 0, max_bp_path = 0;
    for (const auto& pr : paths_ranks) {
        std::uint64_t bp = 0; for (std::uint32_t r : pr) bp += node_len[r];
        if (pr.size() > max_steps_path) max_steps_path = pr.size();
        if (bp > max_bp_path) max_bp_path = bp;
    }
    auto wbits = [](std::uint64_t x) -> std::uint8_t { return x ? (std::uint8_t)(sdsl::bits::hi(x) + 1) : 1; };
    const std::uint8_t w_hand = wbits(node_count ? node_count - 1 : 0);
    const std::uint8_t w_npi  = wbits(paths_ranks.empty() ? 0 : paths_ranks.size() - 1);
    const std::uint8_t w_nr   = wbits(max_steps_path);
    const std::uint8_t w_pos  = wbits(max_bp_path);

    std::cout << "\n=== per-node coverage cap sweep ===\n";
    std::cout << "widths: handles=" << (int)w_hand << " npi=" << (int)w_npi
              << " nr=" << (int)w_nr << " positions=" << (int)w_pos << " bits\n";
    std::cout << "cap    ret_steps     ret%   cover%   full_MiB   lean_MiB   lean_b/step\n";

    auto sz = [](const auto& s) { return (std::uint64_t)sdsl::size_in_bytes(s); };

    auto measure = [&](std::uint64_t cap) {
        // capped steps per node & prefix (node-grouped block layout)
        std::vector<std::uint64_t> spn(node_count);
        std::uint64_t np = 0;
        for (std::uint64_t r = 0; r < node_count; ++r) {
            spn[r] = (cap == 0) ? full_spn[r] : std::min<std::uint64_t>(full_spn[r], cap);
            np += spn[r];
        }
        std::vector<std::uint64_t> node_start(node_count + 1, 0);
        for (std::uint64_t r = 0; r < node_count; ++r) node_start[r + 1] = node_start[r] + spn[r];

        sdsl::bit_vector   np_bv(np, 0);
        sdsl::int_vector<> nr_iv(np, 0, w_nr);
        sdsl::int_vector<> npi_iv(np, 0, w_npi);
        sdsl::int_vector<> handles(np, 0, w_hand);
        sdsl::int_vector<> positions(np, 0, w_pos);   // full-only
        for (std::uint64_t r = 0; r < node_count; ++r) np_bv[node_start[r]] = 1;

        std::vector<std::uint64_t> cursor(node_start.begin(), node_start.end());
        std::vector<std::uint32_t> kept(node_count, 0);
        std::uint64_t hpos = 0, retained_bp = 0;
        for (std::uint64_t pid = 0; pid < paths_ranks.size(); ++pid) {
            std::uint64_t rank_in_path = 0, bp = 0;
            for (std::uint32_t r : paths_ranks[pid]) {
                ++rank_in_path;                              // true rank (over all steps)
                std::uint64_t here = bp; bp += node_len[r];  // true bp position
                if (cap != 0 && kept[r] >= cap) continue;    // node cap reached -> skip
                ++kept[r];
                std::uint64_t idx = cursor[r]++;
                nr_iv[idx] = rank_in_path;
                npi_iv[idx] = pid;
                handles[hpos] = r;
                positions[hpos] = here;
                ++hpos;
                retained_bp += node_len[r];
            }
        }
        sdsl::util::bit_compress(nr_iv);
        sdsl::util::bit_compress(npi_iv);
        sdsl::util::bit_compress(handles);
        sdsl::util::bit_compress(positions);
        sdsl::bit_vector::select_1_type np_bv_select(&np_bv);

        std::uint64_t b_npbv = sz(np_bv) + sz(np_bv_select);
        std::uint64_t b_nr = sz(nr_iv), b_npi = sz(npi_iv), b_hand = sz(handles), b_posn = sz(positions);

        // offsets: a per-path bp bitvector (sized by bp, NOT steps) + select.
        // Estimate its cost rather than build 1614 of them: raw bits = retained_bp,
        // plus ~ n/ (select overhead ~ 0.2 * set_bits worth). Here 'full' keeps it.
        std::uint64_t b_off = (retained_bp + 7) / 8 + (np * 8) / 5;   // bitvector + ~select

        std::uint64_t lean = id2rank_bytes + b_pos + b_npbv + b_nr + b_npi + b_hand;
        std::uint64_t full = lean + b_posn + b_off;

        std::cout.setf(std::ios::fixed); std::cout.precision(1);
        std::cout.width(4);  std::cout << std::left << (cap == 0 ? 0 : cap) << (cap==0?"* ":"  ");
        std::cout.width(11); std::cout << std::right << np << "  ";
        std::cout.width(6);  std::cout << (100.0 * np / gbwt_steps) << "  ";
        std::cout.width(6);  std::cout << 100.0 << "  ";     // coverage always 100% (cap>=1)
        std::cout.width(9);  std::cout << mib(full) << "  ";
        std::cout.width(9);  std::cout << mib(lean) << "  ";
        std::cout.width(9);  std::cout << (8.0 * lean / np) << "\n";
    };

    for (std::uint64_t cap : {std::uint64_t(0), 60ul, 30ul, 15ul, 8ul, 4ul, 2ul, 1ul}) measure(cap);
    std::cout << "(* cap 0 = uncapped/full; cover% is 100 for any cap>=1 since the first\n"
                 " occurrence of every node is always kept)\n";

    std::cout << "\npeak RSS          " << (peak_rss_kb() / 1024.0) << " MiB"
              << "   (probe holds full path cache; the real builder streams)\n";
    std::cout << "\n[xp_probe] done in " << secs_since(t0) << " s\n";
    return 0;
}
