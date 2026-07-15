#include "compartment.hpp"

#include <algorithm>
#include <deque>
#include <iostream>
#include <queue>

namespace gbz2layout {

using handlegraph::handle_t;

CompartmentResult build_compartments(const gbwtgraph::GBZ& gbz,
                                     const gbwtgraph::GBWTGraph& graph,
                                     const XP& xp,
                                     std::uint64_t target_tasks,
                                     std::uint64_t pinch_window,
                                     std::int64_t genome_min_id,
                                     const std::vector<bool>* node_mask,
                                     bool verbose) {
    const std::uint64_t N = xp.node_count();
    CompartmentResult res;
    res.region.assign(N, -2);           // -2 = unassigned (filled below)
    res.freeze.assign(N, false);

    // local id -> rank map over the graph span (safe against sparse ids)
    const std::int64_t gmax = graph.max_node_id();
    const std::uint64_t span = std::uint64_t(gmax - genome_min_id + 1);
    std::vector<std::int32_t> id2rank(span, -1);
    for (std::uint64_t r = 0; r < N; ++r) id2rank[xp.node_id_of_rank(r) - genome_min_id] = (std::int32_t)r;

    // (1) per-rank distinct-haplotype coverage from the GBWT
    const gbwt::GBWT& index = gbz.index;
    std::vector<std::int32_t> cov(N, 0), lastseq(N, -1);
    const std::uint64_t n_seq = index.sequences();
    auto in_graph = [&](std::uint64_t id){ return id >= (std::uint64_t)genome_min_id && id <= (std::uint64_t)gmax && id2rank[id - genome_min_id] >= 0; };
    for (std::uint64_t seq = 0; seq < n_seq; seq += 2) {
        if (node_mask) {   // cheap skip of other chromosomes without full extract
            gbwt::edge_type e = index.start(seq);
            if (e.first == gbwt::ENDMARKER) continue;
            if (!in_graph(gbwt::Node::id(e.first))) continue;
        }
        gbwt::vector_type path = index.extract(seq);
        std::int32_t sid = (std::int32_t)(seq / 2);
        for (gbwt::node_type node : path) {
            std::uint64_t id = gbwt::Node::id(node);
            if (!in_graph(id)) continue;
            std::int32_t r = id2rank[id - genome_min_id];
            if (lastseq[r] != sid) { cov[r]++; lastseq[r] = sid; }
        }
    }

    // (2) walk the reference in order; local-envelope pinch detection
    const auto& rr = xp.ref_ranks();
    const std::uint64_t R = rr.size();
    std::vector<std::int32_t> rcov(R);
    for (std::uint64_t i = 0; i < R; ++i) rcov[i] = cov[rr[i]];
    // sliding-window max over +/- pinch_window
    std::vector<char> is_pinch(R, 0);
    {
        std::deque<std::uint64_t> dq; std::uint64_t rd = 0;
        for (std::uint64_t i = 0; i < R; ++i) {
            std::uint64_t hi = std::min(R, i + pinch_window + 1);
            while (rd < hi) { while (!dq.empty() && rcov[dq.back()] <= rcov[rd]) dq.pop_back(); dq.push_back(rd++); }
            std::uint64_t lo = (i > pinch_window) ? i - pinch_window : 0;
            while (!dq.empty() && dq.front() < lo) dq.pop_front();
            if (rcov[i] >= rcov[dq.front()] && rcov[i] > 0) is_pinch[i] = 1;
        }
    }
    for (std::uint64_t i = 0; i < R; ++i) res.n_pinches += is_pinch[i];

    // (3) balanced cut selection: cut at the next pinch once the running
    // reference-node count reaches R / target_tasks. Force boundaries at ends.
    std::uint64_t target = std::max<std::uint64_t>(1, R / std::max<std::uint64_t>(1, target_tasks));
    std::int32_t chunk = 0;
    std::uint64_t since = 0;
    std::vector<std::int32_t> ref_region(R, 0);
    auto make_boundary = [&](std::uint64_t i){
        std::uint32_t rank = rr[i];
        res.region[rank] = -1;          // anchor: belongs to every compartment
        res.freeze[rank] = true;
        res.boundary_ranks.push_back(rank);
    };
    for (std::uint64_t i = 0; i < R; ++i) {
        ref_region[i] = chunk;
        bool force = (i == 0 || i == R - 1);
        if (force || (since >= target && is_pinch[i])) {
            make_boundary(i);
            if (i != 0) { chunk++; }     // subsequent nodes belong to next chunk
            ref_region[i] = -1;
            since = 0;
        }
        ++since;
    }
    res.n_compartments = (std::uint64_t)chunk;   // number of interior spans

    // assign interior reference nodes their chunk id (skip boundaries already -1)
    for (std::uint64_t i = 0; i < R; ++i) {
        std::uint32_t rank = rr[i];
        if (res.region[rank] == -1) continue;    // boundary anchor
        res.region[rank] = ref_region[i];
    }

    // (4) BFS-propagate chunk id from interior reference nodes into bubbles
    std::queue<std::uint32_t> q;
    for (std::uint64_t i = 0; i < R; ++i) {
        std::uint32_t rank = rr[i];
        if (res.region[rank] >= 0) q.push(rank);
    }
    while (!q.empty()) {
        std::uint32_t r = q.front(); q.pop();
        std::int32_t rg = res.region[r];
        handle_t h = graph.get_handle(xp.node_id_of_rank(r), false);
        for (bool go_left : {false, true})
            graph.follow_edges(h, go_left, [&](const handle_t& nb) {
                std::uint64_t nr = xp.rank_of_handle(nb);
                if (nr < N && res.region[nr] == -2) { res.region[nr] = rg; q.push((std::uint32_t)nr); }
                return true;
            });
    }
    std::uint64_t leftover = 0;
    for (std::uint64_t r = 0; r < N; ++r) if (res.region[r] == -2) { res.region[r] = 0; ++leftover; }

    // (5) balance stats: node count per compartment
    std::vector<std::uint64_t> counts(res.n_compartments + 1, 0);
    for (std::uint64_t r = 0; r < N; ++r) { std::int32_t g = res.region[r]; if (g >= 0 && (std::uint64_t)g <= res.n_compartments) counts[g]++; }
    std::vector<std::uint64_t> nz;
    for (auto c : counts) if (c) nz.push_back(c);
    std::sort(nz.begin(), nz.end());
    if (!nz.empty()) {
        res.min_nodes = nz.front(); res.max_nodes = nz.back();
        res.median_nodes = nz[nz.size() / 2];
    }

    if (verbose) {
        std::cerr << "[compartments] " << res.n_pinches << " candidate pinches, target "
                  << target_tasks << " tasks -> " << res.n_compartments << " compartments, "
                  << res.boundary_ranks.size() << " boundary anchors\n";
        std::cerr << "[compartments] node balance: min " << res.min_nodes
                  << " median " << res.median_nodes << " max " << res.max_nodes
                  << " (leftover " << leftover << ")\n";
    }
    return res;
}

} // namespace gbz2layout
