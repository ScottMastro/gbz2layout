// pinch — probe PATH pinches: reference nodes that every haplotype passes
// through. Unlike compartments.cpp (topological articulation points, fooled by
// repeat back-edges into giant biconnected blocks), this measures the cut
// points that matter for LAYOUT, since PG-SGD only uses path distances. All
// haplotypes converging on one node means the path-constraint graph can be
// split there cleanly, regardless of any bypass edge.
//
// Method: per-node distinct-sequence coverage over the GBWT (last-seq trick),
// then walk the reference path and flag nodes with coverage >= frac * H, where
// H is the peak coverage (the backbone). Report the distribution of bp gaps
// between consecutive pinches = the path-compartment sizes.
//
// usage: pinch <graph.gbz> [frac=1.0]

#include <gbwtgraph/gbz.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <deque>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: pinch <graph.gbz> [frac=1.0]\n"; return 1; }
    std::string gbz_path = argv[1];
    double frac = (argc >= 3) ? std::stod(argv[2]) : 1.0;

    gbwtgraph::GBZ gbz;
    { std::ifstream in(gbz_path, std::ios::binary); if (!in) { std::cerr << "cannot open\n"; return 1; } gbz.simple_sds_load(in); }
    const gbwtgraph::GBWTGraph& graph = gbz.graph;
    const gbwt::GBWT& index = gbz.index;

    std::int64_t minid = graph.min_node_id(), maxid = graph.max_node_id();
    std::uint64_t span = std::uint64_t(maxid - minid + 1);

    // per-node distinct-sequence coverage
    std::vector<std::int32_t> cov(span, 0), lastseq(span, -1);
    const std::uint64_t n_seq = index.sequences();
    std::uint64_t n_paths = 0;
    for (std::uint64_t seq = 0; seq < n_seq; seq += 2) {   // forward sequences
        gbwt::vector_type path = index.extract(seq);
        if (path.empty()) continue;
        ++n_paths;
        std::int32_t sid = (std::int32_t)(seq / 2);
        for (gbwt::node_type node : path) {
            std::uint64_t k = gbwt::Node::id(node) - minid;
            if (lastseq[k] != sid) { cov[k]++; lastseq[k] = sid; }
        }
    }
    std::int32_t H = 0;
    for (std::uint64_t k = 0; k < span; ++k) H = std::max(H, cov[k]);
    std::int32_t thresh = (std::int32_t)std::ceil(frac * H);
    std::cerr << "[pinch] " << n_paths << " paths, peak node coverage H=" << H
              << ", pinch threshold cov>=" << thresh << " (frac " << frac << ")\n";

    // walk the reference path (the one path exposed via the handlegraph API),
    // accumulate bp position, flag pinches, record gaps between consecutive ones.
    bool found = false;
    std::uint64_t ref_nodes = 0, pinches = 0;
    std::uint64_t last_pinch_bp = 0; bool have_last = false;
    std::vector<std::uint64_t> gaps;         // bp between consecutive pinches
    std::vector<std::uint64_t> gap_nodes;    // ref-node count between consecutive pinches
    std::uint64_t nodes_since = 0;
    std::uint64_t bp = 0;
    std::vector<std::int32_t> ref_cov;       // coverage of every reference node (diagnostic)
    std::vector<std::uint64_t> ref_bp;       // cumulative bp position at each reference node
    graph.for_each_path_handle([&](const handlegraph::path_handle_t& p) {
        if (found) return;   // first (reference) path only
        found = true;
        graph.for_each_step_in_path(p, [&](const handlegraph::step_handle_t& s) {
            handlegraph::handle_t h = graph.get_handle_of_step(s);
            std::uint64_t id = graph.get_id(h);
            std::uint64_t len = graph.get_length(h);
            ref_nodes++;
            nodes_since++;
            ref_cov.push_back(cov[id - minid]);
            ref_bp.push_back(bp);
            if (cov[id - minid] >= thresh) {
                pinches++;
                if (have_last) { gaps.push_back(bp - last_pinch_bp); gap_nodes.push_back(nodes_since); }
                last_pinch_bp = bp; have_last = true; nodes_since = 0;
            }
            bp += len;
        });
    });
    if (!found) { std::cerr << "no reference path exposed\n"; return 1; }

    auto report = [](std::vector<std::uint64_t>& v, const char* unit){
        if (v.empty()) { std::cout << "  (no gaps)\n"; return; }
        std::sort(v.begin(), v.end());
        auto q = [&](double p){ return v[std::min(v.size()-1, (std::size_t)(p*v.size()))]; };
        std::uint64_t sum = 0; for (auto x : v) sum += x;
        std::cout << "  count " << v.size() << "  mean " << (sum / v.size()) << ' ' << unit
                  << "  median " << q(0.5) << "  p90 " << q(0.9) << "  p99 " << q(0.99)
                  << "  max " << v.back() << ' ' << unit << "\n";
        // histogram of gap sizes
        std::uint64_t h1=0,h2=0,h3=0,h4=0,h5=0;
        for (auto x : v) { if(x<=100)h1++; else if(x<=1000)h2++; else if(x<=10000)h3++; else if(x<=100000)h4++; else h5++; }
        std::cout << "    <=100:" << h1 << "  101-1k:" << h2 << "  1k-10k:" << h3
                  << "  10k-100k:" << h4 << "  >100k:" << h5 << "\n";
    };

    // DIAGNOSTIC: distribution of coverage over reference nodes. If the genome
    // is "mostly constant", most backbone nodes should sit near the peak H.
    {
        std::vector<std::int32_t> c = ref_cov;
        std::sort(c.begin(), c.end());
        auto q = [&](double p){ return c.empty()?0:c[std::min(c.size()-1,(std::size_t)(p*c.size()))]; };
        std::cout << "=== reference-node coverage distribution (H=" << H << ", " << n_paths << " paths) ===\n";
        std::cout << "min " << (c.empty()?0:c.front()) << "  p10 " << q(0.1) << "  p25 " << q(0.25)
                  << "  median " << q(0.5) << "  p75 " << q(0.75) << "  p90 " << q(0.9)
                  << "  max " << (c.empty()?0:c.back()) << "\n";
        for (double f : {0.50, 0.80, 0.90, 0.95, 0.99, 1.00}) {
            std::int32_t t = (std::int32_t)std::ceil(f * H);
            std::uint64_t n = 0; for (auto v : ref_cov) if (v >= t) n++;
            std::cout << "  cov>=" << (int)(f*100) << "%H (>=" << t << "):  " << n
                      << "  (" << (ref_nodes?100.0*n/ref_nodes:0) << "% of ref nodes)\n";
        }
    }

    // LOCAL-ENVELOPE pinch: a reference node is a pinch if it is covered by as
    // many haplotypes as any node within a local window, i.e. all haplotypes
    // *present here* are on it (absent/gapped haplotypes don't block the cut).
    // This is the layout-relevant criterion; global-H convergence over-counts
    // systematically-missing haplotypes as if they diverted.
    auto local_pinch = [&](std::uint64_t W, double slack){
        std::uint64_t n = ref_cov.size();
        std::vector<std::int32_t> rollmax(n, 0);
        std::deque<std::uint64_t> dq;   // indices, decreasing cov
        // sliding window max over [i-W, i+W]
        std::uint64_t r = 0;
        for (std::uint64_t i = 0; i < n; ++i) {
            std::uint64_t hi = std::min(n, i + W + 1);
            while (r < hi) {
                while (!dq.empty() && ref_cov[dq.back()] <= ref_cov[r]) dq.pop_back();
                dq.push_back(r); ++r;
            }
            std::uint64_t lo = (i > W) ? i - W : 0;
            while (!dq.empty() && dq.front() < lo) dq.pop_front();
            rollmax[i] = ref_cov[dq.front()];
        }
        std::uint64_t np = 0; bool have = false; std::uint64_t last = 0;
        std::vector<std::uint64_t> g;
        for (std::uint64_t i = 0; i < n; ++i) {
            if (ref_cov[i] >= (std::int32_t)std::ceil(slack * rollmax[i]) && rollmax[i] > 0) {
                np++;
                if (have) g.push_back(ref_bp[i] - last);
                last = ref_bp[i]; have = true;
            }
        }
        std::sort(g.begin(), g.end());
        auto q = [&](double p){ return g.empty()?0:g[std::min(g.size()-1,(std::size_t)(p*g.size()))]; };
        std::uint64_t h1=0,h2=0,h3=0,h4=0,h5=0; for (auto x:g){ if(x<=100)h1++; else if(x<=1000)h2++; else if(x<=10000)h3++; else if(x<=100000)h4++; else h5++; }
        std::cout << "  window +/-" << W << " nodes, slack " << slack << ":  pinches " << np
                  << " (" << (ref_nodes?100.0*np/ref_nodes:0) << "% of ref nodes)\n";
        std::cout << "    gap bp:  median " << q(0.5) << "  p90 " << q(0.9) << "  p99 " << q(0.99) << "  max " << (g.empty()?0:g.back()) << "\n";
        std::cout << "    gaps by bp:  <=100:" << h1 << "  101-1k:" << h2 << "  1k-10k:" << h3 << "  10k-100k:" << h4 << "  >100k:" << h5 << "\n";
    };
    std::cout << "=== LOCAL-envelope pinch (all *present* haplotypes converge) ===\n";
    local_pinch(50, 1.0);
    local_pinch(50, 0.98);
    local_pinch(200, 1.0);

    double refbp = (double)bp;
    std::cout << "=== path-pinch (all-haplotype convergence) structure ===\n";
    std::cout << "reference nodes       " << ref_nodes << "\n";
    std::cout << "reference length      " << bp << " bp\n";
    std::cout << "pinches               " << pinches
              << "   (" << (ref_nodes ? 100.0*pinches/(double)ref_nodes : 0) << "% of ref nodes)\n";
    std::cout << "mean pinch spacing    " << (pinches ? (std::uint64_t)(refbp/pinches) : 0) << " bp\n";
    std::cout << "compartment size (bp gap between consecutive pinches):\n";
    report(gaps, "bp");
    std::cout << "compartment size (reference-node count between pinches):\n";
    report(gap_nodes, "nodes");
    return 0;
}
