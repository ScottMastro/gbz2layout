// compartments — probe the block-cut structure of a chromosome graph.
//
// Tests the "compartments along X" hypothesis: cut the graph only at nodes
// that are articulation points (removing one disconnects upstream from
// downstream — a pinch every path must traverse). The pieces between pinches
// are biconnected blocks that can each be laid out independently in parallel
// and stitched at the shared pinch nodes.
//
// We report the block SIZE DISTRIBUTION: if the graph shatters into many small
// blocks the approach parallelizes well; a few giant blocks are the entangled
// regions that would still need a joint solve.
//
// Method: iterative Tarjan for articulation points, then union-find over the
// non-articulation nodes to recover each block's interior + size.
//
// usage: compartments <graph.gbz> [--chromosome NAME]

#include "xp.hpp"

#include <gbwtgraph/gbz.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

using namespace gbz2layout;

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: compartments <graph.gbz> [--chromosome NAME]\n"; return 1; }
    std::string gbz_path = argv[1], chromosome;
    for (int i = 2; i < argc; ++i) { std::string a = argv[i]; if (a == "--chromosome" && i + 1 < argc) chromosome = argv[++i]; }

    gbwtgraph::GBZ gbz;
    { std::ifstream in(gbz_path, std::ios::binary); if (!in) { std::cerr << "cannot open\n"; return 1; } gbz.simple_sds_load(in); }
    const gbwtgraph::GBWTGraph& graph = gbz.graph;

    // dense index over all nodes (optionally restricted to a chromosome component)
    std::int64_t minid = graph.min_node_id(), maxid = graph.max_node_id();
    std::uint64_t span = std::uint64_t(maxid - minid + 1);
    std::vector<std::int32_t> id2idx(span, -1);
    std::vector<std::int64_t> idx2id;

    if (!chromosome.empty()) {
        // BFS the reference contig's component (as in gbz2layout --chromosome)
        std::vector<bool> seen(span, false);
        std::queue<std::int64_t> bfs; bool found = false;
        graph.for_each_path_handle([&](const handlegraph::path_handle_t& p) {
            if (found) return;
            std::string nm = graph.get_path_name(p);
            if (nm.substr(nm.find_last_of('#') + 1) != chromosome) return;
            found = true;
            graph.for_each_step_in_path(p, [&](const handlegraph::step_handle_t& s) {
                std::int64_t id = graph.get_id(graph.get_handle_of_step(s));
                if (!seen[id - minid]) { seen[id - minid] = true; bfs.push(id); }
            });
        });
        if (!found) { std::cerr << "no reference path for '" << chromosome << "'\n"; return 1; }
        while (!bfs.empty()) {
            std::int64_t id = bfs.front(); bfs.pop();
            handle_t h = graph.get_handle(id, false);
            for (bool go_left : {false, true})
                graph.follow_edges(h, go_left, [&](const handle_t& nb) {
                    std::int64_t nid = graph.get_id(nb);
                    if (!seen[nid - minid]) { seen[nid - minid] = true; bfs.push(nid); }
                    return true;
                });
        }
        for (std::uint64_t k = 0; k < span; ++k) if (seen[k]) { id2idx[k] = (std::int32_t)idx2id.size(); idx2id.push_back(minid + k); }
    } else {
        graph.for_each_handle([&](const handle_t& h) {
            std::int64_t id = graph.get_id(h);
            id2idx[id - minid] = (std::int32_t)idx2id.size(); idx2id.push_back(id);
        });
    }
    const std::uint64_t N = idx2id.size();

    // build undirected simple-graph CSR over dense indices
    std::vector<std::pair<std::int32_t, std::int32_t>> ep;
    for (std::uint64_t u = 0; u < N; ++u) {
        handle_t h = graph.get_handle(idx2id[u], false);
        for (bool go_left : {false, true})
            graph.follow_edges(h, go_left, [&](const handle_t& nb) {
                std::int32_t v = id2idx[graph.get_id(nb) - minid];
                if (v >= 0 && (std::uint64_t)v != u) ep.emplace_back(std::min((std::int32_t)u, v), std::max((std::int32_t)u, v));
                return true;
            });
    }
    std::sort(ep.begin(), ep.end());
    ep.erase(std::unique(ep.begin(), ep.end()), ep.end());
    std::vector<std::uint64_t> off(N + 1, 0);
    for (auto& e : ep) { off[e.first + 1]++; off[e.second + 1]++; }
    for (std::uint64_t i = 0; i < N; ++i) off[i + 1] += off[i];
    std::vector<std::int32_t> adj(off[N]);
    { std::vector<std::uint64_t> cur(off.begin(), off.end() - 1);
      for (auto& e : ep) { adj[cur[e.first]++] = e.second; adj[cur[e.second]++] = e.first; } }
    std::cerr << "[compartments] " << N << " nodes, " << ep.size() << " undirected edges\n";

    // iterative Tarjan: articulation points
    std::vector<std::int32_t> disc(N, 0), low(N, 0);
    std::vector<char> is_art(N, 0);
    std::int32_t timer = 1;
    struct Frame { std::int32_t u, parent; std::uint64_t ci; std::int32_t children; };
    for (std::uint64_t s = 0; s < N; ++s) {
        if (disc[s]) continue;
        std::vector<Frame> st;
        st.push_back({(std::int32_t)s, -1, off[s], 0});
        disc[s] = low[s] = timer++;
        while (!st.empty()) {
            Frame& f = st.back();
            if (f.ci < off[f.u + 1]) {
                std::int32_t v = adj[f.ci++];
                if (v == f.parent) continue;
                if (!disc[v]) {
                    disc[v] = low[v] = timer++;
                    f.children++;
                    st.push_back({v, f.u, off[v], 0});
                } else {
                    low[f.u] = std::min(low[f.u], disc[v]);
                }
            } else {
                st.pop_back();
                if (!st.empty()) {
                    Frame& p = st.back();
                    low[p.u] = std::min(low[p.u], low[f.u]);
                    if (p.parent != -1 && low[f.u] >= disc[p.u]) is_art[p.u] = 1;   // non-root cut vertex
                }
                if (f.parent == -1 && f.children > 1) is_art[f.u] = 1;              // root cut vertex
            }
        }
    }
    std::uint64_t nart = 0; for (std::uint64_t i = 0; i < N; ++i) nart += is_art[i];

    // union-find over non-articulation nodes -> block interiors
    std::vector<std::uint64_t> uf(N);
    for (std::uint64_t i = 0; i < N; ++i) uf[i] = i;
    std::function<std::uint64_t(std::uint64_t)> find = [&](std::uint64_t x){ while (uf[x]!=x){ uf[x]=uf[uf[x]]; x=uf[x]; } return x; };
    for (auto& e : ep) if (!is_art[e.first] && !is_art[e.second]) uf[find(e.first)] = find(e.second);
    // block size = interior nodes + distinct articulation neighbours
    std::vector<std::uint64_t> interior(N, 0);
    for (std::uint64_t i = 0; i < N; ++i) if (!is_art[i]) interior[find(i)]++;
    std::vector<std::uint64_t> sizes;
    for (std::uint64_t i = 0; i < N; ++i) if (!is_art[i] && find(i) == i) sizes.push_back(interior[i]);
    // articulation points that neighbour only other articulation points form
    // trivial 2-cliques (backbone links) — count them as size-0 interior blocks
    std::uint64_t backbone_links = 0;
    for (auto& e : ep) if (is_art[e.first] && is_art[e.second]) backbone_links++;

    std::sort(sizes.rbegin(), sizes.rend());
    auto pct = [&](double p){ return sizes.empty() ? 0 : sizes[std::min(sizes.size()-1, (std::size_t)(p*sizes.size()))]; };
    std::uint64_t total_interior = 0, mx = sizes.empty() ? 0 : sizes.front();
    for (auto v : sizes) total_interior += v;

    // histogram
    std::uint64_t h1=0,h2=0,h3=0,h4=0,h5=0,h6=0;
    for (auto v : sizes) { if(v<=2)h1++; else if(v<=5)h2++; else if(v<=20)h3++; else if(v<=100)h4++; else if(v<=1000)h5++; else h6++; }

    std::cout << "=== compartment (biconnected-block) structure ===\n";
    std::cout << "nodes                 " << N << "\n";
    std::cout << "articulation points   " << nart << "   (pinch/glue nodes)\n";
    std::cout << "non-trivial blocks    " << sizes.size() << "   (bubbles/tangles)\n";
    std::cout << "backbone links        " << backbone_links << "   (trivial 2-node blocks)\n";
    std::cout << "interior nodes total  " << total_interior << "\n";
    std::cout << "largest block (nodes) " << mx << "   (" << (N ? 100.0*mx/(double)N : 0) << "% of all nodes)\n";
    std::cout << "median / p90 / p99    " << pct(0.5) << " / " << pct(0.1) << " / " << pct(0.01) << "\n";
    std::cout << "block-size histogram (interior node count):\n";
    std::cout << "  <=2    " << h1 << "\n  3-5    " << h2 << "\n  6-20   " << h3
              << "\n  21-100 " << h4 << "\n  101-1k " << h5 << "\n  >1000  " << h6 << "\n";
    std::cout << "top 10 largest blocks:";
    for (std::size_t i = 0; i < std::min<std::size_t>(10, sizes.size()); ++i) std::cout << ' ' << sizes[i];
    std::cout << "\n";
    return 0;
}
