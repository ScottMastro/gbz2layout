// test_components — weak-component discovery and odgi's component packing.
//
// Builds synthetic multi-component GBZs in memory, because nothing we produce
// naturally has more than one component: per-chromosome GBZs are extracted by
// BFS from a reference path, so they are single-component by construction. That
// left the packing path (bbox -> vertical stack -> offsets) written but never
// executed on real data.
//
// The all-negative case is deliberate. odgi's coord_range_2d_t initialises
// max_x/max_y to std::numeric_limits<double>::min() -- the smallest POSITIVE
// double (~2.2e-308), not the most negative -- so include()'s `if (x > max_x)`
// never fires when every coordinate is negative, and the bbox comes out wrong.
// We use lowest() instead; this pins that difference rather than leaving it as a
// claim in a comment. chr1's real layout has min_x = -17,696,844, so it matters.

#include "components.hpp"
#include "xp.hpp"

#include <gbwtgraph/gbz.h>
#include <gbwtgraph/gbwtgraph.h>
#include <gbwt/dynamic_gbwt.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace gbz2layout;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "  FAIL: " << what << "\n"; ++failures; }
    else      std::cerr << "  ok:   " << what << "\n";
}

// Build a GBZ of `ncomp` disjoint chains, each `chain_len` nodes long, with one
// path per chain. Chains never touch, so each is its own weak component.
static void build_chains(gbwtgraph::GBZ& gbz, std::uint64_t ncomp, std::uint64_t chain_len) {
    gbwtgraph::SequenceSource src;
    std::uint64_t id = 1;
    std::vector<std::vector<std::uint64_t>> chains;
    for (std::uint64_t c = 0; c < ncomp; ++c) {
        chains.emplace_back();
        for (std::uint64_t i = 0; i < chain_len; ++i) {
            src.add_node((gbwtgraph::nid_t)id, "ACGT");
            chains.back().push_back(id);
            ++id;
        }
    }

    gbwt::size_type node_width = gbwt::bit_length(gbwt::Node::encode(id, true));
    gbwt::GBWTBuilder builder(node_width);
    for (auto& chain : chains) {
        gbwt::vector_type path;
        for (std::uint64_t nid : chain) path.push_back(gbwt::Node::encode(nid, false));
        builder.insert(path, true);
    }
    builder.finish();
    gbwt::GBWT compressed(builder.index);
    gbz = gbwtgraph::GBZ(compressed, src);
}

int main() {
    std::cerr << "== single component ==\n";
    {
        gbwtgraph::GBZ gbz;
        build_chains(gbz, 1, 5);
        XP xp; xp.build(gbz, 1, true, nullptr, 1);
        std::vector<std::int32_t> comp;
        std::uint64_t n = weakly_connected_components(gbz.graph, xp, comp);
        check(n == 1, "one chain -> 1 component");
        bool all_zero = true;
        for (auto c : comp) if (c != 0) all_zero = false;
        check(all_zero, "every node labelled component 0");
        check(comp.size() == xp.node_count(), "one label per node");
    }

    std::cerr << "== three components ==\n";
    {
        gbwtgraph::GBZ gbz;
        build_chains(gbz, 3, 4);
        XP xp; xp.build(gbz, 1, true, nullptr, 1);
        std::vector<std::int32_t> comp;
        std::uint64_t n = weakly_connected_components(gbz.graph, xp, comp);
        check(n == 3, "three disjoint chains -> 3 components");

        std::vector<int> sizes(3, 0);
        bool labelled = true;
        for (auto c : comp) { if (c < 0 || c > 2) labelled = false; else sizes[c]++; }
        check(labelled, "every node got a valid label");
        check(sizes[0] == 4 && sizes[1] == 4 && sizes[2] == 4,
              "each component has 4 nodes");
    }

    std::cerr << "== packing separates components ==\n";
    {
        gbwtgraph::GBZ gbz;
        build_chains(gbz, 3, 4);
        XP xp; xp.build(gbz, 1, true, nullptr, 1);
        std::vector<std::int32_t> comp;
        std::uint64_t n = weakly_connected_components(gbz.graph, xp, comp);
        const std::uint64_t N = xp.node_count();

        // Put every component on top of every other: identical coords. This is
        // what an un-packed layout looks like, and what odgi's offsets fix.
        std::vector<std::atomic<double>> X(2 * N), Y(2 * N);
        for (std::uint64_t r = 0; r < N; ++r) {
            X[2 * r].store(0.0);     Y[2 * r].store(0.0);
            X[2 * r + 1].store(10.0); Y[2 * r + 1].store(0.0);
        }

        pack_components(n, comp, X, Y);

        // After packing, each component's y-range must be disjoint from the others.
        std::vector<double> ymin(n, 1e300), ymax(n, -1e300);
        for (std::uint64_t r = 0; r < N; ++r) {
            int c = comp[r];
            for (std::uint64_t j = 2 * r; j <= 2 * r + 1; ++j) {
                ymin[c] = std::min(ymin[c], Y[j].load());
                ymax[c] = std::max(ymax[c], Y[j].load());
            }
        }
        bool disjoint = true;
        for (std::uint64_t a = 0; a + 1 < n; ++a)
            if (ymax[a] >= ymin[a + 1]) disjoint = false;
        check(disjoint, "component y-ranges do not overlap after packing");
        check(std::fabs(ymin[0] - 1000.0) < 1e-6, "first component starts at the 1000 border");
        check(std::fabs(ymin[1] - ymin[0] - 1000.0) < 1e-6,
              "components are separated by the border");
    }

    std::cerr << "== all-negative coords (odgi's numeric_limits::min() bug) ==\n";
    {
        gbwtgraph::GBZ gbz;
        build_chains(gbz, 2, 3);
        XP xp; xp.build(gbz, 1, true, nullptr, 1);
        std::vector<std::int32_t> comp;
        std::uint64_t n = weakly_connected_components(gbz.graph, xp, comp);
        const std::uint64_t N = xp.node_count();

        // Every coordinate negative -- as chr1's real layout is (min_x = -17.7M).
        // With odgi's max_x = numeric_limits<double>::min() (~2.2e-308), max_x
        // would stay positive, height() would be garbage, and the components
        // would be stacked with a nonsense offset.
        std::vector<std::atomic<double>> X(2 * N), Y(2 * N);
        for (std::uint64_t r = 0; r < N; ++r) {
            X[2 * r].store(-500.0);     Y[2 * r].store(-300.0);
            X[2 * r + 1].store(-490.0); Y[2 * r + 1].store(-290.0);
        }

        pack_components(n, comp, X, Y);

        std::vector<double> ymin(n, 1e300), ymax(n, -1e300);
        for (std::uint64_t r = 0; r < N; ++r) {
            int c = comp[r];
            for (std::uint64_t j = 2 * r; j <= 2 * r + 1; ++j) {
                ymin[c] = std::min(ymin[c], Y[j].load());
                ymax[c] = std::max(ymax[c], Y[j].load());
            }
        }
        // Packing shifts, it never rescales, so each component still spans the
        // 10 units it started with. NB this assertion does NOT catch the bbox
        // bug -- it passes either way. Kept only as a sanity check that packing
        // is a pure translation.
        check(std::fabs((ymax[0] - ymin[0]) - 10.0) < 1e-6,
              "packing is a pure translation (span still 10)");
        check(ymin[0] > 0.0, "negative input lifted above the origin by the border");

        // THIS is the assertion with teeth. A wrong bbox does not show up in the
        // component's own coordinates -- it shows up in the NEXT component's
        // offset, via `curr_y_offset += cr.height() + border`. With odgi's
        // max_y = numeric_limits<double>::min() (~2.2e-308) and coords at
        // -300..-290, height() computes as ~300 instead of 10, so component 1
        // lands ~290 units too high. Verified to fail when the bug is restored.
        check(std::fabs(ymin[1] - (ymax[0] + 1000.0)) < 1e-6,
              "next component offset uses a correct bbox height (catches the min()/lowest() bug)");
    }

    std::cerr << (failures ? "FAILED\n" : "all component tests passed\n");
    return failures ? 1 : 0;
}
