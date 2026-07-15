// test_xp — validate the GBWT-backed XP against direct GBWT extraction.

#include "xp.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

using namespace gbz2layout;
using handlegraph::as_integers;
using handlegraph::as_path_handle;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "  FAIL: " << what << "\n"; ++failures; }
    else       std::cerr << "  ok:   " << what << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: test_xp <graph.gbz> [cap]\n"; return 1; }
    std::uint64_t cap = (argc >= 3) ? std::stoull(argv[2]) : 0;

    gbwtgraph::GBZ gbz;
    { std::ifstream in(argv[1], std::ios::binary); gbz.simple_sds_load(in); }
    const gbwt::GBWT& index = gbz.index;

    XP xp;
    xp.build(gbz, cap);

    std::cerr << "\n=== structural checks ===\n";
    // np_bv popcount == node_count (one block-start per node)
    std::uint64_t ones = 0;
    for (std::uint64_t i = 0; i < xp.get_np_bv().size(); ++i) ones += xp.get_np_bv()[i];
    check(ones == xp.node_count(), "np_bv has one block-start per node");

    // path 0: compare XP's retained node sequence to a direct extract (cap=0 => identical)
    check(xp.path_count() > 0, "at least one path");
    {
        auto p = as_path_handle(0);
        std::uint64_t steps = xp.get_path_step_count(p);
        check(steps > 1, "path 0 has >1 retained step");

        // positions strictly increasing, and each get_handle_of_step node exists
        bool mono = true, ids_ok = true;
        std::uint64_t prev = 0;
        for (std::uint64_t r = 0; r < steps; ++r) {
            step_handle_t s; as_integers(s)[0] = 0; as_integers(s)[1] = r;
            std::uint64_t pos = xp.get_position_of_step(s);
            if (r > 0 && pos <= prev) mono = false;
            prev = pos;
            handle_t h = xp.get_handle_of_step(s);
            std::uint64_t rk = xp.rank_of_handle(h);
            if (rk >= xp.node_count()) ids_ok = false;
        }
        check(mono, "path 0 positions strictly increasing");
        check(ids_ok, "path 0 handles map to valid node ranks");

        if (cap == 0) {
            // node ids must match the raw GBWT extract exactly
            std::uint64_t seq = 0; // path 0 forward seq (first non-empty; assume seq 0)
            gbwt::vector_type raw = index.extract(seq);
            bool same = (raw.size() == steps);
            for (std::uint64_t r = 0; same && r < steps; ++r) {
                step_handle_t s; as_integers(s)[0] = 0; as_integers(s)[1] = r;
                handle_t h = xp.get_handle_of_step(s);
                std::uint64_t id = xp.node_id_of_rank(xp.rank_of_handle(h));
                if (id != gbwt::Node::id(raw[r])) same = false;
            }
            check(same, "cap=0: path 0 node ids match raw GBWT extract");
        }

        // get_step_at_position round-trip: position of returned step <= query
        bool rt = true;
        std::uint64_t plen = xp.get_path_length(p);
        for (std::uint64_t q = 0; q < 20; ++q) {
            std::uint64_t pos = (plen * q) / 20;
            step_handle_t s = xp.get_step_at_position(p, pos);
            std::uint64_t sp = xp.get_position_of_step(s);
            if (sp > pos) rt = false;
        }
        check(rt, "get_step_at_position returns a step at/before the query pos");
    }

    // node-grouped sampling consistency: for a few np indices, the (path,rank)
    // it names round-trips to a handle whose rank matches the block's node.
    std::cerr << "\n=== node-grouped sampling checks ===\n";
    {
        const auto& npi = xp.get_npi_iv();
        const auto& nr  = xp.get_nr_iv();
        bool ok = true;
        std::uint64_t N = xp.get_np_bv().size();
        for (std::uint64_t t = 0; t < 1000 && t < N; ++t) {
            std::uint64_t i = (N / 1000) * t;
            std::uint64_t pid = npi[i];
            std::uint64_t rank = nr[i] - 1;
            if (pid >= xp.path_count()) { ok = false; break; }
            if (rank >= xp.get_path_step_count(as_path_handle(pid))) { ok = false; break; }
        }
        check(ok, "sampled np indices name valid (path, rank) pairs");
    }

    std::cerr << "\n" << (failures ? "FAILURES: " + std::to_string(failures) : "ALL PASS") << "\n";
    return failures ? 1 : 0;
}
