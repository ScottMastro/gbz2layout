// xp.cpp — implementation of the GBWT-backed lean capped path index.

#include "xp.hpp"

#include <sdsl/util.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace gbz2layout {

using handlegraph::as_integer;
using handlegraph::as_integers;
using handlegraph::as_path_handle;

static inline std::uint8_t wbits(std::uint64_t x) {
    return x ? (std::uint8_t)(sdsl::bits::hi(x) + 1) : 1;
}

void XP::build(const gbwtgraph::GBZ& gbz, std::uint64_t cap, bool verbose,
               const std::vector<bool>* node_mask, std::int64_t genome_min_id) {
    const gbwtgraph::GBWTGraph& graph = gbz.graph;
    const gbwt::GBWT& index = gbz.index;
    meta_ = index.hasMetadata() ? &index.metadata : nullptr;
    const bool filtered = (node_mask != nullptr);

    auto masked = [&](std::uint64_t id) -> bool {
        return !filtered || (*node_mask)[id - (std::uint64_t)genome_min_id];
    };

    // (0) determine the id range + node count of the (masked) node set
    if (filtered) {
        std::int64_t cmin = 0, cmax = 0; std::uint64_t cnt = 0;
        graph.for_each_handle([&](const handle_t& h) {
            std::uint64_t id = graph.get_id(h);
            if (!masked(id)) return;
            if (cnt == 0) { cmin = cmax = (std::int64_t)id; }
            else { cmin = std::min<std::int64_t>(cmin, id); cmax = std::max<std::int64_t>(cmax, id); }
            ++cnt;
        });
        node_count_ = cnt; min_id_ = cmin; max_id_ = cmax;
    } else {
        node_count_ = graph.get_node_count();
        min_id_ = graph.min_node_id();
        max_id_ = graph.max_node_id();
    }
    const std::uint64_t id_span = std::uint64_t(max_id_ - min_id_ + 1);

    // (1) id -> dense rank remap + node lengths + rank -> id (masked nodes only)
    id2rank_.assign(id_span, 0);
    node_len_.assign(node_count_, 0);
    node_id_by_rank_.assign(node_count_, 0);
    std::uint32_t next_rank = 0;
    graph.for_each_handle([&](const handle_t& h) {
        std::uint64_t id = graph.get_id(h);
        if (!masked(id)) return;
        id2rank_[id - min_id_] = ++next_rank;         // 1-based
        node_len_[next_rank - 1] = graph.get_length(h);
        node_id_by_rank_[next_rank - 1] = id;
    });

    auto rank_of_id_local = [&](std::uint64_t id) -> std::uint32_t {
        return id2rank_[id - min_id_] - 1;
    };

    // (2) first pass over forward sequences: keep only sequences in the masked
    // set (MC chromosomes are disjoint components, so testing the first node
    // suffices). Count full steps per node + path metadata.
    const std::uint64_t n_seq = index.sequences();
    std::vector<std::uint64_t> full_spn(node_count_, 0);
    std::vector<std::uint32_t> path_seq;               // forward seq id per non-empty path
    std::vector<std::uint64_t> path_steps_full;        // full step count per path
    std::vector<std::uint64_t> path_bp;                // full bp length per path
    for (std::uint64_t seq = 0; seq < n_seq; seq += 2) {
        if (filtered) {
            // cheap O(1) peek at the first node; skip other chromosomes without
            // the (expensive) full path extraction (pass-1 is otherwise ~93% waste)
            gbwt::edge_type e = index.start(seq);
            if (e.first == gbwt::ENDMARKER) continue;             // empty sequence
            if (!masked(gbwt::Node::id(e.first))) continue;       // not this chromosome
        }
        gbwt::vector_type path = index.extract(seq);
        if (path.empty()) continue;
        std::uint64_t bp = 0;
        for (gbwt::node_type node : path) {
            std::uint32_t r = rank_of_id_local(gbwt::Node::id(node));
            full_spn[r]++;
            bp += node_len_[r];
        }
        path_seq.push_back((std::uint32_t)seq);
        path_steps_full.push_back(path.size());
        path_bp.push_back(bp);
    }
    const std::uint64_t n_paths = path_seq.size();

    // capped steps per node + retained count per path (need a scatter cursor)
    std::uint64_t np = 0;
    for (std::uint64_t r = 0; r < node_count_; ++r)
        np += (cap == 0) ? full_spn[r] : std::min<std::uint64_t>(full_spn[r], cap);
    total_steps_ = np;

    // widths
    std::uint64_t max_steps_path = 0, max_bp_path = 0;
    for (std::uint64_t p = 0; p < n_paths; ++p) {
        max_steps_path = std::max(max_steps_path, path_steps_full[p]);
        max_bp_path = std::max(max_bp_path, path_bp[p]);
    }
    const std::uint8_t w_hand = wbits(((node_count_ ? node_count_ - 1 : 0) << 1) | 1);
    const std::uint8_t w_pos  = wbits(max_bp_path);
    const std::uint8_t w_npi  = wbits(n_paths ? n_paths - 1 : 0);
    const std::uint8_t w_nr   = wbits(max_steps_path);   // retained rank <= full steps

    // node-grouped block layout
    std::vector<std::uint64_t> node_start(node_count_ + 1, 0);
    for (std::uint64_t r = 0; r < node_count_; ++r) {
        std::uint64_t keep = (cap == 0) ? full_spn[r] : std::min<std::uint64_t>(full_spn[r], cap);
        node_start[r + 1] = node_start[r] + keep;
    }
    np_bv_ = sdsl::bit_vector(np, 0);
    for (std::uint64_t r = 0; r < node_count_; ++r) np_bv_[node_start[r]] = 1;
    nr_iv_  = sdsl::int_vector<>(np, 0, w_nr);
    npi_iv_ = sdsl::int_vector<>(np, 0, w_npi);

    // path-grouped arrays: sized by retained steps, path_start prefix
    path_start_.assign(n_paths + 1, 0);
    // we need retained-per-path; compute during the scatter with a kept[] counter
    handles_iv_   = sdsl::int_vector<>(np, 0, w_hand);
    positions_iv_ = sdsl::int_vector<>(np, 0, w_pos);
    path_bp_len_.assign(n_paths, 0);
    path_seq_id_.assign(n_paths, 0);

    // First compute retained steps per path (respecting the same node-cap decision),
    // so path_start_ is known before we place path-grouped entries.
    {
        std::vector<std::uint32_t> kept(node_count_, 0);
        std::vector<std::uint64_t> retained(n_paths, 0);
        for (std::uint64_t p = 0; p < n_paths; ++p) {
            gbwt::vector_type path = index.extract(path_seq[p]);
            std::uint64_t r_keep = 0;
            for (gbwt::node_type node : path) {
                std::uint32_t r = rank_of_id_local(gbwt::Node::id(node));
                if (cap != 0 && kept[r] >= cap) continue;
                ++kept[r]; ++r_keep;
            }
            retained[p] = r_keep;
        }
        for (std::uint64_t p = 0; p < n_paths; ++p) {
            path_start_[p + 1] = path_start_[p] + retained[p];
            max_path_steps_ = std::max(max_path_steps_, retained[p]);
        }
    }

    // (3) scatter pass: fill node-grouped (nr/npi) and path-grouped (handles/pos)
    std::vector<std::uint64_t> cursor(node_start.begin(), node_start.end()); // node block cursor
    std::vector<std::uint32_t> kept(node_count_, 0);
    for (std::uint64_t p = 0; p < n_paths; ++p) {
        gbwt::vector_type path = index.extract(path_seq[p]);
        path_seq_id_[p] = path_seq[p];
        path_bp_len_[p] = path_bp[p];
        std::uint64_t bp = 0;
        std::uint64_t retained_rank = 0;                    // index within retained path
        std::uint64_t pg = path_start_[p];                  // path-grouped write cursor
        for (gbwt::node_type node : path) {
            std::uint32_t r = rank_of_id_local(gbwt::Node::id(node));
            std::uint64_t here = bp; bp += node_len_[r];
            if (cap != 0 && kept[r] >= cap) continue;        // node cap reached -> skip
            ++kept[r];
            // node-grouped slot
            std::uint64_t idx = cursor[r]++;
            nr_iv_[idx]  = retained_rank + 1;                // 1-based retained rank (odgi convention)
            npi_iv_[idx] = p;
            // path-grouped slot
            bool orient = gbwt::Node::is_reverse(node);
            handles_iv_[pg]   = (std::uint64_t(r) << 1) | (orient ? 1u : 0u);
            positions_iv_[pg] = here;
            ++pg;
            ++retained_rank;
        }
    }

    sdsl::util::bit_compress(nr_iv_);
    sdsl::util::bit_compress(npi_iv_);
    sdsl::util::bit_compress(handles_iv_);
    sdsl::util::bit_compress(positions_iv_);
    sdsl::util::assign(np_bv_select_, sdsl::bit_vector::select_1_type(&np_bv_));

    // (4) reference path for init: the named/reference path that lives in the
    // masked node set (its own chromosome). Unfiltered: the first named path.
    has_ref_ = false;
    graph.for_each_path_handle([&](const path_handle_t& path) {
        if (has_ref_) return;
        // in filtered mode, skip named paths that aren't this chromosome
        handle_t first_h = graph.get_handle_of_step(graph.path_begin(path));
        if (filtered && !masked(graph.get_id(first_h))) return;
        has_ref_ = true;
        std::uint64_t bp = 0;
        graph.for_each_step_in_path(path, [&](const step_handle_t& step) {
            handle_t h = graph.get_handle_of_step(step);
            std::uint32_t r = rank_of_id_local(graph.get_id(h));
            ref_ranks_.push_back(r);
            ref_positions_.push_back(bp);
            bp += node_len_[r];
        });
    });

    if (verbose) {
        std::cerr << "[xp] nodes=" << node_count_ << " paths=" << n_paths
                  << " steps(kept)=" << np << " cap=" << cap
                  << " widths[h=" << (int)w_hand << " pos=" << (int)w_pos
                  << " npi=" << (int)w_npi << " nr=" << (int)w_nr << "]"
                  << " ref=" << (has_ref_ ? ref_ranks_.size() : 0) << " steps\n";
    }
}

// ---- queries ----

std::uint64_t XP::get_path_step_count(const path_handle_t& p) const {
    std::uint64_t pid = as_integer(p);
    return path_start_[pid + 1] - path_start_[pid];
}

std::uint64_t XP::get_path_length(const path_handle_t& p) const {
    return path_bp_len_[as_integer(p)];
}

std::string XP::get_path_name(const path_handle_t& p) const {
    std::uint64_t pid = as_integer(p);
    if (meta_ && meta_->hasPathNames()) {
        std::uint64_t seq = path_seq_id_[pid];
        gbwt::size_type path_idx = seq / 2;                  // forward seq -> path index
        if (path_idx < meta_->paths()) {
            gbwt::FullPathName fp = meta_->fullPath(path_idx);
            return fp.sample_name + "#" + fp.contig_name + "#" + std::to_string(fp.haplotype);
        }
    }
    return "path_" + std::to_string(pid);
}

handle_t XP::get_handle_of_step(const step_handle_t& s) const {
    std::uint64_t pid  = as_integers(s)[0];
    std::uint64_t rank = as_integers(s)[1];
    std::uint64_t packed = handles_iv_[path_start_[pid] + rank];
    std::uint64_t r = packed >> 1;
    bool orient = packed & 1u;
    gbwt::node_type node = gbwt::Node::encode(node_id_by_rank_[r], orient);
    return gbwtgraph::GBWTGraph::node_to_handle(node);
}

std::uint64_t XP::get_position_of_step(const step_handle_t& s) const {
    std::uint64_t pid  = as_integers(s)[0];
    std::uint64_t rank = as_integers(s)[1];
    return positions_iv_[path_start_[pid] + rank];
}

step_handle_t XP::get_step_at_position(const path_handle_t& p, std::uint64_t pos) const {
    std::uint64_t pid = as_integer(p);
    std::uint64_t lo = path_start_[pid], hi = path_start_[pid + 1];
    // largest retained step whose start position <= pos (binary search over positions_iv_)
    std::uint64_t n = hi - lo;
    std::uint64_t left = 0, right = n;                       // [left, right)
    while (left < right) {
        std::uint64_t mid = (left + right) / 2;
        if (positions_iv_[lo + mid] <= pos) left = mid + 1;
        else right = mid;
    }
    std::uint64_t rank = (left == 0) ? 0 : left - 1;         // clamp to first step
    step_handle_t s;
    as_integers(s)[0] = pid;
    as_integers(s)[1] = rank;
    return s;
}

std::uint64_t XP::rank_of_handle(const handle_t& h) const {
    gbwt::node_type node = gbwtgraph::GBWTGraph::handle_to_node(h);
    return rank_of_id(gbwt::Node::id(node));
}

} // namespace gbz2layout
