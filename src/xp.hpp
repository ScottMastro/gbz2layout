// xp.hpp — GBWT-backed, lean, coverage-capped path index for PG-SGD layout.
//
// This is the "core work" of Route B: it presents the same query API that
// odgi's path_linear_sgd_layout consumes, but is built by iterating haplotype
// paths straight out of a GBWT (not an odgi graph_t), stores node ids as a
// dense 0..N-1 rank (ids are sparse in per-chromosome GBZs), and supports
// per-node coverage capping (keep <= C occurrences of each node) to bound
// memory and SGD work at chr1 scale.
//
// Groupings (see README M1 findings):
//   node-grouped (one block per node, used for uniform step sampling):
//     np_bv (block starts), nr_iv (retained step-rank within its path),
//     npi_iv (path id), + np_bv_select
//   path-grouped (concatenated over paths, used for step->node/pos lookup):
//     handles_iv ((rank<<1)|orient), positions_iv (true bp offset in path)
//
// "Lean": we keep positions (O(1) get_position_of_step) but drop odgi's
// per-path offsets bitvector; get_step_at_position uses binary search instead.

#pragma once

#include <gbwtgraph/gbz.h>
#include <gbwtgraph/gbwtgraph.h>
#include <gbwt/gbwt.h>

#include <sdsl/int_vector.hpp>
#include <sdsl/bit_vectors.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gbz2layout {

using handlegraph::handle_t;
using handlegraph::step_handle_t;
using handlegraph::path_handle_t;

class XP {
public:
    // Build from a loaded GBZ. cap = max occurrences indexed per node
    // (0 = uncapped). Iterates forward GBWT sequences (all haplotype paths).
    //
    // node_mask (optional): restrict to one chromosome. Indexed by
    // (node_id - genome_min_id); only masked nodes and the sequences that live
    // in them are indexed (dense ranks are assigned within the masked set).
    // Used to extract & lay out a single chromosome from a whole-genome GBZ.
    void build(const gbwtgraph::GBZ& gbz, std::uint64_t cap, bool verbose = true,
               const std::vector<bool>* node_mask = nullptr,
               std::int64_t genome_min_id = 0);

    // ---- query API consumed by path_linear_sgd_layout ----
    std::uint64_t path_count() const { return path_start_.size() - 1; }
    std::uint64_t node_count() const { return node_count_; }
    std::uint64_t max_path_step_count() const { return max_path_steps_; }

    // node-grouped arrays for uniform step sampling
    const sdsl::bit_vector&   get_np_bv()  const { return np_bv_; }
    const sdsl::int_vector<>& get_nr_iv()  const { return nr_iv_; }
    const sdsl::int_vector<>& get_npi_iv() const { return npi_iv_; }
    const sdsl::bit_vector::select_1_type& get_np_bv_select() const { return np_bv_select_; }

    std::uint64_t get_path_step_count(const path_handle_t& p) const;
    std::uint64_t get_path_length(const path_handle_t& p) const;    // true full bp length
    std::string   get_path_name(const path_handle_t& p) const;

    // step is encoded (as odgi does) as as_integers(step)[0]=path_id, [1]=retained rank
    handle_t      get_handle_of_step(const step_handle_t& s) const;
    std::uint64_t get_position_of_step(const step_handle_t& s) const; // true bp offset in path
    step_handle_t get_step_at_position(const path_handle_t& p, std::uint64_t pos) const;

    // dense 0..N-1 rank for X/Y indexing (replaces odgi's unpack_number(handle))
    std::uint64_t rank_of_handle(const handle_t& h) const;
    std::uint64_t rank_of_id(std::uint64_t node_id) const { return id2rank_[node_id - min_id_] - 1; }
    std::uint64_t node_id_of_rank(std::uint64_t rank) const { return node_id_by_rank_[rank]; }
    std::uint64_t node_length_of_rank(std::uint64_t rank) const { return node_len_[rank]; }

    // reference-path access for reference-anchored init (the single named path)
    // returns false if the GBZ exposes no named/reference path
    bool has_reference() const { return has_ref_; }
    // for each reference step: (node_rank, bp_position). ref_len_bp filled.
    const std::vector<std::uint32_t>& ref_ranks() const { return ref_ranks_; }
    const std::vector<std::uint64_t>& ref_positions() const { return ref_positions_; }

    std::uint64_t total_steps() const { return total_steps_; }

private:
    // node remap
    std::uint64_t node_count_ = 0;
    std::int64_t  min_id_ = 0, max_id_ = 0;
    std::vector<std::uint32_t> id2rank_;        // [id-min] -> rank+1 (0 = absent)
    std::vector<std::uint64_t> node_id_by_rank_;// rank -> node id
    std::vector<std::uint32_t> node_len_;       // rank -> bp length

    // path-grouped
    std::vector<std::uint64_t> path_start_;     // size n_paths+1, offset into step arrays
    std::vector<std::uint64_t> path_bp_len_;    // true full bp length per path
    std::vector<std::uint32_t> path_seq_id_;    // gbwt forward sequence id per path (for name)
    sdsl::int_vector<> handles_iv_;             // (rank<<1)|orient per retained step
    sdsl::int_vector<> positions_iv_;           // true bp offset per retained step

    // node-grouped
    sdsl::bit_vector   np_bv_;
    sdsl::int_vector<> nr_iv_;
    sdsl::int_vector<> npi_iv_;
    sdsl::bit_vector::select_1_type np_bv_select_;

    // reference path (named path 0, if any)
    bool has_ref_ = false;
    std::vector<std::uint32_t> ref_ranks_;
    std::vector<std::uint64_t> ref_positions_;

    std::uint64_t total_steps_ = 0;
    std::uint64_t max_path_steps_ = 0;
    const gbwt::Metadata* meta_ = nullptr;
};

} // namespace gbz2layout
