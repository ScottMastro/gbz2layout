// export_gbz.hpp — write a standalone single-chromosome GBZ extracted from a
// whole-genome GBZ.
//
// The whole-genome GBZ carries every chromosome (disjoint components) and its
// full GBWT, so laying out one chromosome from it keeps the entire ~5 GB genome
// resident. This export builds a compact GBZ containing only the masked
// chromosome: nodes renumbered 1..N (via the XP's dense ranks), a fresh GBWT
// holding just that chromosome's haplotype threads (renumbered), and the
// reference-sample tag + per-path metadata copied so the reference path stays a
// named path (reference-anchored init depends on it). The result loads at
// ~chromosome scale, which is what lets chr1 layout fit a laptop.

#pragma once

#include "xp.hpp"

#include <gbwtgraph/gbz.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gbz2layout {

// Serialize a single-chromosome GBZ to out_path. `mask` (indexed by
// node_id - genome_min_id) selects the chromosome's component; `xp` must have
// been built over that same mask (supplies the 1..N node renumbering).
// Returns true on success.
bool export_chromosome_gbz(const gbwtgraph::GBZ& gbz,
                           const XP& xp,
                           const std::vector<bool>& mask,
                           std::int64_t genome_min_id,
                           const std::string& chromosome,
                           const std::string& out_path,
                           bool verbose = true);

} // namespace gbz2layout
