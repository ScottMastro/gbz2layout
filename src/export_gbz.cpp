#include "export_gbz.hpp"

#include <gbwtgraph/gbwtgraph.h>
#include <gbwtgraph/utils.h>

#include <gbwt/dynamic_gbwt.h>
#include <gbwt/gbwt.h>
#include <gbwt/metadata.h>
#include <gbwt/support.h>
#include <gbwt/utils.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace gbz2layout {

bool export_chromosome_gbz(const gbwtgraph::GBZ& gbz,
                           const XP& xp,
                           const std::vector<bool>& mask,
                           std::int64_t genome_min_id,
                           const std::string& chromosome,
                           const std::string& out_path,
                           bool verbose) {
    const gbwtgraph::GBWTGraph& graph = gbz.graph;
    const gbwt::GBWT&           index = gbz.index;
    const std::uint64_t N = xp.node_count();
    if (N == 0) { std::cerr << "[export] empty chromosome; nothing to write\n"; return false; }

    auto in_mask = [&](std::uint64_t old_id) -> bool {
        std::uint64_t off = old_id - (std::uint64_t)genome_min_id;
        return off < mask.size() && mask[off];
    };

    // ---- (1) sequences: new node id = rank+1, sequence copied from old graph ----
    gbwtgraph::SequenceSource src;
    for (std::uint64_t r = 0; r < N; ++r) {
        std::uint64_t old_id = xp.node_id_of_rank(r);
        handlegraph::handle_t h = graph.get_handle(old_id, false);
        src.add_node((gbwtgraph::nid_t)(r + 1), graph.get_sequence(h));
    }
    if (verbose) std::cerr << "[export] " << N << " nodes copied (renumbered 1.." << N << ")\n";

    // ---- (2) new GBWT over this chromosome's threads, renumbered ----
    gbwt::size_type node_width = gbwt::bit_length(gbwt::Node::encode(N, true));
    // NB: DynamicGBWT construction peaks at ~7 GB per ~2M-node chromosome
    // regardless of the insert batch size (measured — batch size is not the
    // lever). On top of the ~5.4 GB resident whole-genome GBZ that caps the
    // laptop near chr21/chr22 size; bigger chromosomes need a larger-RAM host.
    gbwt::GBWTBuilder builder(node_width);

    const gbwt::Metadata* om = index.hasMetadata() ? &index.metadata : nullptr;
    const bool have_meta = (om != nullptr) && om->hasPathNames();

    // compact metadata dictionaries (only names actually used by kept paths)
    std::vector<std::string> sample_names, contig_names;
    std::unordered_map<std::string, std::uint32_t> sidx, cidx;
    std::vector<gbwt::PathName> new_paths;
    auto intern = [](std::vector<std::string>& names,
                     std::unordered_map<std::string, std::uint32_t>& idx,
                     const std::string& nm) -> std::uint32_t {
        auto it = idx.find(nm);
        if (it != idx.end()) return it->second;
        std::uint32_t id = (std::uint32_t)names.size();
        names.push_back(nm); idx.emplace(nm, id);
        return id;
    };

    // NB: haplotype paths here are named sample#hap#<denovo-contig> (assembly
    // accessions, not "chrY"), so contig-name filtering is useless — the node
    // component mask is the only reliable membership test. Peek the first node
    // via start() (O(1)) to reject off-chromosome threads before full extract.
    std::uint64_t n_paths = index.sequences() / 2;   // bidirectional: forward = 2*i
    std::uint64_t kept = 0;
    for (std::uint64_t i = 0; i < n_paths; ++i) {
        gbwt::edge_type e0 = index.start(2 * i);
        if (e0.first == gbwt::ENDMARKER) continue;        // empty thread
        if (!in_mask(gbwt::Node::id(e0.first))) continue; // not this chromosome
        gbwt::vector_type fwd = index.extract(2 * i);
        if (fwd.empty()) continue;

        gbwt::vector_type out; out.reserve(fwd.size());
        for (gbwt::node_type x : fwd) {
            std::uint64_t nid = xp.rank_of_id(gbwt::Node::id(x)) + 1;
            out.push_back(gbwt::Node::encode(nid, gbwt::Node::is_reverse(x)));
        }
        builder.insert(out, true);   // bidirectional (GBWTGraph needs both orientations)

        if (have_meta) {
            const gbwt::PathName& pn = om->path(i);
            std::uint32_t s = intern(sample_names, sidx, om->sample(pn.sample));
            std::uint32_t c = intern(contig_names, cidx, om->contig(pn.contig));
            new_paths.emplace_back(s, c, pn.phase, pn.count);
        }
        ++kept;
    }
    builder.finish();
    if (verbose)
        std::cerr << "[export] " << kept << " threads kept of "
                  << n_paths << " genome paths\n";
    if (kept == 0) { std::cerr << "[export] no threads for '" << chromosome << "'\n"; return false; }

    gbwt::GBWT compressed(builder.index);

    // ---- (3) metadata + reference-sample tag (keeps the reference path named) ----
    if (have_meta) {
        gbwt::Metadata nm;
        nm.setSamples(sample_names);
        nm.setContigs(contig_names);
        nm.setHaplotypes(kept);
        for (const gbwt::PathName& p : new_paths) nm.addPath(p);
        compressed.addMetadata();
        compressed.metadata = nm;
    }
    const std::unordered_set<std::string>& refs = graph.reference_samples;
    if (!refs.empty())
        compressed.tags.set(gbwtgraph::REFERENCE_SAMPLE_LIST_GBWT_TAG,
                            gbwtgraph::compose_reference_samples_tag(refs));

    // ---- (4) assemble + serialize GBZ ----
    gbwtgraph::GBZ out(compressed, src);
    if (!refs.empty()) out.set_reference_samples(refs);

    std::ofstream os(out_path, std::ios::binary);
    if (!os) { std::cerr << "[export] cannot open " << out_path << " for writing\n"; return false; }
    out.simple_sds_serialize(os);
    os.close();
    if (verbose)
        std::cerr << "[export] wrote " << out_path << " ("
                  << out.graph.get_node_count() << " nodes, "
                  << (refs.empty() ? 0 : out.named_paths()) << " named paths)\n";
    return true;
}

} // namespace gbz2layout
