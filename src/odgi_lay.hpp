// odgi_lay.hpp — odgi's binary .lay format, vendored so our output is
// byte-compatible with `odgi draw` and anything else that reads odgi layouts.
//
// Copied from odgi src/algorithms/layout.{hpp,cpp} (MIT, Erik Garrison et al.).
// The constructor, serialize(), load() and the coordinate accessors are taken
// verbatim so both directions are odgi's own code. odgi's handle-based coords()
// and to_tsv() are omitted -- we index by rank directly and emit our own TSV.
//
// The format, for the record:
//   - min_value = the minimum over ALL X and Y (one scalar, shifts everything >= 0)
//   - each coordinate is stored as the RAW BIT PATTERN of (value - min_value)
//     reinterpreted as uint64, X and Y interleaved: x0,y0,x1,y1,...
//   - that vector goes into an sdsl::enc_vector<> (self-delimiting codes)
//   - serialize writes min_value then the enc_vector
//
// Indexing matches ours exactly: odgi uses 2*unpack_number(handle) +
// unpack_bit(handle), i.e. 2*node_rank + endpoint -- the same layout our X/Y
// vectors already use, so no remapping is needed.

#pragma once

#include <sdsl/enc_vector.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace gbz2layout {

union conv_t { std::uint64_t i; double d; };

class OdgiLayout {
    sdsl::enc_vector<> xy;
    double min_value = std::numeric_limits<double>::max();
public:
    OdgiLayout() { }
    OdgiLayout(const std::vector<double>& X, const std::vector<double>& Y);
    void serialize(std::ostream& out);
    void load(std::istream& in);
    std::size_t size() const;          // entries, i.e. 2 * node_count
    double get_x(std::uint64_t i) const;
    double get_y(std::uint64_t i) const;
};

} // namespace gbz2layout
