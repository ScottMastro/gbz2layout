// Vendored from odgi src/algorithms/layout.cpp (MIT). See odgi_lay.hpp.
// Kept deliberately close to the original so it stays diffable against upstream.

#include "odgi_lay.hpp"

#include <sdsl/util.hpp>

#include <algorithm>

namespace gbz2layout {

OdgiLayout::OdgiLayout(const std::vector<double>& X, const std::vector<double>& Y) {
    std::vector<std::uint64_t> vals;
    vals.reserve(X.size() + Y.size());
    for (auto& v : X) { min_value = std::min(v, min_value); }
    for (auto& v : Y) { min_value = std::min(v, min_value); }
    conv_t x, y;
    for (std::uint64_t i = 0; i < X.size(); ++i) {
        x.d = X[i] - min_value;
        y.d = Y[i] - min_value;
        vals.push_back(x.i);
        vals.push_back(y.i);
    }
    sdsl::util::assign(xy, sdsl::enc_vector<>(vals));
}

void OdgiLayout::serialize(std::ostream& out) {
    sdsl::write_member(min_value, out);
    xy.serialize(out);
}

void OdgiLayout::load(std::istream& in) {
    sdsl::read_member(min_value, in);
    xy.load(in);
}

// NB: odgi's Layout::size() returns xy.size()/2 -- the NODE-END count, since xy
// interleaves x,y. Its get_x(i)/get_y(i) then index entry i, which is a node end
// (2*rank + endpoint), not a node. Kept identical so the semantics match.
std::size_t OdgiLayout::size() const {
    return xy.size() / 2;
}

double OdgiLayout::get_x(std::uint64_t i) const {
    conv_t x;
    x.i = xy[2 * i];
    return x.d + min_value;
}

double OdgiLayout::get_y(std::uint64_t i) const {
    conv_t y;
    y.i = xy[2 * i + 1];
    return y.d + min_value;
}

} // namespace gbz2layout
