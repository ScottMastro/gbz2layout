#include "components.hpp"

#include <limits>

namespace gbz2layout {

using handlegraph::handle_t;

std::uint64_t weakly_connected_components(const gbwtgraph::GBWTGraph& graph,
                                          const XP& xp,
                                          std::vector<std::int32_t>& component) {
    const std::uint64_t N = xp.node_count();
    component.assign(N, -1);
    std::int32_t next = 0;
    std::vector<std::uint64_t> stack;

    // Iterate in rank order so component indices come out in the same order
    // odgi's for_each_handle would produce them.
    for (std::uint64_t r0 = 0; r0 < N; ++r0) {
        if (component[r0] != -1) continue;

        component[r0] = next;
        stack.clear();
        stack.push_back(r0);

        while (!stack.empty()) {
            std::uint64_t r = stack.back();
            stack.pop_back();
            handle_t h = graph.get_handle(xp.node_id_of_rank(r), false);
            // Both directions: weak connectivity ignores orientation.
            for (bool go_left : {false, true}) {
                graph.follow_edges(h, go_left, [&](const handle_t& nb) {
                    std::uint64_t nr = xp.rank_of_handle(nb);
                    if (nr < N && component[nr] == -1) {
                        component[nr] = next;
                        stack.push_back(nr);
                    }
                    return true;
                });
            }
        }
        ++next;
    }
    return (std::uint64_t)next;
}

namespace {
// odgi's coord_range_2d_t, with max_* initialised to lowest() rather than min().
// See components.hpp for why that matters.
struct coord_range_2d_t {
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    double x_offset = 0;
    double y_offset = 0;
    double width()  const { return max_x - min_x; }
    double height() const { return max_y - min_y; }
    void include(double x, double y) {
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }
};
} // namespace

void pack_components(std::uint64_t ncomp,
                     const std::vector<std::int32_t>& component,
                     std::vector<std::atomic<double>>& X,
                     std::vector<std::atomic<double>>& Y,
                     double border) {
    if (ncomp == 0) return;
    const std::uint64_t N = component.size();

    std::vector<coord_range_2d_t> ranges(ncomp);
    for (std::uint64_t r = 0; r < N; ++r) {
        std::int32_t c = component[r];
        if (c < 0) continue;
        for (std::uint64_t j = 2 * r; j <= 2 * r + 1; ++j)
            ranges[c].include(X[j].load(), Y[j].load());
    }

    double curr_y_offset = border;
    for (auto& cr : ranges) {
        cr.x_offset = cr.min_x - border;
        cr.y_offset = curr_y_offset - cr.min_y;
        curr_y_offset += cr.height() + border;
    }

    for (std::uint64_t r = 0; r < N; ++r) {
        std::int32_t c = component[r];
        if (c < 0) continue;
        const auto& cr = ranges[c];
        for (std::uint64_t j = 2 * r; j <= 2 * r + 1; ++j) {
            X[j].store(X[j].load() - cr.x_offset);
            Y[j].store(Y[j].load() + cr.y_offset);
        }
    }
}

} // namespace gbz2layout
