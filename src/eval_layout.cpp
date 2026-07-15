// eval_layout — intrinsic layout-quality metric (no golden node correspondence
// needed). Loads a GBZ + a .lay.tsv produced over that same GBZ, then measures
// how well 2D distance tracks path (bp) distance over sampled same-path step
// pairs. This is exactly what PG-SGD optimizes, so it is a faithful quality
// proxy and is directly comparable across caps / parameter choices.
//
// Reports Pearson & Spearman correlation between path-bp-distance and
// 2D-distance, plus a scale-normalized stress. Higher correlation / lower
// stress = better layout.

#include "xp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace gbz2layout;
using handlegraph::as_integers;
using handlegraph::as_path_handle;

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: eval_layout <graph.gbz> <layout.lay.tsv> [samples] [cap]\n"; return 1; }
    std::string gbz_path = argv[1], tsv_path = argv[2];
    std::uint64_t samples = (argc >= 4) ? std::stoull(argv[3]) : 2000000;
    std::uint64_t cap = (argc >= 5) ? std::stoull(argv[4]) : 0;   // cap 0 = eval over all paths

    gbwtgraph::GBZ gbz;
    { std::ifstream in(gbz_path, std::ios::binary); gbz.simple_sds_load(in); }
    XP xp;
    xp.build(gbz, cap, false);
    const std::uint64_t N = xp.node_count();

    // load layout: idx 2r/2r+1 -> node rank r; use node center
    std::vector<double> cx(N, 0), cy(N, 0);
    std::vector<char> have(N, 0);
    {
        std::ifstream in(tsv_path);
        if (!in) { std::cerr << "cannot open " << tsv_path << "\n"; return 1; }
        std::string line; std::getline(in, line); // header
        std::vector<double> tx(2 * N, 0), ty(2 * N, 0);
        std::vector<char> th(2 * N, 0);
        while (std::getline(in, line)) {
            std::uint64_t idx; double x, y;
            std::istringstream ss(line);
            std::string tok;
            // tolerant parse: glibc stod throws out_of_range on subnormal
            // (underflow) coordinates, e.g. a node pulled to Y~=0; treat as 0.
            auto sd = [](const std::string& s){ try { return std::stod(s); } catch (const std::out_of_range&) { return 0.0; } };
            std::getline(ss, tok, '\t'); idx = std::stoull(tok);
            std::getline(ss, tok, '\t'); x = sd(tok);
            std::getline(ss, tok, '\t'); y = sd(tok);
            if (idx < 2 * N) { tx[idx] = x; ty[idx] = y; th[idx] = 1; }
        }
        for (std::uint64_t r = 0; r < N; ++r) {
            if (th[2 * r] && th[2 * r + 1]) {
                cx[r] = 0.5 * (tx[2 * r] + tx[2 * r + 1]);
                cy[r] = 0.5 * (ty[2 * r] + ty[2 * r + 1]);
                have[r] = 1;
            }
        }
    }

    // sample same-path step pairs; collect (bp_dist, 2d_dist)
    std::mt19937_64 rng(12345);
    const auto& npi = xp.get_npi_iv();
    const auto& nr  = xp.get_nr_iv();
    std::uniform_int_distribution<std::uint64_t> dis_step(0, xp.get_np_bv().size() - 1);

    std::vector<double> bp, d2;
    bp.reserve(samples); d2.reserve(samples);
    std::uint64_t tries = 0, capmax = samples * 20;
    while (bp.size() < samples && tries < capmax) {
        ++tries;
        std::uint64_t si = dis_step(rng);
        std::uint64_t pid = npi[si];
        auto path = as_path_handle(pid);
        std::uint64_t cnt = xp.get_path_step_count(path);
        if (cnt < 2) continue;
        std::uint64_t ra = nr[si] - 1;
        std::uint64_t rb = std::uniform_int_distribution<std::uint64_t>(0, cnt - 1)(rng);
        if (rb == ra) continue;
        handlegraph::step_handle_t sa, sb;
        as_integers(sa)[0] = pid; as_integers(sa)[1] = ra;
        as_integers(sb)[0] = pid; as_integers(sb)[1] = rb;
        std::uint64_t na = xp.rank_of_handle(xp.get_handle_of_step(sa));
        std::uint64_t nb = xp.rank_of_handle(xp.get_handle_of_step(sb));
        if (!have[na] || !have[nb]) continue;
        double pd = std::abs((double)xp.get_position_of_step(sa) - (double)xp.get_position_of_step(sb));
        double dx = cx[na] - cx[nb], dy = cy[na] - cy[nb];
        double dd = std::sqrt(dx * dx + dy * dy);
        bp.push_back(pd); d2.push_back(dd);
    }

    std::uint64_t n = bp.size();
    if (n < 100) { std::cerr << "too few samples (" << n << ")\n"; return 1; }

    auto pearson = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, m = n;
        for (std::uint64_t i = 0; i < n; ++i) { sa += a[i]; sb += b[i]; saa += a[i]*a[i]; sbb += b[i]*b[i]; sab += a[i]*b[i]; }
        double cov = sab - sa*sb/m, va = saa - sa*sa/m, vb = sbb - sb*sb/m;
        return (va > 0 && vb > 0) ? cov / std::sqrt(va*vb) : 0.0;
    };
    double r_p = pearson(bp, d2);

    // Spearman: correlation of value-ranks
    auto rankify = [&](const std::vector<double>& v) {
        std::vector<std::uint64_t> ord(n); for (std::uint64_t i = 0; i < n; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](std::uint64_t a, std::uint64_t b){ return v[a] < v[b]; });
        std::vector<double> rk(n); for (std::uint64_t k = 0; k < n; ++k) rk[ord[k]] = (double)k;
        return rk;
    };
    double r_s = pearson(rankify(bp), rankify(d2));

    // scale-normalized stress: best-fit scale k minimizing sum((d2 - k*bp)^2)
    double num = 0, den = 0;
    for (std::uint64_t i = 0; i < n; ++i) { num += d2[i]*bp[i]; den += bp[i]*bp[i]; }
    double k = den > 0 ? num/den : 0;
    double stress = 0, norm = 0;
    for (std::uint64_t i = 0; i < n; ++i) { double e = d2[i]-k*bp[i]; stress += e*e; norm += d2[i]*d2[i]; }
    double nstress = norm > 0 ? std::sqrt(stress/norm) : 0;

    std::cout << "layout: " << tsv_path << "\n";
    std::cout << "  pairs sampled     " << n << "\n";
    std::cout << "  Pearson(bp,2d)    " << r_p << "\n";
    std::cout << "  Spearman(bp,2d)   " << r_s << "   <- primary quality metric (higher=better)\n";
    std::cout << "  norm. stress      " << nstress << "   (lower=better)\n";
    return 0;
}
