// eval_crossings — legibility metric: estimate how many edges cross other
// edges. Complements eval_layout: Spearman(bp,2D) measures genomic *fidelity*,
// this measures visual *readability* (the classic graph-drawing crossing
// number). Lower = cleaner drawing.
//
// Reads a .lay.tsv (2 rows/node: 2r start, 2r+1 end) and a .links.tsv
// (a<TAB>b node-rank pairs, as written by gbz2layout --emit-links). An edge
// a->b is the segment (x2[a],y2[a]) -> (x1[b],y1[b]), matching render_edges.py.
//
// Exact all-pairs is O(E^2); instead we bucket edges by segment-midpoint X and
// test pairs only within a bucket (width >> typical edge span), which counts
// local crossings — the ones the eye sees — consistently across layouts. Dense
// buckets are capped by uniform subsampling and the count scaled back up.
//
// usage: eval_crossings <lay.tsv> <links.tsv> [bucket_bp] [cap_per_bucket]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static inline int orient(double ax, double ay, double bx, double by, double cx, double cy) {
    double v = (by - ay) * (cx - bx) - (bx - ax) * (cy - by);
    return (v > 0) - (v < 0);
}
// proper intersection only (shared endpoints / collinear touching ignored)
static inline bool seg_int(double ax, double ay, double bx, double by,
                           double cx, double cy, double dx, double dy) {
    int o1 = orient(ax, ay, bx, by, cx, cy);
    int o2 = orient(ax, ay, bx, by, dx, dy);
    int o3 = orient(cx, cy, dx, dy, ax, ay);
    int o4 = orient(cx, cy, dx, dy, bx, by);
    return (o1 != o2) && (o3 != o4);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: eval_crossings <lay.tsv> <links.tsv> [bucket_bp] [cap_per_bucket]\n"; return 1; }
    std::string lay_path = argv[1], links_path = argv[2];
    double   bucket_bp = (argc >= 4) ? std::stod(argv[3]) : 20000.0;
    std::uint64_t cap  = (argc >= 5) ? std::stoull(argv[4]) : 3000;

    auto sd = [](const std::string& s){ try { return std::stod(s); } catch (const std::out_of_range&) { return 0.0; } };

    // load node endpoints: idx 2r -> (x1,y1), 2r+1 -> (x2,y2)
    std::vector<double> x1, y1, x2, y2;
    {
        std::ifstream in(lay_path);
        if (!in) { std::cerr << "cannot open " << lay_path << "\n"; return 1; }
        std::string line; std::getline(in, line);   // header
        while (std::getline(in, line)) {
            std::istringstream ss(line); std::string tok;
            std::getline(ss, tok, '\t'); std::uint64_t idx = std::stoull(tok);
            std::getline(ss, tok, '\t'); double X = sd(tok);
            std::getline(ss, tok, '\t'); double Y = sd(tok);
            std::uint64_t r = idx / 2;
            if (r >= x1.size()) { x1.resize(r + 1, 0); y1.resize(r + 1, 0); x2.resize(r + 1, 0); y2.resize(r + 1, 0); }
            if (idx & 1) { x2[r] = X; y2[r] = Y; } else { x1[r] = X; y1[r] = Y; }
        }
    }
    const std::uint64_t N = x1.size();

    // load edges as segments (x2[a],y2[a]) -> (x1[b],y1[b]); keep node ids to
    // exclude edge pairs that legitimately share a node.
    struct Edge { double ax, ay, bx, by; std::uint32_t na, nb; double mid; };
    std::vector<Edge> edges;
    {
        std::ifstream in(links_path);
        if (!in) { std::cerr << "cannot open " << links_path << "\n"; return 1; }
        std::string line; std::getline(in, line);   // header
        while (std::getline(in, line)) {
            std::istringstream ss(line); std::string tok;
            std::getline(ss, tok, '\t'); std::uint64_t a = std::stoull(tok);
            std::getline(ss, tok, '\t'); std::uint64_t b = std::stoull(tok);
            if (a >= N || b >= N) continue;
            Edge e{ x2[a], y2[a], x1[b], y1[b], (std::uint32_t)a, (std::uint32_t)b, 0.0 };
            e.mid = 0.5 * (e.ax + e.bx);
            edges.push_back(e);
        }
    }
    const std::uint64_t E = edges.size();
    if (E == 0) { std::cerr << "no edges\n"; return 1; }

    // bucket by midpoint X
    double xmin = 1e300, xmax = -1e300;
    for (const auto& e : edges) { xmin = std::min(xmin, e.mid); xmax = std::max(xmax, e.mid); }
    std::uint64_t nb = std::max<std::uint64_t>(1, (std::uint64_t)((xmax - xmin) / bucket_bp) + 1);
    std::vector<std::vector<std::uint32_t>> buckets(nb);
    for (std::uint32_t i = 0; i < E; ++i) {
        std::uint64_t b = (std::uint64_t)((edges[i].mid - xmin) / bucket_bp);
        if (b >= nb) b = nb - 1;
        buckets[b].push_back(i);
    }

    // count crossings within each bucket; subsample capped buckets and rescale
    std::mt19937_64 rng(1234567);
    double est_crossings = 0.0;
    std::uint64_t tested_pairs = 0;
    std::uint64_t capped_buckets = 0;
    for (auto& bk : buckets) {
        std::vector<std::uint32_t>* use = &bk;
        std::vector<std::uint32_t> sample;
        double scale = 1.0;
        if (bk.size() > cap) {
            sample = bk;
            std::shuffle(sample.begin(), sample.end(), rng);
            sample.resize(cap);
            use = &sample;
            // pair count scales as m^2; rescale by (full/sub)^2
            double full = (double)bk.size(), sub = (double)cap;
            scale = (full * (full - 1)) / (sub * (sub - 1));
            ++capped_buckets;
        }
        const auto& v = *use;
        for (std::size_t p = 0; p < v.size(); ++p) {
            const Edge& e1 = edges[v[p]];
            for (std::size_t q = p + 1; q < v.size(); ++q) {
                const Edge& e2 = edges[v[q]];
                if (e1.na == e2.na || e1.na == e2.nb || e1.nb == e2.na || e1.nb == e2.nb) continue;
                ++tested_pairs;
                if (seg_int(e1.ax, e1.ay, e1.bx, e1.by, e2.ax, e2.ay, e2.bx, e2.by))
                    est_crossings += scale;
            }
        }
    }

    std::cout << "layout:   " << lay_path << "\n";
    std::cout << "  nodes            " << N << "\n";
    std::cout << "  edges            " << E << "\n";
    std::cout << "  bucket_bp        " << bucket_bp << "  (" << nb << " buckets, " << capped_buckets << " subsampled)\n";
    std::cout << "  pairs tested     " << tested_pairs << "\n";
    std::cout << "  est. crossings   " << (std::uint64_t)(est_crossings + 0.5)
              << "   <- legibility (lower=better)\n";
    std::cout.precision(6);
    std::cout << "  crossings/edge   " << (est_crossings / (double)E) << "\n";
    return 0;
}
