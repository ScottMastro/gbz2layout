# gbz2layout

Compute an odgi-compatible 2D layout (`.lay.tsv` / `.lay`) **directly from a
GBZ**, skipping `odgi build` + `odgi sort`. Route B from `GBZ_LAYOUT_PROJECT.md`
in the pangyplot repo: a standalone tool that lifts odgi's interface-based
PG-SGD layout + XP path index and links `gbwtgraph` for the graph backend.

**Status:** dependency stack built & validated. No layout code yet.

## Layout

```
deps/     vendored dependencies, pinned to vg 1.69.0 "Bologna" commits
local/    install prefix (static libs + headers) — everything links here
build/    build logs
src/      gbz2layout sources (TODO)
```

## Dependencies

Pinned to the exact commits vg 1.69.0 uses to *produce* the v2 GBZ files, to
avoid sdsl/gbwt version skew (the #1 risk).

| dep            | repo                        | commit    |
|----------------|-----------------------------|-----------|
| sdsl-lite      | vgteam/sdsl-lite (fork)     | `8abd7c9` |
| libhandlegraph | vgteam/libhandlegraph       | `5ab3b33` |
| gbwt           | jltsiren/gbwt               | `bde6858` |
| gbwtgraph      | jltsiren/gbwtgraph          | `0698c1b` |

Source of truth for pins: `deps/` submodules of vg at tag `v1.69.0`.

### Rebuild from scratch

```bash
PREFIX=$PWD/local

# 1. sdsl-lite (writes Make.helper pointing INC_DIR/LIB_DIR at $PREFIX)
cd deps/sdsl-lite && ./install.sh "$PREFIX" && cd ../..

# 2. libhandlegraph (cmake)
cd deps/libhandlegraph && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release .. && make -j4 && make install
cd ../../..

# 3. gbwt (Makefile reads ../sdsl-lite/Make.helper; install manually)
cd deps/gbwt && make -j4
cp include/gbwt/*.h "$PREFIX/include/gbwt/" 2>/dev/null || { mkdir -p "$PREFIX/include/gbwt"; cp include/gbwt/*.h "$PREFIX/include/gbwt/"; }
cp lib/libgbwt.a "$PREFIX/lib/" && cd ../..

# 4. gbwtgraph (Makefile links -lgbwt -lhandlegraph -lsdsl from $PREFIX)
cd deps/gbwtgraph && make -j4
mkdir -p "$PREFIX/include/gbwtgraph"
cp include/gbwtgraph/*.h "$PREFIX/include/gbwtgraph/"
cp lib/libgbwtgraph.a "$PREFIX/lib/" && cd ../..
```

Installed static libs in `local/lib/`: `libsdsl.a`, `libdivsufsort.a`,
`libdivsufsort64.a`, `libhandlegraph.a`, `libgbwt.a`, `libgbwtgraph.a`.

Link order (matters for static): `-lgbwtgraph -lgbwt -lhandlegraph -lsdsl
-ldivsufsort -ldivsufsort64`, plus `-fopenmp -pthread`.

## Validation

The gbwtgraph build produces helper tools under `deps/gbwtgraph/bin/`. They link
libhandlegraph dynamically, so run with:

```bash
LD_LIBRARY_PATH=local/lib deps/gbwtgraph/bin/gbz_stats -g <graph.gbz>
```

Smoke test on the local chrY v2 GBZ passed:

```
$ gbz_stats -g /mnt/data/pangyplot/bench/chrY.v2.gbz
Nodes  1046775
Edges  1394620
# 0.49 s wall, ~197 MB peak RSS
```

## Test data

- `/mnt/data/pangyplot/bench/chrY.v2.gbz` — chrY v2 GBZ (test fixture)
- `/mnt/data/pangyplot/data/hprc-v2.0/chrY/clip/` — chrY golden files
  (`chrY.og`, `chrY.sorted.gfa`, `chrY.lay`, `chrY.lay.tsv`) for M3 layout-quality
  comparison
- `/mnt/data/pangyplot/data/hprc-v2.0/chr22/clip/chr22.unsorted.og` — for the
  §6 XP-memory experiment (2.18M nodes, 578M steps)

## M1 findings (chrY, via `build/xp_probe`)

Load-bearing experiment: does an XP-equivalent path index fit in RAM at scale?
Run: `LD_LIBRARY_PATH=local/lib ./build/xp_probe <graph.gbz>`

**Three structural findings (all confirm project risks):**

1. **Haplotypes live in the GBWT, not the handlegraph API.** `for_each_path_handle`
   exposed only **1** path (the GRCh38 reference, 437K steps); the real **1614
   paths / 124 haplotypes (54.3M steps)** are GBWT sequences. Iterate them via
   `index.extract(seq)` over even (forward) sequence ids. The XP builder must be
   GBWT-backed.
2. **Node IDs are sparse, not dense.** chrY ids span 208.5M–211.3M for 1.05M
   nodes. The doc's "GBWT ids are already dense" is false for per-chromosome
   GBZs → an id→rank remap is mandatory (but cheap: ~10 MiB, 1.2% of XP).
3. **Storage order is locally reference-consistent but globally scrambled**
   (median adjacent-rank jump = 1, but Spearman(pos,rank) = 0.29). odgi's default
   node-order init would get local structure right and the global genome ribbon
   wrong → use **reference-anchored init** (node X = reference bp coordinate);
   this is also what makes `odgi sort` droppable.

**Memory (per-node coverage cap sweep, lean = derive positions/offsets):**

| cap C | steps kept | node coverage | lean MiB (chrY) |
|------:|-----------:|--------------:|----------------:|
| ∞     | 54.3M      | 100%          | 341 |
| 30    | 18.9M      | 100%          | 125 |
| 15    | 10.3M      | 100%          | 74  |
| 8     | 6.0M       | 100%          | 47  |

Coverage stays 100% at any cap≥1 (first occurrence of every node kept; rare
variants preserved, redundant backbone thinned). Capping pays off *more* on
high-haplotype chromosomes: chrY averages 52 steps/node, chr22 ~265.

**chr22 confirmation run (real 578M steps, 464 haplotypes, 2166 paths):**

| cap C | steps kept | ret% | node coverage | lean MiB (chr22) |
|------:|-----------:|-----:|--------------:|-----------------:|
| ∞     | 578.5M     | 100% | 100%          | 3888 |
| 30    | 48.5M      | 8.4% | 100%          | 343  |
| 15    | 25.5M      | 4.4% | 100%          | 190  |
| 8     | 14.3M      | 2.5% | 100%          | 115  |

chr22 full-path lean = 3.9 GiB (fits on its own — chr22 needs no cap); cap-30 =
343 MiB. Extrapolated chr1 (~11M nodes, ~2.9B steps): full-path ~20 GiB (needs
cap), cap-15 ~1.3 GiB, cap-30 ~2.5 GiB — all in budget. chr22 GBZ was built from
`chr22.vg` via `vg gbwt -E` in 9m18s at 11.5 GB peak (fits the 15 GB desktop).

Reference-order on chr22: Spearman 0.41 (median adjacent jump 1, adj-increasing
98.3%) — same local-yes/global-no pattern as chrY. Reference-anchored init.

**Verdict:** viable, conditional on three levers — GBWT-backed lean XP builder,
per-node cap (C≈15–30), reference-anchored init + streaming/mmap build.
Layout *quality* at each cap is unmeasured — validate at M3 vs golden chrY.

## M2/M3: the tool works (chrY validated vs golden)

`make tool` builds `build/gbz2layout`. Pipeline:
`GBZ -> GBWT-backed lean XP (per-node cap) -> reference-anchored init (BFS from
the reference path) -> PG-SGD (odgi CPU port) -> odgi-compatible .lay.tsv`.

```bash
build/gbz2layout <graph.gbz> -o <prefix> --cap 30 --iter 30 --threads 8
build/eval_layout <graph.gbz> <prefix>.lay.tsv          # intrinsic quality
python3 tools/render_layout.py <prefix>.lay.tsv out.png # visual QC
```

Components: `src/xp.{hpp,cpp}` (XP), `src/sgd_layout.{hpp,cpp}` (PG-SGD port,
faithful to odgi incl. vendored dirtyzipf/Xoshiro at odgi's pinned commits under
`src/third_party/`), `src/gbz2layout.cpp` (driver), `src/eval_layout.cpp`,
`src/test_xp.cpp`. The one algorithm adaptation: `unpack_number(handle)` ->
`XP::rank_of_handle` (sparse ids).

**chrY result (cap 30, 30 iters, 8 threads):**

| metric | value |
|---|---|
| Spearman(bp, 2D dist) | **0.970** (Pearson 0.996, norm stress 0.074) |
| peak RSS | **421 MB** (entire layout) |
| time | 8m35s (5.7B SGD updates, CPU) |
| visual vs golden | feature positions along X match; same quiet tail 2.7e7->57e7 |

Proves all four bets: GBWT-native layout works; `odgi sort` is droppable
(reference-anchored init handles the unsorted graph — validated, not assumed);
per-node cap-30 preserves quality; memory is tiny. 421 MB for chrY means chr1 at
cap-15/30 is in reach on the 15 GB box (the project's headline goal).

The layout idx column is odgi-compatible (2 rows/node: 2*rank, 2*rank+1 in
for_each_handle order). The GFA fed to `pangyplot add` must share that node order
— the one integration contract still to verify end-to-end.

## chr1: whole-genome-GBZ extraction (the headline goal)

chr1's per-chromosome `.vg` is 35 GB (won't fit disk/RAM here), so instead the
tool extracts chr1 directly from the **compressed whole-genome clip GBZ** (5.1 GB
file, 13.5 GB loaded) via `--chromosome`:

```bash
build/gbz2layout wg-clip.v2.gbz --chromosome chr1 -o chr1_cap8 \
    --cap 8 --iter 30 --updates-per-node 30 --threads 8
```

`--chromosome NAME` finds NAME's reference path, BFS's its connected component
(MC chromosomes are disjoint components), and builds the XP over only those
nodes/sequences. Pass 1 peeks each sequence's first node via `index.start(seq)`
to skip other chromosomes without extracting them.

**Result (validation ladder):**

| chrom | nodes | source | peak RSS | time | quality |
|-------|------:|--------|---------:|-----:|---------|
| chrY  | 1.05M | per-chr GBZ | 421 MB | 8.6m | Spearman 0.970, matches golden |
| chr22 | 2.18M | per-chr GBZ | 829 MB | 12.6m | Spearman 0.9995 |
| chr1  | 10.6M | **whole-genome GBZ extract** | **13.14 GB** | 27m | coherent ~247 Mb ribbon, centromere at ~1.23e8 |

**chr1 v2 laid out on a 15 GB desktop, from the compressed GBZ, no `odgi build`
/ `odgi sort`.** The project's central goal. Memory bound is the whole-genome
load (13.5 GB) + chr1's cap-8 XP (~1 GB); higher caps would need a bit more RAM
or a pre-extracted per-chr chr1 GBZ (then layout is ~2 GB).

## Next: milestones (see GBZ_LAYOUT_PROJECT.md)

- **M1** — XP-memory experiment (§6): iterate all paths over the chr22 GBWTGraph,
  accumulate XP structures, measure peak RSS. Go/no-go gate.
- **M2** — `gbz2layout`: GBZ loader → GBWT-backed XP builder → lifted
  `path_linear_sgd_layout` (id→rank remap) → emit `.lay.tsv`.
- **M3** — chrY validation vs golden `chrY.lay.tsv`.
- **M4** — path subsampling + reference-anchored init.
