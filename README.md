# gbz2layout

Compute a 2D layout of a pangenome graph **directly from a GBZ file**, and write
it in odgi's `.lay.tsv` format.

Given a GBZ (the compressed pangenome format produced by [vg](https://github.com/vgteam/vg)),
`gbz2layout` produces an x/y coordinate for every node — the same kind of layout
you'd get from `odgi layout`, but without ever building or sorting an odgi graph.

## Why

The usual way to get a 2D pangenome layout is `odgi build` → `odgi sort` →
`odgi layout`. On large graphs that's expensive: `odgi build` needs an
uncompressed graph in memory, and `odgi sort` can need well over 100 GB of RAM.

`gbz2layout` skips both. It reads the compressed GBZ, lays it out in place, and
its memory footprint stays small enough to run whole chromosomes on a desktop.

## How it works

The GBZ already holds the graph topology *and* every haplotype path, compressed.
`gbz2layout` uses them directly:

1. **Path index** — build a lightweight index of path positions straight from the
   GBWT (the path store inside the GBZ). To keep memory bounded on
   high-coverage graphs, an optional **per-node coverage cap** keeps only up to
   *N* occurrences of each node. Every node is always kept at least once, so node
   coverage stays 100% — only redundant backbone is thinned.
2. **Reference-anchored init** — instead of sorting the graph, each node is given
   a starting x-coordinate from the reference path's base-pair position, then
   propagated into bubbles. This gives the layout a globally correct starting
   shape, which is what makes the sort step unnecessary.
3. **Layout** — a port of odgi's PG-SGD layout (`path_linear_sgd_layout`) refines
   the coordinates so that 2D distance between nodes approximates their distance
   along the haplotypes.
4. **Output** — an odgi-compatible `.lay.tsv` (two rows per node, one per node
   end).

## Build

Dependencies are vendored as git submodules, pinned to the exact commits that
match the vg release used to produce GBZ files (this avoids serialization skew):

```bash
git clone --recursive git@github.com:ScottMastro/gbz2layout.git
cd gbz2layout
```

Build the dependency stack once (see the rebuild steps below), then:

```bash
make tool        # builds build/gbz2layout
```

| dependency     | repo                    |
|----------------|-------------------------|
| sdsl-lite      | vgteam/sdsl-lite        |
| libhandlegraph | vgteam/libhandlegraph   |
| gbwt           | jltsiren/gbwt           |
| gbwtgraph      | jltsiren/gbwtgraph      |

<details>
<summary>Building the dependency stack (one time)</summary>

```bash
PREFIX=$PWD/local

# 1. sdsl-lite
cd deps/sdsl-lite && ./install.sh "$PREFIX" && cd ../..

# 2. libhandlegraph
cd deps/libhandlegraph && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release .. && make -j4 && make install
cd ../../..

# 3. gbwt
cd deps/gbwt && make -j4
mkdir -p "$PREFIX/include/gbwt"; cp include/gbwt/*.h "$PREFIX/include/gbwt/"
cp lib/libgbwt.a "$PREFIX/lib/" && cd ../..

# 4. gbwtgraph
cd deps/gbwtgraph && make -j4
mkdir -p "$PREFIX/include/gbwtgraph"; cp include/gbwtgraph/*.h "$PREFIX/include/gbwtgraph/"
cp lib/libgbwtgraph.a "$PREFIX/lib/" && cd ../..
```

</details>

## Usage

```bash
build/gbz2layout graph.gbz -o mygraph --cap 30 --iter 30 --threads 8
```

This reads `graph.gbz` and writes `mygraph.lay.tsv`.

| option | description |
|--------|-------------|
| `-o PREFIX` | output prefix (writes `PREFIX.lay.tsv`) |
| `--cap N` | per-node coverage cap, `0` = uncapped (default 30) |
| `--iter N` | max SGD iterations (default 30) |
| `--threads N` | worker threads |
| `--chromosome NAME` | lay out only one chromosome, extracted on the fly from a whole-genome GBZ |
| `--init ref\|random` | initialization mode (default `ref`) |
| `--emit-links` | also write `PREFIX.links.tsv` (node edges, for visualization) |
| `--seed N` | RNG seed |

For a whole-genome GBZ, `--chromosome chr1` finds that chromosome's connected
component and lays out only those nodes, so you never have to extract a
per-chromosome graph first.

## Tools

- `tools/render_layout.py <lay.tsv> <out.png>` — render a whole layout to a PNG.
- `tools/render_edges.py <lay.tsv> <links.tsv> <out.png> <xlo> <xhi>` — zoomed
  render including edges, so bubbles read as loops rather than loose segments
  (needs `--emit-links` output).
