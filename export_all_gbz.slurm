#!/bin/bash
#SBATCH --job-name=gbz2layout-export-all
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=90G                  # chr1 export peaks ~40 GB build + ~5.4 GB resident genome
#SBATCH --cpus-per-task=4          # export is mostly single-threaded (GBWT build + BFS)
# #SBATCH --partition=<fill in>    # uncomment + set for your cluster
# #SBATCH --account=<fill in>      # uncomment if your cluster requires one
set -euo pipefail

# Extract a standalone per-chromosome GBZ for every reference contig out of the
# whole-genome HPRC-v2 GBZ, in one genome load. This is the memory-heavy step
# the 15 GB laptop can't do for the big chromosomes: rebuilding a chromosome's
# GBWT peaks at ~7 GB per ~2M-node chromosome (chr1 ~40 GB) on top of the ~5.4 GB
# resident genome. On a 64-90 GB node it runs comfortably in one pass.
#
# Contigs are processed smallest-first and skip-if-exists, so a resubmit resumes
# where a previous run stopped (or after you seed OUT with laptop-made files).

# ---- paths: EDIT THESE for your cluster ----
REPO=${REPO:-$HOME/gbz2layout}                          # gbz2layout checkout with a built local/ prefix
GBZ=${GBZ:-$SCRATCH/hprc-v2.0-mc-grch38.gbz}            # whole-genome GBZ (copy from the NAS)
OUT=${OUT:-$SCRATCH/hprc-v2-gbz}                        # per-chromosome GBZ output dir
NAS=${NAS:-}                                            # optional rsync target (e.g. NAS gbz dir); empty = skip

cd "$REPO"

# Build the CUDA-free binary if absent. Requires the vendored local/ prefix
# (fork of gbwt-mmap + vgteam sdsl) to be built already — see the repo README.
# `make tool-nocuda` links a stub GPU backend so no CUDA runtime is needed.
if [ ! -x build/gbz2layout-nocuda ]; then
  echo "building gbz2layout-nocuda ..."
  make tool-nocuda
fi

mkdir -p "$OUT"
echo "host=$(hostname)  start=$(date -Iseconds)"
echo "input=$GBZ"
echo "output=$OUT"

/usr/bin/time -v build/gbz2layout-nocuda "$GBZ" --export-all-gbz "$OUT"

echo "=== exported ==="
ls -la "$OUT"

# Optional: push the main chromosomes to the NAS (skips the tiny decoy contigs).
if [ -n "$NAS" ]; then
  echo "rsync -> $NAS"
  for c in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 \
           chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY chrM; do
    [ -f "$OUT/$c.v2.gbz" ] && rsync -av "$OUT/$c.v2.gbz" "$NAS"/
  done
fi

echo "end=$(date -Iseconds)"
