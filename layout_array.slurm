#!/bin/bash
#SBATCH --job-name=gbz-layout
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --error=slurm-%x-%a-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-24               # one task per chromosome (see CHROMS); add %N to throttle
#SBATCH --constraint=AlmaLinux9    # see GLIBC note below -- do not drop this
# #SBATCH --partition=<fill in>
# #SBATCH --account=<fill in>

# GLIBC: the cluster runs TWO operating systems and the binary only works on one.
#
#   cn675  ActiveFeatures=AlmaLinux9           OS=5.14.0   glibc 2.34   works
#   cn134  ActiveFeatures=AlmaLinux8,avx512    OS=4.18.0   glibc 2.28   fails
#
# The login node is AlmaLinux9, so a binary built there needs glibc 2.34 and will
# not even LOAD on an AlmaLinux8 node:
#   /lib64/libc.so.6: version `GLIBC_2.34' not found
# The task dies in ~0.03s having done nothing, which reads like a flake but is
# deterministic per node.
#
# NOTE the trap: `avx512` is present ONLY on the AlmaLinux8 nodes -- it is an
# older Intel feature the newer machines don't advertise. Constraining on avx512
# therefore selects exactly the nodes that CANNOT run this binary. Constrain on
# the OS, not on a CPU feature standing in for "newer".
#
# The alternative, if AlmaLinux9 nodes are ever scarce: build on an AlmaLinux8
# node (glibc 2.28) and the binary runs on both, since glibc is forward
# compatible. Build old, run anywhere.
#
# (Same family as the -march=native SIGILL on the export array: one binary,
# heterogeneous nodes. That was the instruction set; this is the C library.)

# Per-chromosome PG-SGD layout, one array task each, reading the per-chr GBZs the
# export array already wrote to hprc-v2-gbz/. Wall-clock is the slowest single
# chromosome (chr1) rather than the serial sum. Resubmitting resumes: finished
# chromosomes exit early, and the write is atomic so a killed task never leaves a
# file that looks done.
#
# NO --gpu HERE. Compute nodes have no CUDA, so this uses gbz2layout-nocuda,
# which aborts loudly on --gpu rather than silently pretending. The minibatch
# update runs on the CPU across --cpus-per-task threads. Expect this to be several
# times slower per chromosome than the laptop's GPU runs -- the point is that 25
# of them run at once.
#
# Sizing (measured on the laptop, K=128):
#   chr22  2.18M nodes  peak 2.0 GB
#   chr14  4.21M nodes  peak 4.1 GB
#   chr13  4.32M nodes  peak 5.4 GB   <- peak tracks K x PATH LENGTH, not node count
#   chr1  ~10.6M nodes  peak ~12.5 GB (projected)
# 32G leaves headroom over the worst case. K stays 128 everywhere: coverage and
# total updates are K-invariant, and it only bounds resident memory -- the laptop
# needed K=32 for chr1 because it has 15G total, not because K changes the result.

module load gcc                    # matching libstdc++ for the prebuilt binary
set -euo pipefail

# ---- paths: override via env (sbatch inherits the submit environment) ----
REPO=${REPO:-$PWD}
GBZDIR=${GBZDIR:-$REPO/hprc-v2-gbz}      # written by the export array
OUT=${OUT:-$REPO/hprc-v2-layout}
BIN="$REPO/build/gbz2layout-nocuda"
K=${K:-128}

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 \
        chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY chrM)
CHR=${CHROMS[${SLURM_ARRAY_TASK_ID:?run as an array job (sbatch --array=...)}]}

SRC="$GBZDIR/$CHR.v2.gbz"
DST="$OUT/$CHR.lay.tsv"

mkdir -p "$OUT"
if [ ! -s "$SRC" ]; then echo "$CHR: no GBZ at $SRC -- run the export array first"; exit 1; fi
if [ -s "$DST" ]; then echo "$CHR: already present ($DST) -- nothing to do"; exit 0; fi
if [ ! -x "$BIN" ]; then echo "ERROR: binary missing: $BIN (run cluster_setup.sh)"; exit 1; fi

echo "host=$(hostname)  chr=$CHR  K=$K  threads=${SLURM_CPUS_PER_TASK:-1}  start=$(date -Iseconds)"

# -o takes a PREFIX and appends .lay.tsv / .links.tsv, so the temp is a prefix too.
TMP="$OUT/.$CHR.partial.${SLURM_ARRAY_JOB_ID:-$$}"
/usr/bin/time -v "$BIN" "$SRC" -o "$TMP" \
    --path-batch "$K" \
    --threads "${SLURM_CPUS_PER_TASK:-1}" \
    --emit-links

mv -f "$TMP.lay.tsv"   "$DST"                      # atomic: DST appears only when complete
mv -f "$TMP.links.tsv" "$OUT/$CHR.links.tsv"
echo "chr=$CHR done=$(date -Iseconds) size=$(stat -c%s "$DST") bytes"
