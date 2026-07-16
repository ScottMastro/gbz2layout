#!/bin/bash
#SBATCH --job-name=gbz-export
#SBATCH --output=slurm-%x-%a-%j.out
#SBATCH --error=slurm-%x-%a-%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G                  # chr1 export peaks ~46 GB; smaller chrs waste some but still run
#SBATCH --cpus-per-task=4
#SBATCH --array=0-24               # 25 tasks, one per chromosome (see CHROMS below); add %N to throttle
# #SBATCH --partition=<fill in>
# #SBATCH --account=<fill in>

# Parallel per-chromosome GBZ extraction: one SLURM array task per chromosome,
# each loading the whole-genome GBZ independently and writing OUT/<chr>.v2.gbz.
# Trades a redundant genome load per task for wall-clock = slowest chromosome
# (chr1) instead of the serial sum. skip-if-exists makes a resubmit resume.
#
# NOTE: no NAS step (not mounted on compute nodes). Results stay in OUT; pull
# them back to the laptop afterwards (scp/rsync from a login node).

module load gcc                    # matching libstdc++ at runtime for the prebuilt binary
set -euo pipefail

# ---- paths: EDIT or override via env (sbatch inherits the submit env) ----
REPO=${REPO:-$PWD}
GBZ=${GBZ:-$REPO/hprc-v2.0-mc-grch38.gbz}
OUT=${OUT:-$REPO/hprc-v2-gbz}
BIN="$REPO/build/gbz2layout-nocuda"

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 \
        chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY chrM)
CHR=${CHROMS[${SLURM_ARRAY_TASK_ID:?run as an array job (sbatch --array=...)}]}

mkdir -p "$OUT"
DST="$OUT/$CHR.v2.gbz"
# End early if this chromosome is already done. The write below is atomic
# (temp + rename), so DST only ever exists when the export finished — a task
# killed mid-write leaves only a .partial file and never a false "done".
if [ -s "$DST" ]; then echo "$CHR already present ($DST) — nothing to do, exiting"; exit 0; fi

echo "host=$(hostname)  chr=$CHR  start=$(date -Iseconds)"
TMP="$DST.partial.${SLURM_ARRAY_JOB_ID:-$$}"
/usr/bin/time -v "$BIN" "$GBZ" --chromosome "$CHR" --export-gbz "$TMP"
mv -f "$TMP" "$DST"
echo "chr=$CHR  done=$(date -Iseconds)  size=$(stat -c%s "$DST") bytes"
