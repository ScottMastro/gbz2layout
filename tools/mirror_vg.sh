#!/bin/bash
# Mirror per-chromosome .vg files from the HPRC-v2 S3 scratch path to the NAS.
# Idempotent: skips files already present at the correct size; resumable (curl -C -).
set -u
BASE="https://s3-us-west-2.amazonaws.com/human-pangenomics"
PREFIX="pangenomes/scratch/2025_02_28_minigraph_cactus/hprc-v2.0-mc-grch38/hprc-v2.0-mc-grch38.chroms"
NAS="/run/user/1000/gvfs/smb-share:server=yggdrasil.local,share=home/scott/public_datasets/hprc/v2"
LOG="$NAS/_mirror_vg.log"

# priority order: targets first, then the rest; chr22 already on NAS
CHROMS="chr1 chr9 chr2 chr3 chr4 chr6 chr7 chr5 chr8 chr10 chr11 chr12 chr15 chr14 \
chr16 chr13 chrX chr17 chr19 chr20 chr18 chr21 chrY chrEBV chrM chrOther"

echo "=== mirror start $(date) ===" >> "$LOG"
for c in $CHROMS; do
  url="$BASE/$PREFIX/$c.vg"
  dst="$NAS/$c.vg"
  remote=$(curl -sI "$url" | awk -F': ' 'tolower($1)=="content-length"{gsub(/\r/,"",$2);print $2}')
  if [ -z "$remote" ]; then echo "$(date) SKIP $c (no remote)" >> "$LOG"; continue; fi
  local=$(stat -c%s "$dst" 2>/dev/null || echo 0)
  if [ "$local" = "$remote" ]; then
    echo "$(date) OK   $c already complete ($remote bytes)" >> "$LOG"; continue
  fi
  echo "$(date) GET  $c ($(awk "BEGIN{printf \"%.1f\", $remote/1073741824}") GB) local=$local" >> "$LOG"
  # Resume until complete: a dropped connection leaves a partial, so re-issue
  # curl -C - (byte-range resume) until the local size matches remote, capping
  # attempts so a genuinely-gone remote can't spin forever.
  attempt=0
  final=$(stat -c%s "$dst" 2>/dev/null || echo 0)
  while [ "$final" != "$remote" ] && [ "$attempt" -lt 20 ]; do
    attempt=$((attempt+1))
    curl -sS -C - --retry 5 --retry-delay 10 -o "$dst" "$url" >> "$LOG" 2>&1
    final=$(stat -c%s "$dst" 2>/dev/null || echo 0)
    [ "$final" = "$remote" ] && break
    echo "$(date) RESUME $c attempt=$attempt ($final/$remote)" >> "$LOG"
    sleep 5
  done
  if [ "$final" = "$remote" ]; then echo "$(date) DONE $c" >> "$LOG"
  else echo "$(date) INCOMPLETE $c ($final/$remote) after $attempt attempts" >> "$LOG"; fi
done
echo "=== mirror finished $(date) ===" >> "$LOG"
