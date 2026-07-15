#!/usr/bin/env bash
set -e
cd /home/scott/projects/gbz2layout
GBZ=/mnt/data/pangyplot/bench/chrY.v2.gbz
OUT=/mnt/data/pangyplot/bench/out
mkdir -p "$OUT"

echo "==================== FLAT (baseline) ===================="
/usr/bin/time -v build/gbz2layout "$GBZ" -o "$OUT/chrY_flat" \
    --cap 30 --iter 30 --threads 8 --emit-links 2>&1 | grep -E '\[gbz2layout\]|Maximum resident|Elapsed'
echo "--- eval flat ---"
build/eval_layout "$GBZ" "$OUT/chrY_flat.lay.tsv" 2>&1 | tail -8

echo "==================== HIERARCHICAL ===================="
/usr/bin/time -v build/gbz2layout "$GBZ" -o "$OUT/chrY_hier" \
    --cap 30 --iter 30 --threads 8 --emit-links --hierarchical 2>&1 | grep -E '\[gbz2layout\]|Maximum resident|Elapsed'
echo "--- eval hier ---"
build/eval_layout "$GBZ" "$OUT/chrY_hier.lay.tsv" 2>&1 | tail -8
echo "==================== DONE ===================="
