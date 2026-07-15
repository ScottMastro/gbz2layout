#!/usr/bin/env bash
set -e
cd /home/scott/projects/gbz2layout
make tool 2>&1 | grep -iE 'error' && exit 1 || true
GBZ=/mnt/data/pangyplot/bench/chrY.v2.gbz
O=/mnt/data/pangyplot/bench/out
echo "==================== COMPARTMENTS (target 128 tasks) ===================="
/usr/bin/time -v build/gbz2layout "$GBZ" -o "$O/chrY_comp" \
    --cap 30 --iter 30 --threads 8 --emit-links --compartments 128 2>&1 \
    | grep -E '\[gbz2layout\]|\[compartments\]|Maximum resident|Elapsed'
echo "--- fidelity ---"; build/eval_layout "$GBZ" "$O/chrY_comp.lay.tsv" 2>&1 | tail -4
echo "--- legibility ---"; build/eval_crossings "$O/chrY_comp.lay.tsv" "$O/chrY_comp.links.tsv" 2>&1 | tail -3
echo "==================== DONE ===================="
