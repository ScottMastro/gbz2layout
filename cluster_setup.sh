#!/bin/bash
# Bring the cluster checkout up to date and rebuild, then submit the layout array.
#
#   bash cluster_setup.sh [partition] [account]
#
# Two things this exists to handle:
#
# 1. The old `hierarchical-bubble-layout` branch was merged to main and DELETED
#    from origin. A checkout still sitting on it can't `git pull` -- its upstream
#    is gone. This moves to main first.
#
# 2. `make` will not relink when only the libraries changed, and `clean` used to
#    miss gbz2layout-nocuda entirely, so a rebuilt .a could silently fail to reach
#    the binary. That cost two rounds of SIGILL debugging. This forces the relink.
#
# -march=native is the other trap: it bakes login-node instructions into the
# binary and compute nodes die with SIGILL (signal 4). Everything here builds with
# a portable baseline. sdsl's Make.helper propagates flags to gbwt/gbwtgraph, so
# those must be rebuilt with it too -- fix_build.sh does that.
set -euo pipefail

PARTITION="${1:-}"
ACCOUNT="${2:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

echo "=== 1. move to main (the old branch is gone from origin) ==="
git fetch origin --prune
git checkout main 2>/dev/null || git checkout -b main origin/main
git reset --hard origin/main
git log --oneline -3

echo
echo "=== 2. rebuild with a portable ISA ==="
if [ -x "$REPO/fix_build.sh" ]; then
    bash "$REPO/fix_build.sh"
else
    echo "fix_build.sh missing (it was never committed -- it's a local cluster script)."
    echo "Minimum needed:"
    echo "  export PREFIX=$REPO/local"
    echo "  cd $REPO/deps/gbwt      && make clean && make -j4 && cp lib/libgbwt.a      \$PREFIX/lib/"
    echo "  cd $REPO/deps/gbwtgraph && make clean && make -j4 && cp lib/libgbwtgraph.a \$PREFIX/lib/"
    echo "  cd $REPO && rm -f build/gbz2layout-nocuda && make tool-nocuda ARCH=-msse4.2"
    exit 1
fi

echo
echo "=== 3. preflight ==="
BIN="$REPO/build/gbz2layout-nocuda"
[ -x "$BIN" ] || { echo "ERROR: no binary at $BIN"; exit 1; }
NG=$(find "$REPO/hprc-v2-gbz" -maxdepth 1 -name '*.v2.gbz' 2>/dev/null | wc -l || true)
echo "per-chr GBZs available: ${NG:-0}/25"
[ "${NG:-0}" -gt 0 ] || { echo "ERROR: no GBZs in $REPO/hprc-v2-gbz -- run the export array first"; exit 1; }
ND=$(find "$REPO/hprc-v2-layout" -maxdepth 1 -name '*.lay.tsv' 2>/dev/null | wc -l || true)
echo "layouts already done: ${ND:-0}/25 (those tasks exit early)"

# The binary must be newer than every lib it statically links, else it's a stale
# link and the rebuild never reached it.
for lib in libsdsl.a libgbwt.a libgbwtgraph.a; do
    L="$REPO/local/lib/$lib"
    if [ -f "$L" ] && [ "$L" -nt "$BIN" ]; then
        echo "ERROR: $lib is newer than the binary -- stale link. Re-run fix_build.sh."
        exit 1
    fi
done
echo "ok: binary is newer than its libs (fresh link)"

echo
echo "=== 4. submit ==="
SB="--export=ALL"
if [ -n "$PARTITION" ]; then SB="$SB --partition=$PARTITION"; fi
if [ -n "$ACCOUNT" ];   then SB="$SB --account=$ACCOUNT"; fi
echo "sbatch $SB $REPO/layout_array.slurm"
# shellcheck disable=SC2086
sbatch $SB "$REPO/layout_array.slurm"

echo
echo "watch it:"
echo "  sq"
echo "  ls -la $REPO/hprc-v2-layout/"
echo "  tail -f slurm-gbz-layout-*.err"
echo
echo "resubmit to resume (finished chromosomes exit early):"
echo "  bash cluster_setup.sh $PARTITION $ACCOUNT"
