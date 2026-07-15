#!/usr/bin/env python3
"""Render an odgi-style .lay.tsv as a PNG (node segments) for visual QC.

Each node = a short segment between its two endpoints (idx 2r, 2r+1).
Usage: render_layout.py <lay.tsv> <out.png> [title]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load(path):
    # columns: idx X Y component
    arr = np.loadtxt(path, skiprows=1, usecols=(0, 1, 2))
    idx = arr[:, 0].astype(np.int64)
    X = arr[:, 1]
    Y = arr[:, 2]
    n = idx.max() + 1
    xs = np.full(n, np.nan); ys = np.full(n, np.nan)
    xs[idx] = X; ys[idx] = Y
    n_nodes = n // 2
    x0 = xs[0:2 * n_nodes:2]; y0 = ys[0:2 * n_nodes:2]
    x1 = xs[1:2 * n_nodes:2]; y1 = ys[1:2 * n_nodes:2]
    return x0, y0, x1, y1

def main():
    if len(sys.argv) < 3:
        print("usage: render_layout.py <lay.tsv> <out.png> [title]"); sys.exit(1)
    tsv, png = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else tsv
    x0, y0, x1, y1 = load(tsv)
    segs = np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y1])], axis=1)
    ok = np.isfinite(segs).all(axis=(1, 2))
    segs = segs[ok]

    fig, ax = plt.subplots(figsize=(20, 6), dpi=110)
    lc = LineCollection(segs, linewidths=0.15, colors="#1f5fa8", alpha=0.35)
    ax.add_collection(lc)
    allx = np.concatenate([x0[np.isfinite(x0)], x1[np.isfinite(x1)]])
    ally = np.concatenate([y0[np.isfinite(y0)], y1[np.isfinite(y1)]])
    ax.set_xlim(allx.min(), allx.max())
    ax.set_ylim(ally.min(), ally.max())
    ax.set_title(f"{title}  ({len(segs)} nodes)")
    # auto aspect: stretch Y to reveal bubble structure (ribbons are ~40:1)
    fig.tight_layout()
    fig.savefig(png)
    print(f"wrote {png}  x:[{allx.min():.0f},{allx.max():.0f}] y:[{ally.min():.0f},{ally.max():.0f}]")

if __name__ == "__main__":
    main()
