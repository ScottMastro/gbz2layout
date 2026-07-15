#!/usr/bin/env python3
"""Zoom render of a layout WITH edges, so bubbles read as loops on the spine
rather than floating node-segments.

Usage: render_edges.py <lay.tsv> <links.tsv> <out.png> <xlo> <xhi> [ystretch] [refline]
  ystretch: if 'stretch', auto Y aspect; else equal aspect
  refline:  if 'ref', draw a red y=0 line (for pinned layouts)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

lay, links, png = sys.argv[1], sys.argv[2], sys.argv[3]
xlo, xhi = float(sys.argv[4]), float(sys.argv[5])
stretch = len(sys.argv) > 6 and sys.argv[6] == "stretch"
refline = len(sys.argv) > 7 and sys.argv[7] == "ref"

a = np.loadtxt(lay, skiprows=1, usecols=(0, 1, 2))
idx = a[:, 0].astype(np.int64); X = a[:, 1]; Y = a[:, 2]
n = idx.max() + 1
xs = np.full(n, np.nan); ys = np.full(n, np.nan); xs[idx] = X; ys[idx] = Y
nn = n // 2
x1 = xs[0:2*nn:2]; y1 = ys[0:2*nn:2]; x2 = xs[1:2*nn:2]; y2 = ys[1:2*nn:2]
cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)     # node centers

inwin = (cx >= xlo) & (cx <= xhi) & np.isfinite(cx)

# node segments
nsegs = np.stack([np.column_stack([x1, y1]), np.column_stack([x2, y2])], axis=1)[inwin]

# edges: a.end -> b.start, keep if either endpoint node is in the window
lk = np.loadtxt(links, skiprows=1, dtype=np.int64)
A = lk[:, 0]; B = lk[:, 1]
keep = (inwin[A] | inwin[B]) & (A < nn) & (B < nn)
A = A[keep]; B = B[keep]
esegs = np.stack([np.column_stack([x2[A], y2[A]]), np.column_stack([x1[B], y1[B]])], axis=1)
efin = np.isfinite(esegs).all(axis=(1, 2))
esegs = esegs[efin]

fig, ax = plt.subplots(figsize=(18, 8), dpi=120)
ax.add_collection(LineCollection(esegs, linewidths=0.35, colors="#888888", alpha=0.45))   # edges
ax.add_collection(LineCollection(nsegs, linewidths=1.0, colors="#1f5fa8", alpha=0.75))     # nodes
if refline:
    ax.axhline(0, color="#c0392b", lw=1.0, alpha=0.6)
yv = cy[inwin]; yv = yv[np.isfinite(yv)]
pad = (yv.max() - yv.min()) * 0.05 + 1
ax.set_xlim(xlo, xhi); ax.set_ylim(yv.min() - pad, yv.max() + pad)
if not stretch:
    ax.set_aspect("equal")
ax.set_title(f"{lay.split('/')[-1]}  X in [{xlo:.0f},{xhi:.0f}]  ({inwin.sum()} nodes, {len(esegs)} edges)")
fig.tight_layout(); fig.savefig(png)
print(f"wrote {png}: {int(inwin.sum())} nodes, {len(esegs)} edges")
