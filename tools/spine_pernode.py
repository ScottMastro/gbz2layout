#!/usr/bin/env python3
"""Per-node spine-distance correlation between golden (v1 odgi) and ours.

Node correspondence via original GFA segment names (vg chopped ours; aggregate
our pieces back to segments). Spine distance = 2D distance to nearest reference
node, computed independently in each layout. Correlation is scale-free
(Spearman), so it directly answers: does a node sitting deep in a bubble in the
golden also sit deep in ours?
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

def load_layout(path):
    a=np.loadtxt(path,skiprows=1,usecols=(0,1,2)); idx=a[:,0].astype(np.int64)
    n=idx.max()+1; xs=np.full(n,np.nan); ys=np.full(n,np.nan); xs[idx]=a[:,1]; ys[idx]=a[:,2]
    nn=n//2
    return 0.5*(xs[0:2*nn:2]+xs[1:2*nn:2]), 0.5*(ys[0:2*nn:2]+ys[1:2*nn:2])

def spine_dist(cx,cy,ref_mask):
    ok=np.isfinite(cx)&np.isfinite(cy)
    rr=np.where(ref_mask & ok)[0]
    tree=cKDTree(np.column_stack([cx[rr],cy[rr]]))
    d=np.full(len(cx),np.nan)
    d[ok],_=tree.query(np.column_stack([cx[ok],cy[ok]]),k=1)
    return d

# ours: layout + meta (is_ref, segment)
ocx,ocy=load_layout("/mnt/data/pangyplot/bench/out/chrY_v1_m.lay.tsv")
meta=np.genfromtxt("/mnt/data/pangyplot/bench/out/chrY_v1_m.meta.tsv",skip_header=1,dtype=None,encoding=None)
orank=np.array([m[0] for m in meta]); ois=np.array([m[1] for m in meta],dtype=bool)
oseg=np.array([str(m[2]) for m in meta])
oref=np.zeros(len(ocx),dtype=bool); oref[orank[ois]]=True
od=spine_dist(ocx,ocy,oref)
# aggregate ours per segment (mean and max over chopped pieces)
from collections import defaultdict
agg=defaultdict(list)
for r,s in zip(orank,oseg):
    if np.isfinite(od[r]): agg[s].append(od[r])
ours_mean={s:np.mean(v) for s,v in agg.items()}
ours_max ={s:np.max(v)  for s,v in agg.items()}

# golden: layout; ref nodes from GFA GRCh38 path; rank r <-> segment (r+1)
gcx,gcy=load_layout("/home/scott/projects/pangyplot/data/chrY/chrY.lay.tsv")
gref=np.zeros(len(gcx),dtype=bool)
path=None
with open("/home/scott/projects/pangyplot/data/chrY/chrY.gfa") as f:
    for line in f:
        if line[0]=='P' and line.split('\t',2)[1]=='GRCh38#chrY':
            path=[int(x[:-1]) for x in line.split('\t')[2].split(',')]; break
for sid in path:
    if sid-1 < len(gref): gref[sid-1]=True
gd=spine_dist(gcx,gcy,gref)
golden={str(r+1):gd[r] for r in range(len(gd)) if np.isfinite(gd[r])}   # segment name -> spine dist

# match segments present in both
common=sorted(set(golden)&set(ours_mean), key=lambda s:int(s) if s.isdigit() else 0)
g =np.array([golden[s]   for s in common])
om=np.array([ours_mean[s] for s in common])
ox=np.array([ours_max[s]  for s in common])
print(f"segments: golden={len(golden)} ours={len(ours_mean)} common={len(common)}")
print(f"golden spine-dist: median={np.median(g):.0f} p90={np.percentile(g,90):.0f}")
print(f"ours   spine-dist: median={np.median(om):.0f} p90={np.percentile(om,90):.0f}")
for lbl,ov in [("mean-agg",om),("max-agg",ox)]:
    pr=pearsonr(g,ov)[0]; sr=spearmanr(g,ov)[0]
    # log-space pearson (heavy tailed)
    lp=pearsonr(np.log1p(g),np.log1p(ov))[0]
    print(f"[per-node {lbl}]  Spearman={sr:.4f}  Pearson={pr:.4f}  Pearson(log)={lp:.4f}")
# restrict to bubble nodes (off-spine in golden): does the ranking hold among real bubbles?
bub=g>np.percentile(g,75)
print(f"[bubble nodes only, golden top-25% off-spine, n={bub.sum()}]  "
      f"Spearman={spearmanr(g[bub],om[bub])[0]:.4f}  Pearson(log)={pearsonr(np.log1p(g[bub]),np.log1p(om[bub]))[0]:.4f}")
