#!/usr/bin/env python3
"""Quantify whether two layouts represent bubble structure the same way.

Per node: perpendicular distance to the reference spine (2D dist to nearest
reference node). Bin by TRUE genome bp (arc-length along the reference, from
GFA node lengths) so the two layouts' bins align. Normalize each layout's
distances to bp by its own isometry scale (reference arc-length / bp), so
magnitudes compare. Then correlate the bubble-depth-vs-genome profile.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

def load_layout(path):
    a=np.loadtxt(path,skiprows=1,usecols=(0,1,2)); idx=a[:,0].astype(np.int64)
    n=idx.max()+1; xs=np.full(n,np.nan); ys=np.full(n,np.nan); xs[idx]=a[:,1]; ys[idx]=a[:,2]
    nn=n//2
    return 0.5*(xs[0:2*nn:2]+xs[1:2*nn:2]), 0.5*(ys[0:2*nn:2]+ys[1:2*nn:2])

def golden_ref(gfa):
    seglen={}; path=None
    with open(gfa) as f:
        for line in f:
            if line[0]=='S':
                t=line.split('\t'); seglen[int(t[1])]=len(t[2].rstrip())
            elif line[0]=='P' and line.split('\t',2)[1]=='GRCh38#chrY':
                path=[int(x[:-1]) for x in line.split('\t')[2].split(',')]
    ranks=[]; bps=[]; bp=0
    for sid in path:
        ranks.append(sid-1); bps.append(bp); bp+=seglen.get(sid,0)
    return np.array(ranks), np.array(bps,dtype=float)   # ref ranks (path order), their true bp

def ours_ref(meta, cx):
    m=np.loadtxt(meta,skiprows=1,dtype=np.int64); rr=m[m[:,1]==1,0]
    # ours ref-init sets X = true bp, so ref node X IS its bp; order by bp
    o=np.argsort(cx[rr]); return rr[o], cx[rr][o]

def analyze(cx,cy,ref_ranks,ref_bp,label):
    ok=np.isfinite(cx)&np.isfinite(cy)
    good=np.isfinite(cx[ref_ranks])&np.isfinite(cy[ref_ranks])
    rr=ref_ranks[good]; rbp=ref_bp[good]
    rx,ry=cx[rr],cy[rr]
    tree=cKDTree(np.column_stack([rx,ry]))
    d,nn=tree.query(np.column_stack([cx[ok],cy[ok]]),k=1)
    node_bp=rbp[nn]                              # true genome bp via nearest ref node
    # isometry scale: reference 2D arc-length (in bp order) / total bp
    seg=np.hypot(np.diff(rx),np.diff(ry)); arclen=np.nansum(seg)
    totbp=rbp.max()-rbp.min(); scale=arclen/totbp
    dbp=d/scale                                  # spine distance in bp units
    print(f"[{label}] nodes={ok.sum()} refnodes={len(rr)}  ref arc-len={arclen/1e6:.1f}Mb  bp-span={totbp/1e6:.1f}Mb  isometry-scale={scale:.2f}")
    print(f"[{label}] spine-distance in bp: median={np.median(dbp):.0f} p90={np.percentile(dbp,90):.0f} p99={np.percentile(dbp,99):.0f} max={dbp.max():.0f}")
    return node_bp, dbp

def profile(bp,d,binbp,stat):
    b=(bp/binbp).astype(np.int64); nb=b.max()+1; out=np.full(nb,np.nan)
    o=np.argsort(b); bs=b[o]; ds=d[o]; st=np.searchsorted(bs,np.arange(nb)); st=np.append(st,len(bs))
    for i in range(nb):
        if st[i+1]>st[i]: out[i]=stat(ds[st[i]:st[i+1]])
    return out

GL="/home/scott/projects/pangyplot/data/chrY/chrY.lay.tsv"
GG="/home/scott/projects/pangyplot/data/chrY/chrY.gfa"
OL="/mnt/data/pangyplot/bench/out/chrY_v1_m.lay.tsv"
OM="/mnt/data/pangyplot/bench/out/chrY_v1_m.meta.tsv"

gcx,gcy=load_layout(GL); grr,grbp=golden_ref(GG)
ocx,ocy=load_layout(OL); orr,orbp=ours_ref(OM,ocx)

gbp,gd=analyze(gcx,gcy,grr,grbp,"golden")
obp,od=analyze(ocx,ocy,orr,orbp,"ours")

BIN=20000.0
for stat,name in [(lambda v:np.percentile(v,90),"p90"),(np.mean,"mean")]:
    gp=profile(gbp,gd,BIN,stat); op=profile(obp,od,BIN,stat)
    nb=min(len(gp),len(op)); gp=gp[:nb]; op=op[:nb]; m=np.isfinite(gp)&np.isfinite(op)
    content=m&((gp>np.nanpercentile(gp[m],50))|(op>np.nanpercentile(op[m],50)))
    print(f"[{name} @ {int(BIN)}bp true-bp bins, {m.sum()} bins] "
          f"ALL Pearson={pearsonr(gp[m],op[m])[0]:.3f} Spearman={spearmanr(gp[m],op[m])[0]:.3f} | "
          f"BUBBLE bins({content.sum()}) Pearson={pearsonr(gp[content],op[content])[0]:.3f} Spearman={spearmanr(gp[content],op[content])[0]:.3f} | "
          f"ours/golden(sum)={np.nansum(op[m])/np.nansum(gp[m]):.2f}")
