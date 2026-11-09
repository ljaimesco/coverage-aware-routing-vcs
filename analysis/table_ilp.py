"""Recompute Table II (coverage-aware routing versus the exact ILP) from raw results.

Usage (from the repo root):  python analysis/table_ilp.py results [stamp]
Reads ilp_small_{london,monaco}_<stamp>_*/results.csv and prints, per network and selection rule:
the number of instances, optimal selections, F/F_opt and the ratio of reductions in J relative to
shortest routes (mean and worst), and median selection times. Instances with F_opt = 0 (no vehicle has
a candidate to select, so every rule gives the same outcome) are excluded from both ratios; instances whose
optimal reduction in J is at most 1e-6 J_shortest are also excluded from the J ratio. A selection counts as
optimal if its F is at least (1 - 1e-9) F_opt; the ILP is solved to a zero relative MIP gap.
"""
import glob
import os
import sys

import pandas as pd

res = sys.argv[1]
stamp = sys.argv[2] if len(sys.argv) > 2 else "20261108"
for net in ["london", "monaco"]:
    r = pd.read_csv(glob.glob(os.path.join(res, f"ilp_small_{net}_{stamp}_*", "results.csv"))[0])
    key = ["scenario", "R", "dev_rule"]
    F = r.pivot_table(index=key, columns="sel_rule", values="F_gain")
    J = r.pivot_table(index=key, columns="sel_rule", values="J")
    J0 = r.groupby(key).J_shortest.first()
    keepF = F["ilp"] > 0
    keepJ = keepF & ((J0 - J["ilp"]) > 1e-6 * J0)
    print(f"== {net}: {len(F)} instances; F ratio over {int(keepF.sum())}, J ratio over {int(keepJ.sum())}")
    for s in ["greedy_ls", "greedy", "closest", "random"]:
        fr = (F[s] / F["ilp"])[keepF]
        jr = ((J0 - J[s]) / (J0 - J["ilp"]))[keepJ]
        print(f"  {s:9s}  optimal {int((F[s][keepJ] >= F['ilp'][keepJ] * (1 - 1e-9)).sum()):2d}  F/Fopt mean {fr.mean():.3f} worst {fr.min():.3f}   "
              f"J-reduction ratio mean {jr.mean():.3f} worst {jr.min():.3f}   above ILP: {int((F[s] > F['ilp'] * (1 + 1e-9)).sum())}")
    t = r.groupby("sel_rule").t_select_s.median()
    print(f"  median selection time: greedy_ls {1000 * t['greedy_ls']:.1f} ms, ILP {t['ilp']:.2f} s")
