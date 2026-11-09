"""Run the journal experiments: paired scenarios x budgets x (deviation rule x selection rule).

Usage (from the repo root, inside the project virtualenv):
    python run_experiment.py configs/pilot_london.json
    python run_experiment.py configs/X.json --resume results/X_<timestamp>
With --resume, scenarios already complete in that folder are kept, rows of a partially written
scenario are dropped, and only the remaining scenarios are run (each scenario has its own seed,
so the output is the same as an uninterrupted run).
Outputs go to results/<config name>_<timestamp>/:
    results.csv   one row per scenario, budget, deviation rule and selection rule
    vehicles.csv  one row per vehicle for the same combinations
    config.json   the configuration used, plus seeds, git commit and package versions
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jcov.candidates import build_candidates, prepare_vehicle  # noqa: E402
from jcov.coverage import CoverageGrid  # noqa: E402
from jcov.geometry import resample  # noqa: E402
from jcov.mechanism import (best_response_gain, certificate, compensation_needed,  # noqa: E402
                            diversity_weights, payments_utilities, stage1_targets)
from jcov.network import RoadNetwork  # noqa: E402
from jcov import router  # noqa: E402
from jcov.scenario import draw_costs, make_scenario  # noqa: E402

REPO = os.path.abspath(os.path.dirname(__file__))


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _completed_scenarios(out, cfg):
    """Scenarios are written in order, so every scenario before the last one present is complete;
    the last one counts only if it has all its rows. Rewrite both CSVs without the rest."""
    import pandas as pd
    res = pd.read_csv(os.path.join(out, "results.csv"))
    veh = pd.read_csv(os.path.join(out, "vehicles.csv"))
    if res.empty:
        return set()
    n = res.groupby("scenario").size()
    last = int(n.index.max())
    full = {int(x) for x in n.index if x < last}
    n_sel = len([r for r in cfg["selection_rules"] if r != "ilp"])
    expected = len(cfg["budgets"]) * len(cfg["deviation_rules"]) * n_sel
    if "ilp" not in cfg["selection_rules"] and n.loc[last] == expected:
        full.add(last)
    res[res.scenario.isin(full)].to_csv(os.path.join(out, "results.csv"), index=False)
    veh[veh.scenario.isin(full)].to_csv(os.path.join(out, "vehicles.csv"), index=False)
    return full


def run(cfg_path, resume=None):
    cfg = json.load(open(cfg_path))
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    done = set()
    if resume:
        out = os.path.abspath(resume)
        done = _completed_scenarios(out, cfg)
        print(f"resuming {out}: keeping scenarios {sorted(done)}")
    else:
        out = os.path.join(REPO, "results", f"{name}_{dt.datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(out)
    meta = dict(cfg, config_file=cfg_path, git_commit=git_commit(), started=dt.datetime.now().isoformat(),
                numpy=np.__version__, resumed_keeping=sorted(done))
    json.dump(meta, open(os.path.join(out, "config.json"), "w"), indent=2)

    net = RoadNetwork.from_sumo(os.path.join(REPO, cfg["network"]))
    dom = cfg.get("domain", {"type": "bbox"})
    if dom["type"] == "road_buffer":
        rp = net.road_points(25.0)
        make_grid = lambda cell: CoverageGrid(net.bbox, cell, road_points=rp, buffer_m=dom["buffer_m"])
    elif dom["type"] == "bbox":
        make_grid = lambda cell: CoverageGrid(net.bbox, cell)
    else:
        raise ValueError(dom)
    grid = make_grid(cfg["grid_cell_m"])
    # extra grids on which every selected route set is re-scored (measurement only, Sec. VII)
    eval_grids = {int(cell): make_grid(cell) for cell in cfg.get("eval_grid_cells", [])}
    print(f"network: {len(net.nodes)} junctions; grid: {len(grid.area)} of {grid.n_box_cells} box cells "
          f"of {cfg['grid_cell_m']} m (domain {dom}); eval grids {list(eval_grids)}")
    meta.update(n_cells=len(grid.area), n_box_cells=grid.n_box_cells,
                eval_grid_n_cells={k: len(g.area) for k, g in eval_grids.items()})
    json.dump(meta, open(os.path.join(out, "config.json"), "w"), indent=2)

    mode = "a" if resume else "w"
    res_f = open(os.path.join(out, "results.csv"), mode, newline="")
    veh_f = open(os.path.join(out, "vehicles.csv"), mode, newline="")
    res_w, veh_w = None, None
    if resume:                                   # reuse the existing headers
        import pandas as pd
        res_w = csv.DictWriter(res_f, fieldnames=list(pd.read_csv(os.path.join(out, "results.csv"), nrows=0).columns), restval="")
        veh_w = csv.DictWriter(veh_f, fieldnames=list(pd.read_csv(os.path.join(out, "vehicles.csv"), nrows=0).columns))
    sp = cfg["stage2"]

    for sc in range(cfg["n_scenarios"]):
        if sc in done:
            continue
        seed = cfg["seed"] + sc
        rng = np.random.default_rng(seed)
        ods = make_scenario(net, cfg["groups"], cfg["od_radius_m"], cfg["min_od_m"], rng,
                            max_od=cfg.get("max_od_m"))
        M = len(ods)
        vrs = [prepare_vehicle(net, s, d) for s, d in ods]
        L = np.array([v.L for v in vrs])
        D = cfg["detour_limit_factor"] * L
        c = draw_costs(M, cfg["costs"], rng)
        def_polys = [net.route_polyline(v.default) for v in vrs]
        x, _ = diversity_weights([resample(p, sp["resample_m"]) for p in def_polys],
                                 rho=cfg["rho_m"], sigma=cfg["sigma"], self_term=cfg["self_term"])
        phi_default = [grid.phi(p) for p in def_polys]
        J_short = grid.penalty(phi_default)
        J_short_eval = {k: g.penalty([g.phi(p) for p in def_polys]) for k, g in eval_grids.items()}

        for R in cfg["budgets"]:
            t0 = time.perf_counter()
            s1 = stage1_targets(c, x, R)
            t_stage1 = time.perf_counter() - t0
            dev_rules = {"ldd": s1["t"], "uniform": np.full(M, s1["t"].mean())}
            for dev_name in cfg["deviation_rules"]:
                targets = dev_rules[dev_name]
                # candidate sets (shared by every selection rule)
                t0 = time.perf_counter()
                cand, phis, gaps, fixed_route, n_cand = {}, {}, {}, {}, np.zeros(M, int)
                unrealized = np.zeros(M, bool)
                for i in range(M):
                    if targets[i] <= 0:
                        fixed_route[i] = (vrs[i].default, 0.0)
                        continue
                    cs, fb, _ = build_candidates(net, vrs[i], targets[i], D[i], sp["delta_m"],
                                                 sp["K"], sp["eta_m"], sp["resample_m"])
                    n_cand[i] = len(cs)
                    if not cs:
                        fixed_route[i] = fb
                        unrealized[i] = True
                        continue
                    cand[i] = cs
                    phis[i] = np.vstack([grid.phi(net.route_polyline(r)) for r, _ in cs])
                    gaps[i] = np.array([abs(dv - targets[i]) for _, dv in cs])
                t_cand = time.perf_counter() - t0
                phi_fixed = [grid.phi(net.route_polyline(r)) for r, _ in fixed_route.values()]

                for sel in cfg["selection_rules"]:
                    t0 = time.perf_counter()
                    if sel == "closest":
                        choice = router.closest_length(cand, gaps)
                    elif sel == "random":
                        choice = router.random_choice(cand, np.random.default_rng(seed * 1000 + 7))
                    elif sel == "greedy":
                        choice = router.greedy(cand, phis, gaps, phi_fixed, grid)
                    elif sel == "greedy_ls":
                        choice = router.greedy_ls(cand, phis, gaps, phi_fixed, grid)
                    elif sel == "ilp":
                        n_pairs = sum(len(v) for v in cand.values())
                        if n_pairs * len(grid.area) > cfg.get("ilp_max_vars", 2e5):
                            continue
                        choice, _, ilp_status = router.ilp(cand, phis, phi_fixed, grid,
                                                           mip_rel_gap=cfg.get("ilp_mip_rel_gap", 0.0))
                        if choice is None or ilp_status != 0:   # 0: solved to the requested gap
                            print(f"  ILP skipped: status {ilp_status}")
                            continue
                    else:
                        raise ValueError(sel)
                    t_sel = time.perf_counter() - t0
                    t_real = np.zeros(M)
                    all_phi = list(phi_fixed)
                    sel_polys = [net.route_polyline(r) for r, _ in fixed_route.values()]
                    for i, (r, dv) in fixed_route.items():
                        t_real[i] = dv
                    for i, rix in choice.items():
                        t_real[i] = cand[i][rix][1]
                        all_phi.append(phis[i][rix])
                        sel_polys.append(net.route_polyline(cand[i][rix][0]))
                    J = grid.penalty(all_phi)
                    P, u = payments_utilities(t_real, c, x, R)
                    row = dict(scenario=sc, seed=seed, M=M, R=R, dev_rule=dev_name, sel_rule=sel,
                               J=J, J_shortest=J_short, J_reduction_pct=100 * (J_short - J) / J_short,
                               F_gain=router.coverage_gain(choice, phis, phi_fixed, grid),
                               added_travel_m=float(t_real.sum()), added_travel_pct=100 * t_real.sum() / L.sum(),
                               n_active=int((targets > 0).sum()), n_unrealized=int(unrealized.sum()),
                               mean_candidates=float(n_cand[targets > 0].mean()) if (targets > 0).any() else 0.0,
                               util_mean=float(u.mean()), util_min=float(u.min()), n_negative_util=int((u < -1e-9).sum()),
                               compensation=compensation_needed(u), compensation_pct_R=100 * compensation_needed(u) / R,
                               t_stage1_s=t_stage1, t_candidates_s=t_cand, t_select_s=t_sel)
                    for k, g in eval_grids.items():
                        Je = g.penalty([g.phi(p) for p in sel_polys])
                        row[f"J_eval{k}"] = Je
                        row[f"J_reduction_eval{k}_pct"] = 100 * (J_short_eval[k] - Je) / J_short_eval[k]
                    row.update(x_max_over_min=float(x.max() / x.min()), x_sum=float(x.sum()))
                    gaps_all = np.abs(t_real - targets)
                    row.update(max_gap_m=float(gaps_all.max()), mean_gap_active_m=float(gaps_all[targets > 0].mean()) if (targets > 0).any() else 0.0)
                    if dev_name == "ldd":                      # equilibrium analysis only for LDD targets
                        cert = certificate(t_real, s1, x, c, R)
                        br = best_response_gain(t_real, c, x, R)
                        act = s1["active"]
                        row.update(S_star=s1["S_star"], Delta=cert["Delta"], gap_condition=bool(cert["condition"]),
                                   eps_max_active=float(np.nanmax(cert["eps"][act])) if cert["condition"] else np.nan,
                                   eps_max_inactive=float(np.nanmax(cert["eps"][~act])) if cert["condition"] and (~act).any() else np.nan,
                                   max_br_gain_active=float(br[act].max()), max_br_gain_inactive=float(br[~act].max()) if (~act).any() else 0.0,
                                   ir_violations=int(np.nansum(1 - cert["ir_ok"][act])) if cert["condition"] else -1,
                                   n_active_second_order=int(np.isfinite(cert["eps2"][act]).sum()),
                                   eps2_max_active=float(np.nanmax(cert["eps2"][act])) if np.isfinite(cert["eps2"][act]).any() else np.nan)
                    if res_w is None:
                        keys = list(row) + [k for k in ["S_star", "Delta", "gap_condition", "eps_max_active",
                                                        "eps_max_inactive", "max_br_gain_active",
                                                        "max_br_gain_inactive", "ir_violations",
                                                        "n_active_second_order", "eps2_max_active"] if k not in row]
                        res_w = csv.DictWriter(res_f, fieldnames=keys, restval="")
                        res_w.writeheader()
                    res_w.writerow(row)
                    for i in range(M):
                        vrow = dict(scenario=sc, R=R, dev_rule=dev_name, sel_rule=sel, vehicle=i, L_m=L[i],
                                    D_m=D[i], x=x[i], c=c[i], target_m=targets[i], realized_m=t_real[i],
                                    gap_m=abs(t_real[i] - targets[i]), active_stage1=bool(s1["active"][i]),
                                    unrealized=bool(unrealized[i]), n_candidates=int(n_cand[i]),
                                    payment=P[i], utility=u[i])
                        if veh_w is None:
                            veh_w = csv.DictWriter(veh_f, fieldnames=list(vrow))
                            veh_w.writeheader()
                        veh_w.writerow(vrow)
                    res_f.flush(); veh_f.flush()
                    print(f"sc={sc} R={R:g} {dev_name:7s} {sel:8s} J_red={row['J_reduction_pct']:6.2f}% "
                          f"travel=+{row['added_travel_pct']:5.1f}% unreal={row['n_unrealized']} "
                          f"maxgap={row['max_gap_m']:.1f} m")
    res_f.close(); veh_f.close()
    print("results in", out)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], resume=a[a.index("--resume") + 1] if "--resume" in a else None)
