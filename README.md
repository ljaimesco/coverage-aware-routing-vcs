# Coverage-aware routing for location-diversity vehicular crowdsensing

Code, configurations and raw results for the manuscript

> L. G. Jaimes and A. Dina, "Coverage-Aware Routing with Approximate Equilibrium Guarantees for
> Location-Diversity Vehicular Crowdsensing," submitted to *IEEE Open Journal of Intelligent
> Transportation Systems*.

The mechanism has two stages. Stage 1 is a platform-centric game that sets each vehicle's deviation
target from location-diversity weights. Stage 2 selects one route per vehicle among candidate via-routes
to reduce the coverage penalty J (greedy selection plus local search, with a 1/2 approximation
guarantee on the coverage gain). The Theorem 2 certificate checks whether the realized routes form an
approximate equilibrium. Routes are computed on SUMO network files with `sumolib`; TraCI is not needed.

| Paper | Module |
|---|---|
| Sec. III, diversity weights (with self term) | `jcov/mechanism.py: diversity_weights` |
| Sec. IV, Stage 1 equilibrium targets (Supplement S1) | `jcov/mechanism.py: stage1_targets` |
| Sec. V-A, candidate via-routes, tolerance filter, Frechet pruning, fallback | `jcov/candidates.py` |
| Sec. V-B, coverage penalty J and coverage gain F | `jcov/coverage.py` |
| Sec. V-C, Algorithm 1 (greedy plus local search), exact ILP (Supplement S3), baselines | `jcov/router.py` |
| Sec. VI, Theorem 2 certificate and best-response gains | `jcov/mechanism.py: certificate, best_response_gain` |

## Setup

Tested with Python 3.13 and SUMO 1.28.0 on Linux.

```bash
python3 -m venv ~/venvs/carvcs && source ~/venvs/carvcs/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
bash get_networks.sh        # downloads the Monaco network (needed only for Monaco configs)
```

## Networks

* **London**: `networks/london-seg4.net.xml.gz`, a SUMO network built from OpenStreetMap data
  (© OpenStreetMap contributors, available under the Open Database License, ODbL). `sumolib` reads
  the gzipped file directly.
* **Monaco**: `most.net.xml` from the MoST scenario (Codeca and Harri, 2017; GPL-3.0), pinned to commit
  `b29b2f65f1096a9c69a601ec62a724815cb4a43f`. It is downloaded by `get_networks.sh` and not tracked here.
  Only the road graph is used, not the MoST traffic demand.

## Reproducing the paper

`results/` holds the raw outputs behind every table and figure (one folder per configuration).

```bash
python analysis/table_ilp.py results             # Table II: Algorithm 1 versus the exact ILP
python analysis/paper_figures.py results figs    # Figures of Section VII
```

To rerun the experiments (writes new timestamped folders under `results/`):

```bash
python run_experiment.py configs/ilp_small_london.json                  # about 1 minute
JOBS=4 nohup bash run_all.sh > run_all.log 2>&1 &                         # all 34 configs, about 5 to 6 h
python run_experiment.py configs/X.json --resume results/X_<timestamp>   # resume an interrupted run
```

Runs are deterministic given the seeds in each config: rerunning `ilp_small_london` reproduces the
stored J and F values exactly. The `config.json` files inside `results/` record the paths used when the
runs were made (an earlier layout with a `journal/` prefix); the configs in `configs/` are otherwise identical.

| Configs | Purpose |
|---|---|
| `ilp_small_{london,monaco}` | Algorithm 1 versus the exact ILP (HiGHS, zero optimality gap) |
| `main_{london,monaco}_{equal,hetero}` | 2 x 2 ablation (deviation rule x selection rule), equal and lognormal costs |
| `detour_london_hetero_D15`, `_D05` | detour-limit sweep (D = 1.5 L and 0.5 L) |
| `place_*` | balanced and dispersed vehicle placement |
| `res_*_{50,100,200}` | grid resolution, with every route set re-scored on all three grids |
| `sens_*_base`, `sens_*_noself` | effect of the self term in the diversity weights |
| `sens_london_*` | one-at-a-time sensitivity to rho, delta, K and eta, plus a 2 x 2 delta-eta check |

## Output files

Each run writes `results/<config>_<timestamp>/` with:

* `results.csv`: one row per scenario, budget R, deviation rule (`ldd`, `uniform`) and selection rule
  (`greedy_ls` is Algorithm 1; also `greedy`, `ilp`, `closest`, `random`). Columns include the coverage
  penalty J and its reduction against shortest routes, the coverage gain `F_gain`, added travel,
  unrealized vehicles, realization gaps, payments and utilities, the Theorem 2 certificate and the
  best-response gains, and runtimes.
* `vehicles.csv`: per-vehicle target, realized deviation, gap, diversity weight, cost and utility.
* `config.json`: the configuration, seeds, git commit and package versions.

## Design notes

* **Paired scenarios.** Each scenario is drawn once from `seed + scenario` and reused for every budget and
  method, so differences between methods are not due to different trips.
* **Uniform deviation** gives every vehicle the mean LDD target, so total deviation is equal.
* **Detour limit** `D_i = detour_limit_factor * L_i`. Vehicles whose target has no candidate get the
  fallback route and are counted as unrealized.
* **Coverage domain.** Grid cells whose center lies within 500 m of a passenger-car road.
* **ILP** runs only when (#candidates x #cells) <= `ilp_max_vars`.

## License

Code: MIT (see `LICENSE`). The London network file is derived from OpenStreetMap data and is
distributed under the ODbL.
