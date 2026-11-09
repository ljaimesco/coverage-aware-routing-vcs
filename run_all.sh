#!/usr/bin/env bash
# Run all journal experiments. From the repo root, inside the virtualenv:
#   source ~/venvs/spatialcov/bin/activate
#   bash get_networks.sh                       # once: downloads the MoST road network
#   JOBS=4 nohup bash run_all.sh > run_all.log 2>&1 &
# JOBS = number of configs run in parallel (default 1); use at most the number of physical cores.
# Rough single-core times: London 100 m configs ~25 to 90 min each, Monaco ~10 to 25 min,
# 50 m grid configs ~2 to 3 times longer. Whole set: about 19 h on one core, about 5 to 6 h with JOBS=4.
# Configs are ordered new-first; each writes results/<config>_<timestamp>/ and its own log.
set -e
cd "$(dirname "$0")"
JOBS=${JOBS:-1}
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
CONFIGS="
ilp_small_monaco main_monaco_hetero main_monaco_equal
place_monaco_balanced place_monaco_dispersed place_london_balanced place_london_dispersed
res_monaco_200 res_monaco_100 res_monaco_50 res_london_200 res_london_100 res_london_50
sens_monaco_base sens_monaco_noself sens_london_base sens_london_noself
sens_london_rho500 sens_london_rho2000 sens_london_delta50 sens_london_delta200
sens_london_K10 sens_london_K40 sens_london_eta50 sens_london_eta200
sens_london_delta50_eta50 sens_london_delta50_eta200 sens_london_delta200_eta50 sens_london_delta200_eta200
ilp_small_london main_london_equal main_london_hetero detour_london_hetero_D15 detour_london_hetero_D05
"
echo "=== start $(date), JOBS=$JOBS"
printf '%s\n' $CONFIGS | xargs -P "$JOBS" -I{} sh -c \
  'echo "--- {} start $(date +%T)"; python run_experiment.py configs/{}.json > logs/{}.log 2>&1 \
   && echo "--- {} done $(date +%T)" || echo "!!! {} FAILED, see logs/{}.log"'
echo "=== all done $(date)"
