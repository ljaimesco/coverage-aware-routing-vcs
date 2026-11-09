"""jcov: coverage-aware route selection under location-diversity incentives.

Implements the journal-extension mechanism:
  Stage 1  continuous location-diversity game  -> deviation targets t*
  Stage 2  coverage-aware selection of road routes near the targets
  Analysis realization gaps and the epsilon-equilibrium certificate (Theorem 2)

All distances are in meters (SUMO network coordinates).
"""
