# UnifiedGLAMAttempt

## TOML simulation/fitting runner

Run a full simulation, allocation fit, regression fit, and artifact export with:

```bash
julia --project=. --startup-file=no run_simulation_fit.jl example_gamma_params.toml
```

The parameter file controls simulation priors, cluster anchoring, allocation MCMC, regression-data construction (`truth`, `map`, or `posterior_mean`), and dense or gamma-aware regression MCMC. Each run writes separate serialized artifacts for simulated data, allocation chain, fitted supervised data, regression chain, and run metadata under the configured `run.output_dir`.

Progress output is enabled by default. Set `progress = false` in `[run]`, `[allocation_mcmc]`, or `[regression_mcmc]` to silence phase messages or bars; set `progress_every` in the MCMC sections to control update frequency.
