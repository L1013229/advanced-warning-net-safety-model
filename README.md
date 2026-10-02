# Net safety benefit model for a portable advanced warning sign

Model, configuration and analysis code for:

> Tilton, D., van der Walt, J.D. **Quantifying the net safety benefit of a portable advanced warning sign: a probabilistic threshold model for low-impact roadside work.** *Accident Analysis & Prevention* (submitted, October 2026).

The model compares expected serious injury (MAIS 3+) per job with and without a portable advanced warning sign for low-impact roadside work outside the travelled way. It uses a roadside encroachment framework, propagates uncertainty in every input by Monte Carlo simulation, and returns the probability that the sign reduces net serious harm, P(ΔH < 0), across a grid of traffic flows and work durations.

## Releases

- **v1.1** is the model reported in the paper. Each iteration is one job under one draw of the uncertain inputs, with vehicle-to-vehicle variation averaged inside it. The sign acts by making a share of departing drivers aware sooner.
- **v1.0** is the earlier version the paper refers to as release v1.0. It drew one vehicle per iteration. The current code keeps that construction behind `config/scenarios/representative_vehicle.yaml`.

## Install and run

```bash
pip install -e .[dev]          # Python >= 3.10
pytest                         # unit tests and both reproducibility gates
python scripts/run_suite.py    # the full analysis reported in the paper
```

Results are written to `outputs/`. Every run records its merged configuration and run metadata, so any number in the paper traces to the inputs that produced it.

## Layout

- `src/aw_model/` the model: input distributions and rank correlation, injury severity curves, the encroachment and harm model, the job-expectation iteration, and the sensitivity analysis.
- `config/` base parameters and the overlay files for the scenario, correlation and severity variants reported in the paper.
- `scripts/run_suite.py` runs the full analysis. `scripts/make_figures.py` draws the figures.
- `results/` the results of record for this release at full precision, with the tolerance declared for each column in `results/RESULTS_OF_RECORD.md`.
- `tests/` unit and structural tests, plus two reproducibility gates: `test_reproduction.py` against the original model's reference outputs, and `test_result_of_record.py` against this release's own published grid.

## Reproducibility

`.github/workflows/reproducibility.yml` re-runs the model on a pinned interpreter, a pinned dependency graph (`requirements-lock.txt`) and single-threaded BLAS, and requires the results of record to regenerate byte for byte. The random number stream order is part of the pinning: `sample_all` draws one array per distribution from a single seeded generator, and `validate_config` enforces that order.

## License

MIT. See [LICENSE](LICENSE).
