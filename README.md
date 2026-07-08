# Net-safety-benefit model for a portable advanced warning sign

Model, configuration, and analysis code for:

> Tilton, D., van der Walt, J.D. **Quantifying the net safety benefit of a portable advanced warning sign: a probabilistic threshold model for low-impact roadside work.** *Accident Analysis & Prevention* (submitted).

The model compares expected serious injury (MAIS 3+) per job with and without a portable advanced warning sign for low-impact roadside work conducted outside the travelled way. It uses a roadside-encroachment framework, propagates uncertainty in every input by Monte Carlo simulation, and returns the probability that the sign reduces net serious harm, P(ΔH < 0), across a grid of traffic flows and work durations.

## Install

```bash
pip install -e .[dev]   # Python >= 3.10
```

## Run

```bash
python scripts/run_suite.py          # full analysis grid + robustness suite
python scripts/make_figures.py       # regenerate every manuscript figure
python scripts/make_results_digest.py
pytest                               # test suite, including a bit-exact reproduction gate
```

Results are written to `outputs/` (gitignored).

## Structure

- `src/aw_model/` — model package: input distributions, injury-severity risk functions, the encroachment/harm model, and the sensitivity analysis (PRCC).
- `config/` — base parameters, scenario grids, correlation structures, and severity-curve variants.
- `scripts/` — run the suite, generate figures, build the results digest.
- `tests/` — unit tests and a bit-exact reproduction gate against the reference outputs.
- `paper/` — manuscript source, reference list, and supplementary material.

## Citation

Please cite the paper above.

## License

Released under the MIT License. See [LICENSE](LICENSE).
