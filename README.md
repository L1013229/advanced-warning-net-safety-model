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
pytest                               # test suite, including both reproducibility gates
python scripts/run_suite.py          # full analysis suite: primary grid, correlation,
                                     # response-allocation, encroachment-reduction,
                                     # sensitivity, convergence, and validation phases
```

`pytest` also works directly from a clean checkout without installing (the test
configuration adds `src/` to the import path). Results are written to `outputs/`
(gitignored); every run records its fully merged configuration and run metadata
so any number in the paper traces to the exact inputs that produced it.

## Structure

- `src/aw_model/` — model package: input distributions and Iman-Conover rank correlation, injury-severity risk functions, the encroachment/harm model, and the sensitivity analysis (PRCC with bootstrap confidence intervals).
- `config/` — base parameters plus overlay files for the scenario, correlation, and severity-curve variants reported in the paper.
- `scripts/run_suite.py` — executes the full analysis reported in the paper.
- `results/` — the results of record for this model version, at full float64 precision, with the declared tolerance for each column (`results/RESULTS_OF_RECORD.md`).
- `tests/` — unit and structural tests, plus two reproducibility gates: `test_reproduction.py` against the earlier model version's reference outputs, and `test_result_of_record.py` against this version's own published grid.

## Reproducibility

`.github/workflows/reproducibility.yml` re-runs the model on a pinned
interpreter, a pinned dependency graph and pinned BLAS threading, and requires
that the results of record regenerate byte-identically. `p_benefit` and its
confidence bounds are held bit-exact; the continuous quantities are held to a
declared tolerance, tighter on the correlation-free path than on the
Iman-Conover path, which goes through LAPACK. The reasoning behind each bound
is in `results/RESULTS_OF_RECORD.md`.

The RNG consumption order is part of the pinning: `sample_all` draws one array
per distribution from a single seeded generator, so the order of the
`distributions:` block decides which slice of the stream each variable
receives. `validate_config` now enforces that order against
`CANONICAL_DIST_ORDER` rather than only checking that each name is present.

The manuscript and supplementary material are available from the journal; this repository carries the current, final version of the analysis code that implements and executes the method.

## Citation

Please cite the paper above.

## License

Released under the MIT License. See [LICENSE](LICENSE).
