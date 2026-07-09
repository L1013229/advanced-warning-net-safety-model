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
pytest                               # test suite, including a bit-exact reproduction gate
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
- `tests/` — unit and structural tests, including a bit-exact reproduction gate against the reference outputs of the earlier model version.

The manuscript and supplementary material are available from the journal; this repository carries the current, final version of the analysis code that implements and executes the method.

## Citation

Please cite the paper above.

## License

Released under the MIT License. See [LICENSE](LICENSE).
