# Results of record

`results_of_record.csv` holds the model's grid outputs at the current released
configuration: 49 grid points for each of four cases, at full float64
precision.

It exists because nothing previously pinned the answer *this* version of the
model reports. `tests/fixtures/v01_*.csv` pin the *previous* version (v0.1) and
prove the rebuild did not change the legacy answer — a useful check, but it
says nothing about whether today's code still produces the numbers that were
written down from today's code.

## What is pinned

| Case | Overlay | Numerics path |
|---|---|---|
| `baseline` | — | elementwise NumPy only |
| `highPR` | `scenarios/high_pr.yaml` | elementwise NumPy only |
| `baseline_fastDeploy` | `scenarios/fast_deploy.yaml` | elementwise NumPy only |
| `baseline_corr_plausible` | `correlation/plausible.yaml` | Iman-Conover → LAPACK |

Columns: `mean_deltaH`, `median_deltaH`, `median_rel_deltaH`, `p_benefit`,
`p_benefit_lo95`, `p_benefit_hi95`.

## Declared tolerance

| Column | Standard | Why |
|---|---|---|
| `p_benefit` | **bit-exact** (`==`) | It is `k / n_iter` for an integer count `k`. An identical Monte Carlo stream gives an identical float; any perturbation that moves one draw across zero moves `k`. |
| `p_benefit_lo95`, `p_benefit_hi95` | **bit-exact** (`==`) | Pure arithmetic on `p_benefit` and `n_iter`. |
| `mean_deltaH`, `median_deltaH`, `median_rel_deltaH` | rtol `1e-12` | The correlation-free cases run through elementwise NumPy only (`exp`, `log`, `clip`). Identical on one build; `1e-12` absorbs last-ulp differences in libm across NumPy builds. |
| the same three, `*_corr_*` case | rtol `1e-9` | Iman-Conover goes through `np.linalg.cholesky` and `np.corrcoef`, i.e. the system BLAS/LAPACK, whose last bits legitimately differ across backends and thread counts. Claiming `1e-12` here would be a tolerance we could not honour. |

Both bounds sit many orders of magnitude below the last digit of any reported
value — the paper's representative table prints three decimal places on
quantities of order `1e-2` — so a real change in a reported number cannot hide
inside either.

## What is pinned around it

- **Interpreter and dependencies**: `requirements-lock.txt`, exact `==` pins,
  installed with `--no-deps` by the reproducibility workflow.
- **Seeds**: `monte_carlo.seed` in `config/base.yaml`; per-grid-point seed is
  `seed + index`; the Iman-Conover score matrix derives its own generator from
  `SeedSequence([seed, 0x1C0C])`; the PRCC bootstrap uses `seed=20260704`.
- **RNG stream order**: `CANONICAL_DIST_ORDER` in `aw_model.config`, now
  enforced by `validate_config` and followed directly by `sample_all`.
  Reordering the `distributions:` block changes which slice of the stream each
  variable receives and therefore every result; before this was enforced, a
  YAML tidy-up would have done it silently.
- **BLAS threading**: the workflow caps every threading environment variable
  at 1, so a runner with a different core count cannot move a decomposition.

## Serialisation

The CSV is written with `%.17g`, which round-trips float64 exactly, and read
back with `float_precision="round_trip"`. Both halves matter. Pandas' default
CSV parser is *not* correctly rounded and reads `0.1483` back as
`0.1482999999999999`; a gate reading the reference through it would either
fail on unchanged code or, with a loose enough tolerance, hide a real change
behind its own parse error.

The same shape of problem — a number passing through two rounding steps
instead of one — is what put a wrong final digit into a reported table
before. The record is deliberately not a rounding step.

## Regenerating

```bash
python scripts/make_results_of_record.py          # rewrite
python scripts/make_results_of_record.py --check  # CI: fail if it would change
```

Regenerating is a deliberate act. It means a reported number changed, and the
commit that does it has to say why.
