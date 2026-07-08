---
title: "Supplementary material"
---

# Supplementary material for: Quantifying the net safety benefit of a portable advance warning sign: a probabilistic threshold model for low-impact roadside work

D. Tilton, J.D. van der Walt

This supplement provides the deterministic verification table (S1), convergence diagnostics (Fig. S1), the correlation robustness detail (S2), the harm-definition robustness detail (S3), and the response-allocation detail (S4). The full input specification is Table 1 of the main text; the model source code, configuration files, and per-iteration traces that generate every value are available from the repository listed in the Data and code availability statement.

## Table S1. Full input specification

Primary parameterisation (baseline). The faster-deployment variant replaces d_AW with Triangular(20, 30, 40) m, v_walk with Triangular(1.2, 1.5, 2.0) m/s, and t_handle with Triangular(5, 10, 15) s; the optimistic-response variant fixes p_R = 0.60. Monte Carlo: 20,000 iterations per scenario; per-scenario seed = 12345 + grid index; grid Q in {50, 100, 200, 400, 600, 800, 1000} veh/h by T in {0.05, 0.1, 0.25, 0.5, 1, 2, 4} h.

| Parameter | Symbol | Distribution | Units |
|---|---|---|---|
| Advance warning distance | d_AW | Triangular(30, 50, 70) | m |
| Walking speed | v_walk | Triangular(0.9, 1.3, 1.7) | m/s |
| Handling time per episode | t_handle | Triangular(10, 20, 40) | s |
| Mean baseline speed | mu_V | Normal(47, 3) | km/h |
| Baseline speed spread | sigma_V | Uniform(6, 12) | km/h |
| Speed cap (truncation) | V_cap | Fixed 80 | km/h |
| Response probability | p_R | Uniform(0, 0.30) | - |
| Speed reduction if responding | delta-V | Uniform(0, 15) | km/h |
| Comfortable deceleration | a | Uniform(0.8, 2.5) | m/s^2 |
| Encroachment rate | r_E | Log-uniform(1e-7, 1e-5) | enc/veh-km |
| Lateral reach parameter | alpha | Uniform(0.02, 0.15) | 1/m |
| Work lateral offset | c_work | Uniform(1.0, 3.0) | m |
| Deployment lateral offset | c_deploy | Uniform(0.5, 2.0) | m |
| Worker exposure length | L_worker | Fixed 10 | m |
| Work-vehicle exposure length | L_vehicle | Fixed 6 | m |
| Deployment exposure length | L_deploy | Fixed 1 | m |
| Striking vehicle mass | m_1 | Uniform(1200, 2000) | kg |
| Work vehicle mass | m_2 | Uniform(1800, 3000) | kg |
| Encroachment modifier (extension only) | k_E | Fixed, 0.50 to 0.95 | - |

Correlation structures (imposed by Iman-Conover on Spearman rank): plausible {(mu_V, r_E, 0.3)}; stress {(mu_V, r_E, 0.5), (mu_V, sigma_V, 0.3), (v_walk, t_handle, -0.3), (p_R, delta-V, 0.4)}; behaviourally adverse {(mu_V, p_R, -0.3), (mu_V, r_E, 0.3)}.

## Table S2. Deterministic verification: hand calculation versus model

All inputs fixed at the values below; Q = 400 veh/h, T = 0.25 h; the residual differences in probability terms reflect the near-degenerate speed distribution used for the check (the code clamps the speed spread at 0.1 km/h).

Fixed inputs: d_AW = 50 m, v_walk = 1.25 m/s, t_handle = 20 s, mu_V = 50 km/h, p_R = 0, delta-V = 10 km/h, a = 2.0 m/s^2, r_E = 10^-6 per veh-km, alpha = 0.10 m^-1, c_work = 2.0 m, c_deploy = 1.0 m, L_worker = 10 m, L_vehicle = 6 m, L_deploy = 1 m, m_1 = 1500 kg, m_2 = 2500 kg.

| Quantity | Hand calculation | Model | Relative difference |
|---|---|---|---|
| T_deploy_s | 2.000000e+02 | 2.000000e+02 | 0.0e+00 |
| lambda_w0 | 8.187308e-07 | 8.187308e-07 | 2.6e-16 |
| lambda_v0 | 4.912385e-07 | 4.912385e-07 | 2.2e-16 |
| lambda_d | 2.010750e-08 | 2.010750e-08 | 4.9e-16 |
| p_w(V0) | 3.318122e-01 | 3.317192e-01 | 2.8e-04 |
| p_o(dv0) | 2.241625e-02 | 2.240899e-02 | 3.2e-04 |
| H0 | 2.826766e-07 | 2.825968e-07 | 2.8e-04 |
| H1 | 2.893485e-07 | 2.892669e-07 | 2.8e-04 |
| deltaH | 6.671914e-09 | 6.670042e-09 | 2.8e-04 |

## Fig. S1. Convergence of Monte Carlo estimates

Running mean of dH and running P(dH < 0) for five representative grid points, computed to 50,000 iterations (production runs use 20,000; dashed line). The largest change in P(dH < 0) between 20,000 and 50,000 iterations across these points is given in the table below.

| Q (veh/h) | T (h) | P at 20,000 | P at 50,000 | Absolute drift |
|---|---|---|---|---|
| 400 | 0.05 | 0.1250 | 0.1260 | 0.0010 |
| 400 | 0.25 | 0.1467 | 0.1459 | 0.0008 |
| 400 | 2.0 | 0.1502 | 0.1513 | 0.0010 |
| 50 | 0.05 | 0.1290 | 0.1293 | 0.0003 |
| 1000 | 4.0 | 0.1507 | 0.1511 | 0.0005 |

## Table S3. Correlation robustness detail

Grid-wide P(dH < 0) ranges and the largest per-scenario change against the independent baseline, for each imposed rank-correlation structure (Section 2.10 of the main text). Achieved rank correlations matched targets within 0.01 in every run (pairs involving p_R are omitted in the optimistic-response case, where p_R is fixed and a correlation with it is undefined); correlations imposed on mu_V propagate to the realised speed V0 attenuated by the within-site spread sigma_V (achieved V0 correlations 0.09 to 0.15).

| Case | Structure | P(dH < 0) range | Max abs. change in P | Median ratio of mean dH |
|---|---|---|---|---|
| Standard response | Plausible (rho(mu_V, r_E) = 0.3) | 0.124 to 0.156 | 0.0004 | 1.04 |
| Standard response | Stress (0.5/0.3/-0.3/0.4, see 2.10) | 0.132 to 0.154 | 0.0120 | 1.17 |
| Standard response | Behaviourally adverse (rho(mu_V, p_R) = -0.3; rho(mu_V, r_E) = 0.3) | 0.125 to 0.152 | 0.0055 | 1.02 |
| Optimistic response | Plausible (rho(mu_V, r_E) = 0.3) | 0.505 to 0.604 | 0.0012 | 1.03 |
| Optimistic response | Stress (0.5/0.3/-0.3; p_R pair omitted, see 2.10) | 0.506 to 0.604 | 0.0012 | 1.05 |

## Table S4. Harm-definition robustness detail

| Case | Harm definition / curve set | P(dH < 0) range | Max abs. change in P | Median ratio of mean dH |
|---|---|---|---|---|
| Standard response | Fatality only (worker Rosen et al. 2010 fatality; occupant Wang 2022 fatality) | 0.132 to 0.156 | 0.0101 | 0.31 |
| Standard response | Occupant MAIS 3+, frontal crashes (Wang 2022 Table 4-3) | 0.124 to 0.156 | 0.0002 | 1.00 |
| Standard response | Earlier-generation occupant curve (pre-2022) | 0.125 to 0.156 | 0.0008 | 1.03 |
| Optimistic response | Fatality only | 0.541 to 0.605 | 0.0383 | 0.31 |
| Optimistic response | Occupant MAIS 3+, frontal crashes | 0.505 to 0.604 | 0.0006 | 1.00 |
| Optimistic response | Earlier-generation occupant curve | 0.508 to 0.604 | 0.0031 | 1.03 |

The fatality-only ratio of mean dH (about 0.31) reflects the smaller absolute scale of fatality outcomes; the probability ranges show the deploy-or-omit conclusion is unchanged.

## Table S5. Response-allocation robustness detail

Responders allocated preferentially to the slow or fast tail of the operating-speed distribution at fixed marginal response probability (Section 2.11.2).

| Case | Allocation | P(dH < 0) range | Max abs. change in P | Median ratio of mean dH |
|---|---|---|---|---|
| Standard response | Slow tail responds | 0.128 to 0.155 | 0.0052 | 0.79 |
| Standard response | Fast tail responds | 0.122 to 0.157 | 0.0045 | 1.20 |
| Optimistic response | Slow tail responds | 0.502 to 0.586 | 0.0200 | 0.78 |
| Optimistic response | Fast tail responds | 0.479 to 0.587 | 0.0291 | 1.17 |

## Table S6. Sensitivity detail: PRCC with bootstrap 95% confidence intervals

Value-scale PRCCs (against dH) at Q = 400 veh/h; decision-scale PRCCs (against dH / r_E) isolate drivers of the deploy-or-omit decision (Section 2.12). 1000 bootstrap resamples.

| Input | Scale | PRCC (3 min) | PRCC (15 min) | PRCC (2 h) |
|---|---|---|---|---|
| r_E_per_veh_km | dH | +0.698 [+0.673, +0.721] | +0.672 [+0.647, +0.696] | +0.656 [+0.630, +0.681] |
| p_R | dH | -0.189 [-0.214, -0.165] | -0.201 [-0.225, -0.178] | -0.191 [-0.216, -0.166] |
| mu_v_kmh | dH | +0.104 [+0.074, +0.131] | +0.105 [+0.077, +0.134] | +0.104 [+0.078, +0.131] |
| d_aw_m | dH | +0.082 [+0.054, +0.112] | +0.083 [+0.055, +0.111] | +0.072 [+0.046, +0.102] |
| v_walk_m_s | dH | -0.065 [-0.092, -0.038] | -0.069 [-0.098, -0.041] | -0.075 [-0.103, -0.049] |
| t_handle_s | dH | +0.054 [+0.025, +0.083] | +0.038 [+0.011, +0.065] | +0.026 [+0.000, +0.053] |
| alpha_lat_m_inv | dH | -0.051 [-0.078, -0.022] | -0.032 [-0.059, -0.002] | -0.028 [-0.057, -0.001] |
| deltaV_kmh | dH | -0.074 [-0.097, -0.050] | -0.016 [-0.041, +0.010] | -0.019 [-0.048, +0.008] |
| mu_v_kmh | dH / r_E | +0.212 [+0.182, +0.240] | +0.216 [+0.190, +0.245] | +0.217 [+0.191, +0.246] |
| d_aw_m | dH / r_E | +0.173 [+0.145, +0.201] | +0.168 [+0.142, +0.197] | +0.170 [+0.143, +0.196] |
| p_R | dH / r_E | -0.144 [-0.170, -0.119] | -0.161 [-0.186, -0.135] | -0.147 [-0.174, -0.118] |
| v_walk_m_s | dH / r_E | -0.136 [-0.163, -0.108] | -0.140 [-0.168, -0.113] | -0.145 [-0.173, -0.117] |
| t_handle_s | dH / r_E | +0.097 [+0.070, +0.125] | +0.085 [+0.056, +0.112] | +0.074 [+0.049, +0.101] |
| alpha_lat_m_inv | dH / r_E | -0.080 [-0.109, -0.052] | -0.065 [-0.094, -0.037] | -0.068 [-0.097, -0.039] |
| c_deploy_m | dH / r_E | -0.057 [-0.086, -0.028] | -0.057 [-0.086, -0.030] | -0.059 [-0.086, -0.032] |
| deltaV_kmh | dH / r_E | -0.047 [-0.073, -0.021] | -0.023 [-0.050, +0.003] | -0.020 [-0.047, +0.008] |