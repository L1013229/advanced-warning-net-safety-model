---
title: "Quantifying the net safety benefit of a portable advance warning sign: a probabilistic threshold model for low-impact roadside work"
link-citations: true
---

# Abstract {.unnumbered}

Temporary traffic management treats advance warning as a near-universal requirement, yet the worker who places a portable warning sign is exposed to live traffic while doing so. For low-impact roadside work conducted entirely outside the travelled way, where drivers are not required to change speed or path, the sign's protective value is uncertain while its deployment cost is certain. No quantitative method has been available to determine whether the sign produces a net reduction in serious harm once both effects are counted.

This paper develops a probabilistic, encroachment-based model that compares expected serious injury outcomes (MAIS 3+) per job with and without a portable advance warning sign, across 49 combinations of traffic flow (50 to 1000 veh/h) and work duration (3 min to 4 h). The model propagates uncertainty in encroachment frequency, driver response, operating speed, deployment exposure, and injury severity through Monte Carlo simulation, and reports the probability that the sign reduces net serious harm, P(dH < 0).

Under evidence-based assumptions about driver response to passive signs, the sign was unlikely to reduce serious harm in any scenario (P(dH < 0) = 0.12 to 0.16); the median outcome added 0.1 to 11 percent to per-job serious-harm expectation, with the largest penalties for the shortest jobs. Under a deliberately optimistic response assumption the sign became marginally more likely to help than harm (0.51 to 0.60) but never reached the 0.80 high-confidence threshold. These conclusions were insensitive to imposed input correlations, to the harm definition (severe injury or fatality only), and to which drivers respond. High-confidence benefit arose only when the sign was assumed to reduce road departures during the work period, by at least 15, 10, and 5 percent for 3-minute, 6-minute, and 15-minute-or-longer jobs respectively. The framework converts an assumed protective default into a testable, decision-ready quantity, and identifies the evidence that would most improve the decision.

**Keywords:** temporary traffic management; advance warning sign; work zone safety; worker exposure; probabilistic risk assessment; Monte Carlo simulation; roadside encroachment

# 1. Introduction

Temporary traffic management is often treated as an unqualified safety good: a set of signs, devices, delineation, and operating procedures that compensates for temporarily degraded road conditions during road works. In risk terms it is better understood as an intervention that reallocates and transforms risk across the work system rather than simply reducing it. Every device that a worker places on or beside a road creates a period of additional exposure: the worker must walk to a position near live traffic, handle equipment, and return. For devices that require upstream placement, such as advance warning signs, that exposure occurs where the worker is not yet protected by the device being deployed. There is therefore a risk cost to deploying the control that must be balanced against the good the control does, and whether the net effect is beneficial depends on the size of the downstream benefit relative to the upstream cost.

This trade-off is well understood in principle but has not been quantified in a form that supports operational decision-making. Temporary traffic management (TTM) standards in most jurisdictions (the function termed temporary traffic control in North American practice) treat advance warning as a foundational requirement for nearly all work activities near the road, including short-duration, low-impact tasks conducted entirely outside the travelled way [@nzta_code_2018; @nzta_nzgttm_2024; @austroads_guide_2019; @fhwa_manual_2023]. In those settings the downstream hazard is narrow: the work imposes no lane closure, merge, or speed reduction, and drivers are not required to change their path. The main remaining harm pathway is a vehicle leaving the travelled way and reaching the work area, a rare event that a passive warning sign influences only weakly, if at all [@ullman_flagger_1987; @charlton_conspicuity_2006]. Meanwhile the deployment task concentrates risk into a short window in which a worker is on foot, close to the edgeline, and engaged in a manual task that limits situational awareness [@wang_worker_2012; @debnath_common_2015; @pegula_fatal_2013; @blackman_workzone_2020].

The gap is not a lack of research on work zone safety or warning-sign effectiveness. A substantial body of work examines the speed and behavioural effects of advance warning in lane closures, tapers, and high-speed work zones [@richards_field_1985; @benekohal_speed_1992; @bai_analyzing_2010; @banerjee_influence_2019; @shahin_effects_2023; @oh_enhancing_2024], and systematic reviews catalogue work zone hazards and countermeasure technologies [@nnaji_improving_2020; @rathnasiri_state_2024; @yang_work_2015; @albayati_managing_2023]. That literature concerns settings where the work itself requires a driver response. For low-impact work outside the live lane the mechanisms by which a static sign might reduce harm are different and far less evidenced, and there is little empirical work on the exposure created by on-foot sign deployment, which for short jobs can be a substantial fraction of total time on the shoulder [@wang_worker_2012].

The absence of a quantitative framework means the deploy-or-omit decision is made by compliance with general rules rather than by any assessment, at the class or site level, of net safety benefit. In jurisdictions that have moved from prescriptive to risk-based TTM, practitioners are expected to judge each configuration's residual risk and justify control selection, yet the system provides no explicit method for determining whether a given control produces a net reduction in harm in a given context [@peace_reasonably_2017; @stephens_other_2017]. In New Zealand, whose transition motivates this study's parameterisation, a documentary benchmarking of the TTM system against risk analysis and uncertainty characterisation frameworks found that no stage of the decision cycle produces an explicit risk description [@tilton_benchmarking_2026]; controls continue to be applied by convention, without a risk description that accounts for the costs they introduce [@aven_risk_2016; @pate-cornell_uncertainties_1996]. Two questions are blended that ought to be separated: whether a control is required to meet a compliance baseline, and whether it is expected to reduce serious harm in the specific context once deployment and removal are counted. This paper provides a method for the second question.

The paper develops a probabilistic risk model that compares expected serious injury outcomes (Maximum Abbreviated Injury Scale 3 or higher, MAIS 3+) per job with and without a portable advance warning sign, for low-impact roadside work at operating speeds of 50 km/h or below conducted entirely outside the travelled way. The model represents three collision pathways (worker strike during work, work-vehicle strike during work, and worker strike during sign deployment and removal), propagates uncertainty in all inputs by Monte Carlo simulation, and evaluates an explicit decision rule: deploy the sign if the probability of net benefit, P(dH < 0), exceeds a stated threshold p\*, evaluated here at p\* = 0.5 (more likely to help than harm) and p\* = 0.8 (high confidence).

***Research question.*** For low-impact roadside work conducted outside the live lane at speeds of 50 km/h or below, does a portable static advance warning sign, placed and removed by a worker on foot, produce a net reduction in expected serious harm (MAIS 3+) when the additional worker exposure from deployment and removal is included?

***Hypothesis.*** The sign reduces net serious harm only when traffic volume and driver response to the sign are high; otherwise the added exposure from deployment dominates and the sign increases expected serious harm.

The study pursues three objectives within this scope: (1) quantify the net change in expected serious harm, dH, and the probability that the sign reduces it, across the modelled ranges of traffic flow and work duration; (2) identify break-even conditions at which the sign transitions from net harmful to net beneficial under the stated decision rule; and (3) determine which uncertain inputs drive the deploy-or-omit decision, through global sensitivity screening, to establish where better evidence would most improve decision quality.

# 2. Methods

## 2.1. Scope, strategies, and scenario definition

The model evaluates a single marginal decision: whether to deploy one portable static advance warning sign for a specific low-impact roadside work task, compared with performing the same task without the sign. In practice a worksite may carry several devices; the marginal, one-device comparison is used because it isolates the contribution of the advance warning function itself, and because it corresponds to the operational choice that low-impact work crews actually face. All other temporary traffic controls (work-vehicle positioning, high-visibility clothing, delineation, lighting) are held identical between the two strategies, S0 (no sign) and S1 (sign deployed).

The scope is restricted to work activities that satisfy all of the following: (a) posted or operating speed of 50 km/h or below; (b) all workers and work vehicles remain outside the travelled way throughout the task, with a minimum lateral offset of 1 m from the edge of the travelled way to the nearest worker or work vehicle (Section 2.5); (c) no lane closure, merge, taper, or temporary speed restriction is imposed, so through traffic can continue on its normal path with no TTM-related requirement to change speed or lateral position; and (d) no flagging or other active traffic control is present. Typical examples include roadside inspections, meter reading, minor vegetation clearance, and utility-pit access performed from the shoulder or verge.

Two scenario variables define the analysis grid: traffic flow rate Q past the site (50, 100, 200, 400, 600, 800, and 1000 veh/h) and work duration T (0.05, 0.1, 0.25, 0.5, 1, 2, and 4 h, that is, 3 min to 4 h), giving 49 combinations spanning the operating envelope of low-impact urban roadside work.

## 2.2. Model class and rationale

Conventional crash prediction models estimate crash frequency from segment characteristics such as AADT and cross-section, but they lack the lateral resolution to distinguish a worker standing at the edge of the travelled way from one standing 2 m away, which is precisely the distinction that matters for outside-lane work. The model therefore uses the roadside encroachment framework that underpins the Roadside Safety Analysis Program (RSAP) and related roadside safety analysis [@mak_roadside_2003; @ray_six_2023]. This framework separates collision risk into (a) the rate at which vehicles unintentionally leave the travelled way and (b) the probability that an encroaching vehicle reaches a given lateral offset, which allows the model to represent mathematically the safety value of distance from the live lane.

Expected serious harm per job under each strategy is the sum, over the collision pathways defined in Section 2.3, of expected strike frequency multiplied by the probability of a serious outcome given a strike (Fig. 1). Because all inputs carry material uncertainty, the model is evaluated by Monte Carlo simulation: each iteration draws one realisation of every uncertain input, computes expected harm under S0 and S1 in closed form, and records the difference dH = H(S1) - H(S0). Negative dH means the sign reduces expected serious harm. The primary output is P(dH < 0), the fraction of iterations in which the sign helps, together with the mean and median of dH. Each scenario uses 20,000 iterations with fixed, reproducible seeds; Monte Carlo uncertainty in P(dH < 0) is quantified with 95 percent Wilson score intervals [@wilson_probable_1927], and convergence diagnostics are reported in Section 3.7. The model was implemented in Python (NumPy) and the full model specification was fixed before the production analyses were run; the code, configuration files, and per-iteration traces are available as described in the Data and code availability statement.

The model treats the whole passing stream in a given iteration as travelling at that iteration's sampled speed. Because expected harm is linear in per-vehicle severity, sampling one representative speed per iteration and averaging over iterations yields the same expected harm as integrating severity over the within-stream speed distribution; the across-iteration speed distribution (Section 2.6) plays the role of the driver-to-driver spread.

![Fig. 1. Model structure. Under S0 two collision pathways operate during the work period; S1 adds the deployment pathway, in which the worker is exposed while placing and removing the sign. The net change dH = H(S1) - H(S0) determines whether the sign yields a net safety benefit.](../model/outputs/figures/fig1_model_schematic.png){width=16cm}


## 2.3. Collision pathways

Three pathways define serious-harm accounting; each is a frequency (expected events per job) multiplied by a severity (probability of MAIS 3+ given the event):

* **Pathway 1, worker strike during the work period.** A vehicle departs the travelled way, reaches the worker's lateral position, and strikes the worker at the prevailing operating speed. Operates under S0 and S1; under S1 the operating speed may be reduced by the sign (Section 2.6).
* **Pathway 2, work-vehicle strike during the work period.** An encroaching vehicle reaches and strikes the stationary work vehicle; the harm is carried by the occupant of the striking vehicle through the collision change in velocity (Section 2.7). Operates under S0 and S1 with the same speed modification as Pathway 1. The work vehicle is unoccupied while the crew is at the work area.
* **Pathway 3, worker strike during sign deployment and removal.** While placing or retrieving the sign the worker is on foot near the edgeline for a duration T_deploy; an encroaching vehicle reaches the worker's position during that window. Operates only under S1. Severity uses the baseline speed V0, because the sign is not yet in place during deployment and has just been removed during retrieval.

These three pathways were selected because, within the stated scope, they are the collision-mediated routes to MAIS 3+ harm that differ between the strategies or dominate the exposure. Three pathways are excluded by design: pedestrian and cyclist route-deviation risks introduced by sign placement; minor occupational handling injuries; and vehicle impacts with the sign or stand as a roadside object, which portable-sign crashworthiness requirements render negligible as a source of MAIS 3+ occupant harm [@schmidt_analysis_2015; @seo_computational_2011]. Events are treated as disjoint: at the encroachment frequencies considered here (order 10^-7^ to 10^-5^ per vehicle-kilometre) the probability that a single encroachment generates multiple harms, for example striking the work vehicle and then the worker, is second-order small, consistent with the treatment of encroachment events in RSAP [@mak_roadside_2003].

## 2.4. Encroachment frequency

The expected number of strikes on target j during an exposure window is

$$\lambda_j = N \, (r_E \, L_j) \, P(\text{reach} \ge c_j)$$

where N is the number of vehicles passing during the window (N = Q T for the work period, dimensionless count from veh/h times h), r_E is the encroachment rate in encroachments per vehicle-kilometre, L_j is the effective longitudinal exposure length of the target in kilometres, and P(reach >= c_j) is the probability that an encroaching vehicle reaches the target's lateral offset c_j in metres. Each lambda is therefore dimensionless (expected strikes per job); the units of every input are listed in Table 1.

The encroachment rate is sampled from a log-uniform distribution spanning 10^-7^ to 10^-5^ encroachments per vehicle-kilometre. Section 2.9 derives this range from the encroachment literature and shows the estimates it brackets. In the primary analysis the same sampled r_E applies under both strategies: the sign is assumed to change operating speed, not the rate at which vehicles leave the road. That assumption is relaxed in the extension of Section 2.11.3.

Lateral reach is modelled with an exponential exceedance function, P(reach >= c) = exp(-alpha c), a single-parameter approximation to the lateral-extent distributions used in roadside safety analysis, which decay steeply with distance from the edgeline [@mak_roadside_2003; @cooper_analysis_1980; @ray_six_2023]. The decay parameter alpha is sampled from Uniform(0.02, 0.15) m^-1^; the implied probability of an encroachment reaching 2 m ranges from 0.74 (alpha = 0.15) to 0.96 (alpha = 0.02), a deliberately conservative envelope that assigns substantial reach probability at the small offsets in scope.

## 2.5. Deployment exposure

Under S1 the worker walks from the work vehicle to the sign position, installs the sign, walks back, and repeats the sequence in reverse at pack-down. Total on-foot exposure time is

$$T_{deploy} = 2\left(\frac{2\,d_{AW}}{v_{walk}} + t_{handle}\right)$$

where d_AW is the advance warning placement distance, v_walk the walking speed, t_handle the stationary handling time per episode, and the outer factor 2 covers installation plus removal. During this window the worker is treated as a point target (L_deploy = 1 m) at lateral offset c_deploy sampled from Uniform(0.5, 2.0) m, closer to the edgeline than the work activity itself (c_work from Uniform(1.0, 3.0) m) because sign placement requires approaching the road edge for visibility. The vehicles exposed during deployment number N_deploy = Q T_deploy.

Bounded task parameters with a defensible minimum, most-likely, and maximum but sparse distributional data (d_AW, v_walk, t_handle) are given triangular distributions, the standard elicitation form for such inputs; their ranges are justified in Section 2.9.

## 2.6. Warning-sign effect on operating speed

Baseline operating speed V0 is drawn from a normal distribution with uncertain mean mu_V ~ Normal(47, 3) km/h and uncertain spread sigma_V ~ Uniform(6, 12) km/h, truncated to the interval 0 to 80 km/h. Section 2.9 grounds these parameters in New Zealand urban speed survey data.

The sign's effect is a Bernoulli mixture defined at the work location. With probability p_R a passing driver responds effectively, meaning the driver's speed at the work location is reduced relative to baseline; p_R therefore collapses the full chain of detection, comprehension, decision, and persistence of the response over the warning distance into a single effective parameter, which is why its prior is wide. There is no separate response-at-the-sign parameter: any slowing that has decayed before the work location does not count as an effective response. Conditional on response, speed is reduced by delta-V sampled from Uniform(0, 15) km/h, capped by the kinematically feasible reduction over the available distance at a comfortable deceleration a from Uniform(0.8, 2.5) m/s^2^, using v_1^2 = v_0^2 - 2 a d_AW in consistent units (Fig. 2). The work-period speed under S1 is V1 = max(V0 - I_R min(delta-V, delta-V_max), 0), with I_R the response indicator; non-responding drivers retain V0.

In the standard-response case p_R is sampled from Uniform(0, 0.30); in the optimistic-response case p_R is fixed at 0.60 as a stress test, deliberately strong for a passive sign (Section 2.9). The primary analysis draws the response indicator independently of V0. Because the severity benefit of slowing is concentrated among faster drivers, Section 2.11.2 additionally tests response allocations in which responders are drawn preferentially from the slow or the fast tail of the speed distribution at fixed marginal p_R.

![Fig. 2. (a) Operating-speed mixture at the work location: baseline V0 and the with-sign distribution V1 for an illustrative response share and speed reduction. (b) Kinematic bound on the achievable speed reduction over the warning distance at comfortable decelerations.](../model/outputs/figures/fig2_warning_effect.png){width=16cm}


## 2.7. Injury severity

Severity is represented by logistic risk curves selected from the published crash-injury literature, with the harm outcome defined at person level as MAIS 3+ on the Abbreviated Injury Scale [@aaam_abbreviated_2016]. In this study MAIS 3+ includes fatal outcomes; the fatality-only robustness check in Section 2.11.1 uses curves in which the outcome is death.

*Worker (person on foot).* The probability of MAIS 3+ given impact speed V in km/h is P = 1/(1 + exp(-(-4.6 + 0.078 V))), the severe-injury regression estimated from German In-Depth Accident Study pedestrian cases (n = 694) by @rosen_pedestrian_2010, whose fatality regression from the same dataset (coefficients -7.5 and 0.096, n = 755) is used in the fatality-only check. These GIDAS-based curves are consistent with the earlier fatality curve of @rosen_sander_2009 and with United States estimates [@tefft_impact_2013]. The same worker curve applies to Pathways 1 and 3; no modifiers are applied for posture or personal protective equipment, which the source models do not parameterise.

*Occupant of the striking vehicle.* The probability of MAIS 3+ given the occupant's change in velocity is taken from the current NHTSA injury probability curves estimated from 2010 to 2015 NASS-CDS tow-away crashes [@wang_mais0508_2022]: for all crash modes, P = e^z^/(1 + e^z^) with z = -6.9540 + 0.1637 D, D the delta-V in mph; the frontal-crash and fatality curves from the same report are used in robustness checks. The occupant delta-V is obtained from the operating speed with a perfectly inelastic, one-dimensional momentum transfer, delta-V = V m_2/(m_1 + m_2), where m_1 is the striking vehicle mass (Uniform(1200, 2000) kg) and m_2 the stationary work vehicle mass (Uniform(1800, 3000) kg), the standard simplified mapping used when delta-V-based risk curves are applied to conflict configurations [@shelby_deltav_2011; @evans_driver_1994; @joksch_velocity_1993].

Both curves are strongly nonlinear, rising steeply above roughly 40 km/h (Fig. 3). This nonlinearity is the mechanism by which a modest speed reduction can produce a disproportionate severity benefit, and equally the reason the benefit is concentrated in the fast tail of the speed distribution.

![Fig. 3. Injury severity curves used in the primary analysis and robustness checks: (a) worker (person on foot) severe injury (MAIS 3+) and fatality versus impact speed; (b) occupant of the striking vehicle, MAIS 3+ (all crashes and frontal) and fatality versus delta-V.](../model/outputs/figures/fig3_severity_curves.png){width=16cm}


## 2.8. Harm computation and structural properties of the comparison

For each iteration,

$$H(S_0) = \lambda_1 p_w(V_0) + \lambda_2 p_o(\Delta v_0)$$
$$H(S_1) = \lambda_1' p_w(V_1) + \lambda_2' p_o(\Delta v_1) + \lambda_3 p_w(V_0)$$
$$dH = H(S_1) - H(S_0)$$

where lambda_1 and lambda_2 are the work-period worker and work-vehicle strike frequencies (primed values differ from unprimed only in the extension of Section 2.11.3), and lambda_3 is the deployment-pathway frequency.

Two structural properties of this comparison should be stated openly, because they are consequences of model construction rather than empirical findings. First, the deployment pathway always adds harm under S1; the only compensating mechanism in the primary model is the speed reduction V0 to V1 among responding drivers. Second, every term in dH is proportional to the sampled encroachment rate, so the sign of dH, and therefore the deploy-or-omit decision and P(dH < 0), is algebraically invariant to r_E. The encroachment rate scales how much harm is at stake; it cannot determine which strategy is better. This separation between magnitude drivers and decision drivers is developed in the sensitivity analysis (Sections 2.12 and 3.4) and matters for interpreting evidence priorities (Section 4.3).

## 2.9. Key assumptions and evidence for the dominant inputs

This subsection provides the validation layer for the inputs that drive the results: for each, the empirical basis, how the range was constructed, and the directional consequence of changing it. Table 1 summarises all inputs.

*Encroachment rate r_E (dominant magnitude driver).* Direct measurements of encroachment frequency on low-speed urban roads do not exist; the classical datasets are medians of divided highways [@hutchinson_safety_1967] and Canadian run-off-road studies [@cooper_analysis_1980], synthesised across six decades by @ray_six_2023. For urban arterial streets, @glennon_roadside_1976 estimated roadside accident frequency as y = 0.474 + 0.000254 ADT accidents per mile-year and converted accidents to encroachments with a factor of 5.23, describing the results as order-of-magnitude estimates. Applied across this study's flow range (ADT approximately 1200 to 24,000), those relations imply roughly 2 to 6 x 10^-6^ encroachments per vehicle-kilometre. The log-uniform prior from 10^-7^ to 10^-5^ brackets these estimates with an order of magnitude on each side, weighting each decade equally to represent genuine ignorance rather than false confidence. Because of the invariance identified in Section 2.8, widening or narrowing this range rescales the absolute harm at stake without changing P(dH < 0); its influence returns in any absolute-risk or cost-effectiveness extension.

*Driver response p_R and speed reduction delta-V (dominant decision drivers).* Field studies of passive, static work zone signage consistently find small average speed effects with substantial non-response. @bai_analyzing_2010 measured motorist responses to a standard static advance warning sign in rural two-lane work zones and found portable changeable message signage more effective for most vehicle classes; @benekohal_speed_1992 found that around 63 percent of drivers reduced speed after the first sign of a signed rural work zone with active lane closure, a setting with far stronger hazard cues than in-scope work; @richards_field_1985 found standard signing the least effective of the speed-control techniques they evaluated, with meaningful reductions requiring flagging, enforcement, or dynamic devices; simulator and observational studies report responses that decay quickly when the signed hazard is not visible or credible [@banerjee_influence_2019; @steinbakk_analysing_2017; @steinbakk_speed_2019; @vignali_road_2019; @charlton_conspicuity_2006]. Mapping this evidence to the model: Uniform(0, 0.30) for p_R spans zero response (credibility failure, common where no hazard is visible) up to a minority response share consistent with the largest effects measured for purely passive signage in low-cue settings, and Uniform(0, 15) km/h for delta-V spans the observed range from no change to the largest mean reductions reported for enhanced static signing. The optimistic case (p_R = 0.60) intentionally exceeds the passive-sign evidence to test whether even a strong response rescues the sign. Higher p_R or delta-V move the decision toward deployment; the results section quantifies how far.

*Operating speed mu_V and sigma_V.* The New Zealand national speed monitoring survey measures urban 50 km/h sites annually: all-vehicle mean speeds were 40.0 to 44.1 km/h across 2018 to 2024 (85th percentiles 51.2 to 54.3 km/h), while the pre-2016 free-running car series recorded means of 50.4 to 56.5 km/h [@nzta_annual_2025]. The prior mu_V ~ Normal(47, 3) km/h centres between the all-vehicle and free-flow measurement conventions, representing the free-flowing passing stream relevant to encroachment risk; sigma_V ~ Uniform(6, 12) km/h covers the spread implied by the surveyed 85th-minus-mean gaps of 9 to 11 km/h. Higher operating speeds increase severity on all pathways including deployment, and the results show they make the sign less favourable on balance.

*Deployment parameters d_AW, v_walk, t_handle.* The placement distance triangular(30, 50, 70) m centres on the 50 m advance warning placement of New Zealand low-speed urban practice, prescribed by the former code of practice and carried into standard layouts under the current risk-based guide [@nzta_code_2018; @nzta_nzgttm_2024], with bounds covering compliant site variation; the faster-deployment variant (triangular(20, 30, 40) m) represents the reduced placement that guidance permits for short-duration low-speed work. Walking speed triangular(0.9, 1.3, 1.7) m/s sits inside the comfortable-to-brisk adult gait range measured by @bohannon_comfortable_1997 and confirmed meta-analytically [@bohannon_normal_2011], slowed from unencumbered means because the worker carries a sign for half the distance. Handling time triangular(10, 20, 40) s per episode is an operational estimate consistent with New Zealand TTM training practice; no published time-and-motion data exist for this task, which is itself a finding about the evidence base. Longer distances, slower walks, and longer handling all increase deployment exposure and disfavour the sign; the faster-deployment assumption set bounds this influence from below.

*Comfortable deceleration a.* Uniform(0.8, 2.5) m/s^2^ represents discretionary, comfort-range slowing in response to advisory information, below the 3.4 m/s^2^ that AASHTO adopts as a comfortable design deceleration for stopping sight distance [@fambro_driver_2000] and consistent with the passenger-comfort deceleration bands synthesised in the ride-comfort literature [@elbanhawi_passenger_2015]. This parameter only binds when the sampled delta-V exceeds what the warning distance permits, so its influence is limited (Section 3.4).

*Severity curves.* The curve sources and their provenance are given in Section 2.7; Section 2.11.1 quantifies the effect of replacing them.

Because the model is specified from these evidence-based ranges rather than calibrated to a desired outcome, and because the optimistic branch deliberately biases assumptions toward the sign, the design tests the hypothesis rather than supporting it by construction; conservative and optimistic choices are identified as such throughout.

## 2.10. Uncertainty propagation and input correlation

Unless stated otherwise, inputs are sampled independently. Independence is a defensible default here because most inputs describe physically distinct processes (gait speed and encroachment propensity, for example, have no plausible common cause at a given site), and because of the structural insensitivity of the decision to the scale factor r_E (Section 2.8). It is nonetheless a simplification, and small correlations can matter in Monte Carlo models, so dependence was tested directly rather than argued away.

Rank correlation was imposed with the distribution-free method of @iman_conover_1982, which preserves every marginal exactly while reordering samples to match a target Spearman structure. Three structures were run on the full grid: a plausible structure (rho = 0.3 between mu_V and r_E, faster sites see more departures); a stress structure (rho = 0.5 between mu_V and r_E, 0.3 between mu_V and sigma_V, -0.3 between v_walk and t_handle, and 0.4 between p_R and delta-V, so that responding populations also slow more); and a behaviourally adverse structure (rho = -0.3 between mu_V and p_R, faster streams respond less, plus 0.3 between mu_V and r_E). Pairs involving p_R are omitted in the optimistic-response case, where p_R is fixed and a correlation with it is undefined. Achieved rank correlations were verified against targets in every run and are reported with the outputs, including the attenuated correlation that reaches the realised speed V0 through mu_V.

## 2.11. Robustness analyses

**2.11.1. Harm definition.** Because the argument leans on severity nonlinearity, the analysis was repeated with (a) the occupant curve restricted to frontal crashes, (b) the pre-2022 occupant curve generation used in an earlier version of this model, and (c) a fatality-only harm definition using the worker fatality curve of @rosen_pedestrian_2010 and the occupant fatality curve of @wang_mais0508_2022. The fatality-only definition changes the outcome variable, not just its scale, because the fatality curves are steeper and shift the balance further toward tail speeds.

**2.11.2. Response allocation.** Holding the marginal response probability fixed, responders were allocated preferentially to the slow tail of the speed distribution (rank weights declining linearly with speed, the behaviourally pessimistic case if inattentive and faster drivers are least responsive) or to the fast tail (the optimistic case), against the independent baseline. This turns the question of whether the sign influences the drivers who matter most from an assumption into a quantified sensitivity.

**2.11.3. Encroachment-rate reduction extension.** The primary model gives the sign no influence on departure frequency. The extension scales the work-period encroachment rate under S1 by k_E from 0.95 down to 0.50 (5 to 50 percent reductions), leaving the deployment pathway unscaled because the sign is absent during deployment and removal, and identifies the smallest reduction at which the decision rule favours deployment. No empirical basis exists to assign a probability distribution to k_E in this setting, so it is treated as a what-would-it-take diagnostic rather than a claimed mechanism.

## 2.12. Break-even, decision classification, and sensitivity analysis

For each traffic flow, the break-even duration at threshold p\* is the shortest tested duration with P(dH < 0) >= p\*. Scenarios are classified with the Wilson bounds: sign supported when the lower 95 percent bound on P(dH < 0) is at least 0.8; sign not supported when the upper bound is below 0.5; uncertain otherwise.

Global sensitivity uses partial rank correlation coefficients (PRCC) between each sampled input and dH, controlling for all other inputs, computed on rank-transformed values by the regression-residual method [@marino_methodology_2008; @blower_sensitivity_1994] at Q = 400 veh/h for 3-minute, 15-minute, and 2-hour jobs, with 95 percent bootstrap confidence intervals (1000 resamples). Because dH is proportional to r_E (Section 2.8), PRCCs are reported on two scales: against dH, which ranks drivers of the harm magnitude, and against dH/r_E, which isolates drivers of the deploy-or-omit decision.

Table 1. Model inputs, distributions, units, and evidence basis (cross-references give the subsection where each is justified).

| Parameter | Symbol | Distribution | Units | Basis |
|---|---|---|---|---|
| Advance warning distance | d_AW | Tri(30, 50, 70); fast variant Tri(20, 30, 40) | m | NZ practice (2.9) |
| Walking speed | v_walk | Tri(0.9, 1.3, 1.7); fast variant Tri(1.2, 1.5, 2.0) | m/s | Gait literature (2.9) |
| Handling time per episode | t_handle | Tri(10, 20, 40); fast variant Tri(5, 10, 15) | s | Operational estimate (2.9) |
| Mean baseline speed | mu_V | Normal(47, 3) | km/h | NZ speed surveys (2.9) |
| Baseline speed spread | sigma_V | Uniform(6, 12) | km/h | NZ speed surveys (2.9) |
| Speed cap (truncation) | V_cap | Fixed 80 | km/h | Physical bound (2.6) |
| Response probability | p_R | Uniform(0, 0.30); optimistic fixed 0.60 | - | Static-sign evidence (2.9) |
| Speed reduction if responding | delta-V | Uniform(0, 15) | km/h | Static-sign evidence (2.9) |
| Comfortable deceleration | a | Uniform(0.8, 2.5) | m/s^2^ | Comfort braking (2.9) |
| Encroachment rate | r_E | Log-uniform(10^-7^, 10^-5^) | enc/veh-km | Encroachment literature (2.9) |
| Lateral reach parameter | alpha | Uniform(0.02, 0.15) | m^-1^ | Lateral extent models (2.4) |
| Work lateral offset | c_work | Uniform(1.0, 3.0) | m | Scope definition (2.1) |
| Deployment lateral offset | c_deploy | Uniform(0.5, 2.0) | m | Placement practice (2.5) |
| Worker exposure length | L_worker | Fixed 10 | m | Work-area footprint (2.4) |
| Work-vehicle exposure length | L_vehicle | Fixed 6 | m | Light commercial vehicle (2.4) |
| Deployment exposure length | L_deploy | Fixed 1 | m | Point target, walking (2.5) |
| Striking vehicle mass | m_1 | Uniform(1200, 2000) | kg | Passenger fleet (2.7) |
| Work vehicle mass | m_2 | Uniform(1800, 3000) | kg | Light commercial fleet (2.7) |
| Encroachment modifier (extension) | k_E | Fixed, 0.50 to 0.95 | - | Diagnostic sweep (2.11.3) |

# 3. Results

Results are organised by the three objectives, followed by the robustness analyses. dH values are reported as MAIS 3+ outcomes per million jobs, and, where it aids interpretation, as the median relative change in per-job serious-harm expectation, dH/H(S0). For scale, the baseline per-job expectation H(S0) itself averages 0.014 (lightest scenario) to 23 (heaviest scenario) MAIS 3+ outcomes per million jobs across the grid. Four assumption sets define the primary analysis (Table 2): standard response with standard deployment (the baseline), standard response with faster deployment, and the two optimistic-response counterparts.

Table 2. Assumption sets for the primary analysis (all other inputs per Table 1).

| Case | Driver response p_R | Deployment (d_AW, v_walk, t_handle) |
|---|---|---|
| (a) Standard, standard | Uniform(0, 0.30) | Tri(30,50,70) m, Tri(0.9,1.3,1.7) m/s, Tri(10,20,40) s |
| (b) Standard, faster | Uniform(0, 0.30) | Tri(20,30,40) m, Tri(1.2,1.5,2.0) m/s, Tri(5,10,15) s |
| (c) Optimistic, standard | Fixed 0.60 | as (a) |
| (d) Optimistic, faster | Fixed 0.60 | as (b) |

## 3.1. Net change in expected serious harm (Objective 1)

Under the standard-response cases the sign was unlikely to reduce serious harm in any of the 49 scenarios. P(dH < 0) ranged from 0.124 to 0.156 under standard deployment and 0.135 to 0.156 under faster deployment (Fig. 4), so in 84 to 88 percent of iterations the sign increased expected serious harm. The median dH was positive in every scenario: the typical deployment added between 0.1 and 11 percent to the per-job serious-harm expectation, with the largest relative penalties at the shortest durations, where deployment exposure is largest relative to the work period (faster deployment halves the upper end to 5.3 percent). The mean dH was positive for 3-minute jobs and negative for long jobs (Fig. 5), reflecting a minority of iterations that combine high response with high speeds and produce large benefits; this mean-median divergence is why the probability and median, not the mean alone, carry the decision information.

![Fig. 4. Probability that the sign reduces net serious harm, P(dH < 0), across traffic flow and work duration for the four assumption sets of Table 2. The colour scale is centred at 0.5; no scenario approaches the 0.8 high-confidence threshold.](../model/outputs/figures/fig4_probability_heatmaps.png){width=16cm}


Under the optimistic-response cases the sign became marginally more likely to help than harm: P(dH < 0) ranged from 0.506 to 0.604 (standard deployment) and 0.553 to 0.605 (faster deployment), with both mean and median dH negative in every scenario (typical reduction 0.6 to 12.9 percent of per-job expectation, growing with duration). P(dH < 0) did not reach 0.80 anywhere on the grid: even a response share double the strongest passive-sign evidence does not deliver high-confidence net benefit once deployment exposure is counted.

Table 3 gives representative values at 400 veh/h. Within each assumption set, P(dH < 0) is nearly flat in traffic flow (Fig. 4), because flow scales the work-period benefit and the deployment cost together; what varies the outcome is duration, which changes only the work-period side of the balance.

![Fig. 5. Mean net change in serious harm (MAIS 3+ per million jobs). Negative values indicate the sign reduces expected harm on average; the mean-median divergence in the standard-response cases is discussed in Section 3.1.](../model/outputs/figures/fig5_mean_dh_heatmaps.png){width=16cm}


Table 3. Representative results at Q = 400 veh/h (dH in MAIS 3+ per million jobs; Wilson 95% CI on P(dH < 0); relative change is median dH/H(S0)).

| Case | Duration | Mean dH | Median dH | P(dH < 0) [95% CI] | Median relative change |
|---|---|---|---|---|---|
| (a) Standard, standard | 3 min | +0.008 | +0.0037 | 0.129 [0.125, 0.134] | +11.0% |
| (a) Standard, standard | 15 min | -0.013 | +0.0036 | 0.146 [0.141, 0.151] | +2.2% |
| (a) Standard, standard | 2 h | -0.211 | +0.0035 | 0.150 [0.145, 0.155] | +0.3% |
| (b) Standard, faster | 3 min | +0.001 | +0.0018 | 0.140 [0.135, 0.145] | +5.3% |
| (b) Standard, faster | 15 min | -0.019 | +0.0017 | 0.148 [0.144, 0.153] | +1.1% |
| (b) Standard, faster | 2 h | -0.206 | +0.0017 | 0.150 [0.145, 0.155] | +0.1% |
| (c) Optimistic, standard | 3 min | -0.009 | -0.0002 | 0.506 [0.499, 0.513] | -0.8% |
| (c) Optimistic, standard | 15 min | -0.097 | -0.0092 | 0.580 [0.573, 0.586] | -9.7% |
| (c) Optimistic, standard | 2 h | -0.858 | -0.0959 | 0.599 [0.592, 0.606] | -12.3% |
| (d) Optimistic, faster | 3 min | -0.014 | -0.0014 | 0.556 [0.550, 0.563] | -6.8% |
| (d) Optimistic, faster | 15 min | -0.099 | -0.0102 | 0.588 [0.581, 0.595] | -10.9% |
| (d) Optimistic, faster | 2 h | -0.820 | -0.0962 | 0.600 [0.593, 0.607] | -12.4% |

## 3.2. Break-even conditions (Objective 2)

Under both standard-response cases no break-even duration exists on the tested grid for either threshold: P(dH < 0) never reaches 0.5 between 3 minutes and 4 hours at any flow (Fig. 6). Under both optimistic-response cases the 0.5 threshold is met at the shortest tested duration, 3 minutes, at every flow, because the assumed response is strong enough to offset even the proportionally largest deployment cost; the 0.8 threshold is never met at any flow or duration. Break-even behaviour therefore depends almost entirely on the response assumption, not on traffic volume, which contradicts the intuition that "enough traffic" justifies the sign.

![Fig. 6. P(dH < 0) versus work duration at three traffic flows, with 95 percent Wilson intervals (shaded) and reference lines at 0.5 and 0.8.](../model/outputs/figures/fig6_p_vs_duration.png){width=16cm}


## 3.3. Decision confidence classification

Applying the classification of Section 2.12 (Fig. 7): under standard response, all 49 scenarios classify as sign not supported (the upper 95 percent bound on P(dH < 0) is below 0.5 everywhere; maximum upper bound 0.161). Under optimistic response, all 49 scenarios classify as uncertain (lower bounds between 0.499 and 0.598, never reaching 0.8). No scenario in any assumption set reaches the sign-supported class. The Monte Carlo sample size is not the constraint: the Wilson intervals are approximately plus or minus 0.007 wide, so the uncertain classifications reflect genuine parameter uncertainty, not simulation noise.

![Fig. 7. Decision classification by scenario: sign not supported (upper 95 percent bound on P(dH < 0) below 0.5), uncertain, or sign supported (lower bound at or above 0.8).](../model/outputs/figures/fig7_decision_classification.png){width=16cm}


## 3.4. Sensitivity: what drives the harm, and what drives the decision (Objective 3)

On the harm scale, the encroachment rate dominates dH at every duration (PRCC +0.70, +0.67, and +0.66 at 3 minutes, 15 minutes, and 2 hours; bootstrap 95 percent CIs within plus or minus 0.03), followed by the response probability (about -0.19 to -0.20), mean baseline speed (+0.10), warning distance (+0.07 to +0.08), and walking speed (-0.07) (Fig. 8a). By the structural argument of Section 2.8, however, r_E cannot influence the sign of dH; its dominance on this scale reflects the two-decade prior on how much harm is at stake.

On the decision scale (PRCC against dH/r_E, Fig. 8b) the ranking changes: mean baseline speed leads (mean absolute PRCC 0.215), followed by warning distance (0.170), response probability (0.151), and walking speed (0.140), with handling time (0.085) and lateral reach (0.071) behind. The deploy-or-omit decision is therefore governed by the speed environment, the deployment geometry and effort, and the response share, while the encroachment rate governs the stakes. Higher mean speeds disfavour the sign in this low-speed setting: they raise severity on the deployment pathway, where the worker is closest to traffic and unprotected, by more than the responding minority's speed reduction recovers downstream.

![Fig. 8. Partial rank correlation coefficients with 95 percent bootstrap intervals at Q = 400 veh/h for three work durations: (a) against dH (harm scale); (b) against dH divided by the encroachment rate (decision scale).](../model/outputs/figures/fig8_prcc.png){width=16cm}


## 3.5. Robustness of the conclusions

*Input correlation.* Against the independent baseline, the plausible structure changed P(dH < 0) by at most 0.0004 anywhere on the grid; the stress structure by at most 0.012 (range 0.132 to 0.155); the behaviourally adverse structure by at most 0.006; and the optimistic-case counterparts by at most 0.003. Mean dH shifted by 17 percent at the median cell under the stress structure (driven by the imposed coupling between response share and response magnitude), and by more than 20 percent in 22 of 49 cells, with the largest relative shifts where the mean is near zero; no scenario changed direction or classification. Achieved rank correlations matched their targets within 0.01 for every imposed pair (pairs involving the fixed p_R are omitted in the optimistic case by design, Section 2.10). The conclusions are insensitive to plausible dependence, as the structural analysis predicts for correlations involving r_E and as the flat flow-response explains for the rest.

*Harm definition.* Restricting the occupant curve to frontal crashes changed P(dH < 0) by at most 0.0006; substituting the earlier-generation occupant curve, by at most 0.003. Under the fatality-only definition, the scale of harm at stake falls to roughly 31 percent of the MAIS 3+ scale, and P(dH < 0) moves to 0.132 to 0.156 (standard response) and 0.541 to 0.605 (optimistic response): slightly more favourable to the sign, because the steeper fatality curves concentrate more of the outcome in the fast tail, but with no scenario changing class. The conclusions do not depend on where the serious-harm threshold is drawn.

*Response allocation.* With the marginal response share held fixed, allocating responders to the fast tail of the speed distribution increased the magnitude of the mean benefit by about 20 percent, and allocating them to the slow tail reduced it by about 21 percent (optimistic case: +17 and -22 percent), confirming that severity nonlinearity concentrates the sign's value among faster drivers. The probability of net benefit, however, moved by at most 0.005 (standard) and 0.029 (optimistic), and at 3-minute optimistic scenarios fast-tail allocation pushed P(dH < 0) below 0.5 (minimum 0.479): the same fast streams that make responses more valuable also make the unprotected deployment window more dangerous. Who responds changes how much the sign helps when it helps; it does not rescue the deploy-or-omit decision.

## 3.6. Extension: the encroachment-reduction pathway (what it would take)

If the sign is credited with reducing work-period road departures, modest reductions change the decision (Fig. 9): a 15 percent reduction achieves both P(dH < 0) >= 0.5 and >= 0.8 at every flow for 3-minute jobs; 10 percent suffices for 6-minute jobs; 5 percent suffices for jobs of 15 minutes or longer. The step from "not supported anywhere" to "supported everywhere" is abrupt because a frequency reduction, unlike the speed pathway, benefits every iteration deterministically; once it outweighs the deployment exposure, it does so with near certainty. These results quantify the evidential burden the always-warn default implicitly carries: the default is justified within this scope only if a passive sign reliably prevents at least one road departure in twenty during the work period (one in seven for 3-minute jobs), and no direct evidence for such an effect exists in low-impact settings.

![Fig. 9. Minimum reduction in work-period road departures required for the sign to reach P(dH < 0) at or above 0.5 and 0.8, by work duration (requirements coincide for the two thresholds).](../model/outputs/figures/fig9_ke_requirement.png){width=10cm}


## 3.7. Verification and convergence

The deterministic verification case (all inputs fixed) matched hand calculation exactly for every exposure and frequency quantity (relative differences below 10^-15^) and to within 0.04 percent for harm quantities, the residual reflecting the near-degenerate speed distribution used in that check (Supplementary Table S2). Running estimates of P(dH < 0) at five representative grid points changed by no more than 0.0011 between 20,000 and 50,000 iterations (Supplementary Fig. S1), confirming the production sample size is sufficient.

# 4. Discussion

## 4.1. The hypothesis, and what is new here

The hypothesis was that the sign reduces net serious harm only when traffic volume and driver response are high, and that otherwise deployment exposure dominates. The evidence supports the response clause and rejects the volume clause. Driver response was decisive: moving the response share from the evidence-based range to the optimistic stress value shifted P(dH < 0) from about 0.13 to about 0.58, and no other lever in the primary model produced a comparable shift. The volume clause was not supported: within the tested range, more traffic does not favour the sign. P(dH < 0) was nearly flat in flow within every assumption set, because volume scales the downstream benefit and the deployment cost together. Duration, not volume, moves the balance, and it moves it against the sign as jobs get shorter.

The contribution is not the observation that controls carry deployment risk, which practitioners know, but the quantification: to our knowledge this is the first framework that places the deployment exposure of a TTM control and its downstream benefit in a single probabilistic accounting and returns the deploy-or-omit decision as a probability with stated confidence. Within its scope it converts an article of faith into a testable quantity, and the answer it returns for the low-impact case is that the faith is not currently supported by the evidence on passive-sign response. The result also has a decision-theoretic corollary: a control that is more likely to add expected serious harm than to remove it cannot be justified by any weighing of its costs against its safety benefit, because it fails before costs are counted; monetisation could only widen the deficit.

## 4.2. Why the speed pathway cannot carry the sign in this setting

Three structural features explain the result, all conditional on the model's assumptions. First, the hazard being managed is narrow by scope: with no merge, stop, or path change required of drivers, the behavioural mechanisms for which upstream warning has its clearest empirical support do not exist here, and what remains is a rare initiating event that a passive sign addresses only through speed. Second, severity nonlinearity concentrates the speed pathway's value in the fast tail of the traffic stream, and the response-allocation analysis quantifies what was previously an assumption-level worry: even when responders are drawn preferentially from the fast tail, the probability of net benefit barely moves, because the same fast streams raise severity in the unprotected deployment window. If, as the credibility and attention literature suggests, faster and less attentive drivers are the least likely to respond to a low-credibility passive sign [@charlton_conspicuity_2006; @steinbakk_speed_2019; @vignali_road_2019], the realistic case sits at or below the model's independent allocation. Third, the deployment cost is not merely additional minutes: it is a qualitatively worse exposure, on foot, nearer the edgeline, with attention on a manual task, and the lateral-decay structure of encroachment risk makes those metres decisive.

## 4.3. Stakes versus decision: reading the sensitivity results correctly

The two sensitivity scales separate questions that are often conflated. The encroachment rate dominates the harm scale but cannot influence the decision, because it multiplies every pathway equally; the decision is governed by the speed environment, the deployment parameters, and the response share. This has two practical consequences. For decision-making within this scope, better evidence on passive-sign response in low-cue settings, and leaner deployment methods, are worth more than better encroachment data; the response evidence bounds the achievable benefit, and deployment redesign attacks the cost side directly. For risk management more broadly, the encroachment rate still matters enormously: it sets the absolute scale of harm (here spanning three orders of magnitude across the prior), and therefore drives any question that involves absolute risk, prioritisation across sites, or value-of-information for measurement programmes. Both statements are conditional on the model's structure, and the extension shows how the picture changes if the sign is allowed to influence departure frequency: the required effects are small (5 to 15 percent), decisive if real, and unmeasured in this setting. Direct measurement of whether passive advance warning changes road-departure frequency in low-impact environments is the single study that would most change this decision.

## 4.4. Practical implications

For the class of work modelled here, the results do not support a default requirement to deploy a portable advance warning sign. Under evidence-based response assumptions the sign is more likely to add serious harm than to remove it in every tested scenario, and the typical penalty is largest exactly where such signs are most commonly deployed by rule rather than by judgement: short jobs. Three implications follow.

First, TTM guidance should separate compliance from protection. A control can be required and still fail to protect in a defined context; the framework provides the risk description that makes that distinction auditable. In jurisdictions whose safety duties are framed around what is reasonably practicable [@peace_reasonably_2017; @hse_reducing_2001], a mandatory control that increases net expected harm within a defined scope invites reassessment of the mandate for that scope, which is an argument for evidence-conditioned guidance, not for removing controls by default. Because the accounting is performed once for a defined class of work and embedded in guidance, the analytical effort sits with the guidance-setter, not the site: routine tasks whose individual scale could never justify site-by-site analysis still inherit an evidence-based default.

Second, the deployment pathway is a design target. Because deployment parameters rank among the decision drivers, changing how the warning function is delivered (vehicle-mounted or remotely activated devices, work sequencing that eliminates the on-foot walk) attacks the cost side of the trade-off and could flip the decision without any change in driver behaviour. These alternatives are not modelled here and should be evaluated with the same accounting before being mandated in their turn.

Third, where practice relies on advance warning as protective in low-impact settings, that reliance implicitly assumes the encroachment-reduction mechanism, and the extension prices that assumption: at least a one-in-twenty reduction in departures for typical jobs, one-in-seven for the shortest. Absent evidence of that size, the default should be treated as a convention under review rather than a safety fact.

## 4.5. Limitations

The conclusions are bounded by scope, and the boundary is the first limitation: they apply to low-impact, outside-lane, low-speed work with no required driver manoeuvre, and do not transfer to lane closures, higher speeds, constrained geometry, or any setting where warning addresses an actual required behaviour change. Within scope, several simplifications matter. Some findings are structural consequences of the accounting rather than empirical discoveries (the deployment pathway always adds harm under S1; the decision's invariance to r_E), and we have flagged them as such where they occur. Site geometry beyond lateral offset (curvature, sight distance, longitudinal extent) is held at fixed effective values; sites where geometry impairs perception may derive more value from warning even within the low-impact class. Excluded pathways (pedestrian and cyclist route deviation, handling injuries, sign-as-object impacts, multi-harm sequences) were reasoned to be second-order for MAIS 3+ accounting but were not modelled. The severity curves come from crash populations (German pedestrian cases, United States tow-away occupant crashes) that approximate rather than match the modelled configurations, although the harm-definition robustness suggests little sensitivity to curve choice. System-level behavioural effects, in particular the possibility that ubiquitous low-credibility warnings erode response to warnings in general, are outside the per-site accounting and would, if real, strengthen the case for selective use. Finally, the model quantifies a trade-off that experienced practitioners also weigh intuitively; a structured expert elicitation on the same question would provide an independent line of evidence, and convergence or divergence between elicited judgement and this model would each be informative [@cooke_experts_1991; @hemming_practical_2018].

# 5. Conclusions

For low-impact roadside work conducted entirely outside the travelled way at speeds of 50 km/h or below, does a portable static advance warning sign, placed and removed by a worker on foot, produce a net reduction in expected serious harm once deployment exposure is counted? Across 49 combinations of traffic flow and work duration, and across every robustness test applied, the answer from this model is: not reliably, and under evidence-based response assumptions, probably not at all.

With driver response sampled from the range supported by passive-sign field studies, the sign was more likely to increase than to reduce serious harm in all 49 scenarios (P(dH < 0) = 0.12 to 0.16), typically adding 0.1 to 11 percent to per-job serious-harm expectation, with the largest penalties for the shortest jobs and no rescue from faster deployment. Under a stress-test response share of 0.60, double the strongest passive-sign evidence, the sign became marginally more likely to help than harm (0.51 to 0.60) but never reached high confidence (0.80). These findings were insensitive to input correlation, to the harm definition, and to which drivers respond. The deploy-or-omit decision proved structurally independent of the encroachment rate, flat in traffic volume, and governed instead by the speed environment, deployment exposure, and the driver-response share; high-confidence benefit appeared only under an assumed 5 to 15 percent reduction in road departures for which no direct evidence exists in this setting.

Three evidence priorities follow directly: field measurement of driver response to passive advance warning where no visible hazard or required manoeuvre exists; direct tests of whether such warning changes road-departure frequency in low-impact environments; and time-and-motion data on on-foot deployment exposure. The framework itself is transferable to any temporary traffic management control whose deployment creates a competing exposure, and that is its wider point: risk-based practice implies exactly this kind of accounting. For controls applied millions of times per year, knowing whether the default helps or harms is not optional refinement; it is the analytical substance that risk-based language promises.

# CRediT authorship contribution statement {.unnumbered}

**D. Tilton:** Conceptualisation, Methodology, Software, Formal analysis, Investigation, Data curation, Visualisation, Writing - original draft. **J.D. van der Walt:** Methodology, Validation, Writing - review and editing, Supervision.

# Declaration of competing interest {.unnumbered}

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Funding {.unnumbered}

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

# Declaration of generative AI and AI-assisted technologies in the writing process {.unnumbered}

During the preparation of this work the authors used large language model tools (OpenAI ChatGPT and Anthropic Claude) to assist with analysis code development, reference management, and manuscript drafting and formatting. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

# Data and code availability {.unnumbered}

The complete model source code, configuration files, unit tests, per-iteration simulation traces, and the scripts that generate every figure, table, and numerical value in this article are openly available at https://github.com/L1013229/phd-adv-warning (archived at the time of submission). Supplementary material includes the full input specification (Table S1), the deterministic verification table (Table S2), convergence diagnostics (Fig. S1), the robustness result tables (Tables S3 to S5), and the sensitivity tables with bootstrap intervals (Table S6).

# References {.unnumbered}
