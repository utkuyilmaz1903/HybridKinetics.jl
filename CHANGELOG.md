# Changelog

All notable changes to HybridKinetics.jl are documented in this file.
Releases up to and including 0.14.0 were published under the package's former
name, BioDynaX.jl, and the entries below keep that name where it is what
shipped.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.19.0] - 2026-09-14

A measurement release aimed at one target: 0.18 showed that a substantially
better solution exists and the default training does not reach it. 0.19
measured how far away it is and what closes the distance, and found that the
distance training descends is not the one that matters. Given two, four and
eight times the iteration budget the fit closes the whole loss gap and
overshoots it, reaching a learned rate more accurate than the true-rate fit's
-- and recovers the same wrong support. One option is added, off by default;
no default, threshold or public function changes, and the 0.15 fingerprint
suite reproduces exactly.

### Added

- `restarts` on `TrainingConfig`: fit that many random initialisations of the
  neural term and keep the one reaching the lowest final loss, with every
  attempt's loss in the result's metadata. Defaults to 1, and restart one is
  the fit the package would have done on its own, so the option is a strict
  superset of today's behaviour. Not a default; the evidence for and against
  making it one is on the benchmarks page.
- `src/OptimisationGapStudy.jl`: a two-state reference training whose starting
  point and iteration budget the caller controls, the reference discovery run
  on a rate the caller supplies, the training loss at an arbitrary parameter
  vector, the loss along the straight line between two solutions, and a
  destruction rate estimated from the data alone.
- A "Closing the optimisation gap" section on the benchmarks page with the
  floor, the ceiling and what each approach recovers of the distance between
  them, including the approaches that recover nothing; matching paragraphs on
  the limitations page and in the README.

### Changed

- Nothing in the shipped path. The support F1 stays at 0.571, every threshold
  and seed is untouched, and the discovery code was not modified at all.

### Fixed

- `restarts` reached only `train_experiments_with_warmup`. The option lives on
  `TrainingConfig`, so `discover_unknown_terms` and `train_experiments`
  accepted it and then silently fitted once. `discover_unknown_terms` now
  wraps the warm-up and the joint fit together, which is the only correct
  place for the loop -- a restart redraws the whole neural block, so
  restarting the joint fit alone would discard the warm-up on every attempt
  after the first. `train_experiments` acts on it directly. `train_ude` raises
  instead of accepting it, because it fits one experiment and the lowest
  single-experiment loss is not the lowest joint loss. With `restarts` at 1
  every path is unchanged down to the last float.

### Measured, and not adopted

- A curvature penalty on the learned rate was not built: smoothing a rate a
  training already produced leaves the discovered support unchanged in all
  thirty runs measured and makes the rate worse, so there is no roughness for
  a penalty to act on.
- Pre-training the network to the scale of a data-estimated rate, with none of
  its shape, changes nothing — as the parameter-space measurement predicted.

## [0.18.0] - 2026-09-13

A measurement release. Nothing in the package's behaviour changes: no default
moves, no threshold moves, no public function changes, and the 0.15
fingerprint suite reproduces exactly. What is new is what is known about
three weaknesses of the existing workflow, each measured against a criterion
written down before the run, and the instruments that make those
measurements repeatable. The version is a minor bump rather than a patch
because three modules and a benchmark entry point are added to what ships,
not because behaviour changed.

### Added

- `src/ExtraTermsStudy.jl` and `benchmark/extra_terms_study.jl`: four
  explanations for the extra terms that survive the reference discovery,
  each measured on one training's samples so that only the discovery step
  varies. 2310 rows in `benchmark/results/extra_terms_study.csv`.
- `src/AdjacentBiasStudy.jl`: the adjacent-node bias under the joint fit, a
  fit started at the true rates, sequential training in both orderings and
  inverse-variance loss weighting, plus a scan of the held-out residual over
  the two learned rates' scale factors.
- `src/ReliabilityStudy.jl`: the spread of one run over ten random
  initialisations of the neural term; the warm-up length at zero, at the
  package's value and at twice it, with and without hidden observations; and
  an audit of every library-comparison run whose hybrid model will not
  resimulate, recording what the discovered candidate alone says about it.
- Three sections on the benchmarks page with the results and the criteria
  they are read against, and the matching paragraphs on the limitations page
  and in the README.

### Changed

- Nothing in the shipped path. The support F1 stays at 0.571, the sparsity
  threshold at 1e-3, the cross-term warning at 0.46 and the denominator floor
  at 1e-8, because every change that would have moved them failed the
  criterion written for it.

### Fixed

- Three defects in the new study code, each now pinned by a test: a frozen
  rate read through `sample_unknown_destruction` returns no gradient and the
  sequential fit silently trained nothing; a held-out residual averaged over
  hidden observations is `NaN` and made every masked run read as a
  divergence; and `Statistics.median` takes no `init` keyword, which stopped
  the adjacency summary as soon as a setting recorded no cross-term value.

## [0.17.2] - 2026-09-13

### Fixed

- Three call sites treated masked (unobserved, `NaN`) observations as
  observed; found in the p53–Mdm2 case study, where Mdm2 is unobserved, and
  first reported in the closed pull request #62:
  1. the warm-up `train_ude` in `discover_unknown_terms`
     (`src/DiscoverUnknownTerm.jl`, and `warmup_first_experiment` in
     `src/TrainingReuse.jl`) trained without the experiment's mask, so a
     `NaN` state gave a `NaN` warm-up loss and a corrupted start for the
     joint fit; `train_ude` now takes `mask` and applies it on every horizon
     slice;
  2. `fisher_information_matrix` and the production/destruction cosine of
     `production_destruction_tradeoff` (`src/Identifiability.jl`) masked only
     the residual and kept the Jacobian rows of unobserved entries; both are
     now computed over the observed entries only;
  3. `_regulator_grid` (`src/Recovery.jl`) took the extrema of every
     regulator sample, `NaN` included, so one missing value made the whole
     default discovery grid `NaN`; it now uses the observed, finite values
     and raises an error when there are none.
  Fully observed data are unaffected: the 0.15 fingerprint suite reproduces
  exactly and a rerun cell of the multi-term study gives identical rows. The
  p53–Mdm2 case-study page is recomputed with the fix and says what changed
  (training loss 0.0625 → 0.0689, collinearity 0.9997 → 0.9905, Fisher
  condition number 3,973 → 23,700, a differently shaped learned rate; the
  conclusions, a flagged scale and no accepted rational rate, are the same).

## [0.17.1] - 2026-09-12

### Fixed

- Stale single-unknown-term wording, left from before 0.16, replaced with
  what holds at 0.17: any number of unknown destruction terms on distinct
  nodes, two on the same node refused, an unknown production term out of
  scope, and the measured cost of several terms from the 0.16 study. Changed
  in the README ("Scope and limitations" and the overview), the
  documentation index, concepts, how-to, tutorial, getting-started and
  limitations pages, the `network_from_reactionsystem` docstrings, the
  contributing guide and the feature-request template. The guard-rail errors
  no longer name a release ("0.16 supports …") and say that an unknown
  production term is out of scope. The reference recovery protocol behind
  the benchmarks remains a single-term instrument, and the pages say so.

## [0.17.0] - 2026-09-12

### Changed (breaking)

- Three names are renamed and six are no longer exported, with no aliases:
  the package is not yet registered, so the old names disappear and a 0.16
  script fails at the first use with `UndefVarError`. The how-to page has the
  migration table ("Migrating from 0.16").

  | 0.16 | 0.17 |
  |---|---|
  | `discover_unknown_rate` | `regress_unknown_rate` |
  | `ude_system` | `ude_rhs` |
  | `UnknownTermsResult` | `DiscoveryRun` (`UnknownTermResult` stays the per-term type) |
  | `pack_parameters`, `positive_parameter`, `allocate_cache`, `RECOVERY_THRESHOLDS`, `EmptyMetadata`, `MetadataLike` | kept with their docstrings, not exported: `HybridKinetics.pack_parameters` or `using HybridKinetics: pack_parameters` |

  No number changes: the 0.15 fingerprint test still passes exactly on the
  recording environment.

### Added

- How-to: "Which right-hand side builder do I want?", a table for
  `build_ude_function` / `ude_rhs`, `compose_hybrid_rhs` and `export_rhs`; a
  sentence on which result type is which (`DiscoveryRun` is the whole run,
  `UnknownTermResult` one term).

### Changed

- The package description (`Project.toml`, README, documentation index,
  `CITATION.cff`) now reads "learn the unknown rate laws from time-series
  data, then recover them symbolically", since a network may carry several.
  0.17.0 is the version submitted to the General registry.

### Fixed

- The 0.15 fingerprint test could fail on a machine other than the recording
  one when a discovered coefficient sat on a five-digit rounding boundary
  (one CI runner printed `-2.2079` where the recording has `-2.208`, with a
  numeric drift of 3e-7). In the same-Julia-version tolerance mode the
  equation strings and the report are now compared with their numeric
  literals masked; the coefficients themselves are still compared at the
  tolerance, and the exact mode is unchanged.

## [0.16.0] - 2026-09-12

### Changed (breaking)

- `discover_unknown_term` and `report_unknown_term` are replaced by
  `discover_unknown_terms` and `report_unknown_terms`, which handle any
  number of unknown destruction terms (one per node). The old names still
  exist and raise an error that names the replacement and links the
  migration section of the how-to page; nothing runs with a changed meaning.
  `discover_unknown_terms` takes the same keywords except `term` and returns
  an `UnknownTermsResult`: `result.params`, `result.training`,
  `result.residuals` and `result.settings` stay at the top level, and the
  per-term identifiability, discovery, samples and extras are `result[:S]`
  (by node name), `result[1]`, or an element of `unknown_terms(result)`, each
  an `UnknownTermResult`. With one unknown term every number and the report
  text are identical to 0.15: `test/support/fingerprints_015.toml` records
  them from 0.15 on Julia 1.10.12 and the test suite asserts exact equality
  on that environment, equality within a scale-relative 1e-3 on another
  machine with the same Julia version, and only the RNG-independent parts on
  another Julia version (Julia's seeded random streams differ between
  versions, so the initial parameters differ there).
- `export_mtk_system(model; discovered = result)` substitutes the discovered
  rate of every term; a bare candidate is accepted only for a one-term model.

### Added

- Several unknown destruction terms on distinct nodes: `UnknownTerm(node;
  regulators, library)`, `BiologicalNetwork(...; unknown = [...])`,
  `unknown_terms(network)`, one network per term trained jointly, per-term
  discovery with per-term stability selection, and the per-term
  identifiability diagnostic with `production_param` (`:auto` finds each
  node's own production parameter).
- `cross_term_collinearity`, the pairwise cross-term diagnostic: the cosine
  between the trajectory sensitivities to a scale change of each term,
  reported per pair in the result and the report, with a warning above
  `CROSS_TERM_COLLINEARITY_THRESHOLD` (0.46, the value below which no run
  of the study showed a measurable cost of the extra term).
- Guard rails: two unknown terms on the same node and an unknown production
  term are errors that name the node and the milestone scope.
- `network_from_reactionsystem(rs; unknown = [...])` marks several reactions
  unknown; `symbolic(result; node)` selects a term.
- The multi-term study (`benchmark/multi_term_study.jl`) with the fixtures
  `build_two_term_separate_network`, `build_two_term_coupled_network` and
  `build_three_term_network`, resumable rows in
  `benchmark/results/multi_term_study.csv`. Its finding, on the benchmarks
  page: in 30 two-unknown runs the two terms never compensated for each
  other; on non-adjacent nodes the second term costs nothing measurable, on
  adjacent nodes the downstream term's learned rate comes out about 16% low
  and the cross-term diagnostic flags the case; three terms (noise 0 only)
  cost more, with the adjacent pair 1.4 and 2.2 times further from the
  truth and the third term's support lost in two of five seeds.

### Fixed

- `symbolic(result)` and `latexify(result)` name the discovered rate's
  variables by the unknown term's regulators among the dynamic states.
  Before, the regulator indices were applied to the full node list, so on a
  network with an `INPUT` node in front of the states (the p53 example, where
  `DNA_Damage` is node 1) the Mdm2-regulated rate was written as a function
  of p53. The test suite now checks that network.

## [0.15.0] - 2026-09-08

### Changed

- The package is renamed from BioDynaX to **HybridKinetics**. The module, the
  package file, the extensions, the repository and the documentation URL all
  follow the new name; the UUID is unchanged. Update your code to
  `using HybridKinetics`, and install from the new URL:

  ```julia
  using Pkg
  Pkg.add(url = "https://github.com/utkuyilmaz1903/HybridKinetics.jl")
  ```

  An environment that already tracks the package under the old name should
  remove it and add it again from the new URL; the recorded name no longer
  matches the one in the package's `Project.toml`.
- New one-line description: "Hybrid kinetic models for small biochemical
  networks: learn the one unknown rate law from time-series data, then recover
  it symbolically."
- The environment variables that select the heavy test tier and the example
  smoke runs are renamed to `HYBRIDKINETICS_TEST_HEAVY` and
  `HYBRIDKINETICS_SMOKE`.
- No numerical behaviour changed: thresholds, seeds, initial-condition counts,
  the training and holdout split, optimizer settings and the discovery library
  are untouched, and the test suite is unchanged apart from the new name.

## [0.14.0] - 2026-09-06

### Added

- Catalyst input: `network_from_reactionsystem(rs; unknown)` (extension
  `BioDynaXCatalystExt`, loaded with `using Catalyst`) converts a
  `ReactionSystem` into a `BiologicalNetwork` with the same species, the
  known rate laws compiled to the matching terms (a parameter, `k * Y`,
  `hill(Y, v, K, n)`, `mm(Y, v, K)`), and the reaction named by `unknown`
  (index or `description` metadata) as the one unknown destruction term;
  other rates raise an error naming the reaction. The tutorial network
  converted from Catalyst simulates and discovers exactly as the fixture
  written by hand, and its ModelingToolkit export equals Catalyst's own ODE
  system (tests). How-to section "Starting from a Catalyst model".
- Symbolic output: `symbolic(candidate, names)`, `symbolic(result, names)`,
  and `symbolic(result::UnknownTermResult)` (extension
  `BioDynaXSymbolicsExt`, `using Symbolics`) return the discovered rational
  rate as a `Symbolics.Num`; with `using Latexify`, `latexify` of a result
  or candidate renders it (extension `BioDynaXLatexifyExt`).
- `BioDynaX.export_mtk_system(model; discovered)` replaces the placeholder
  of the unknown term with the discovered rate, so the completed model can
  be handed to ModelingToolkit and OrdinaryDiffEq.
- Documentation for SciML readers: a "Where BioDynaX fits" page comparing
  the package with a plain universal differential equation,
  DataDrivenDiffEq with DataDrivenSparse, and Catalyst alone; the
  getting-started page starts from a Catalyst model; ColPrac adopted in
  CONTRIBUTING.md with the ColPrac and SciML Style badges.

### Changed

- The states of the `ODESystem` returned by `BioDynaX.export_mtk_system`
  are named after the network's nodes (`S(t)`, `R(t)`) instead of `x1(t)`,
  `x2(t)`.
- `Symbolics` and `Latexify` are weak dependencies; Catalyst,
  ModelingToolkit, Symbolics, and Latexify are test dependencies.

### Removed

- The names deprecated in 0.12: `BioDynaX.UNIQUE_CLAIM_PROTOCOL`,
  `BioDynaX.unique_claim_experiment_set`, and
  `BioDynaX.unique_claim_discovery_config`. Use `REFERENCE_PROTOCOL`,
  `reference_protocol_experiment_set`, and
  `reference_protocol_discovery_config`. (`report` was renamed to
  `report_unknown_term` before 0.11.0 shipped and never had an alias.)

## [0.13.0] - 2026-09-06

### Added

- A second case study on measured data, the p53–Mdm2 loop in single MCF7
  cells after 4 Gy of ionizing radiation (`examples/p53_mdm2/`:
  `download_data.jl` with SHA-256 checks, `preprocess.jl`,
  `run_case_study.jl`, `plot_case_study.jl`; the "Case study: p53–Mdm2"
  page). The traces (Stewart-Ornstein and Lahav 2017, from the CC BY 4.0
  CODEX deposit on Mendeley Data) carry p53 only, so Mdm2 is an unobserved
  state and the case study is partially observed; the page records every
  data source tried, in order. The data are downloaded by the script and
  do not ship with the package; nothing runs in the tests or CI.
- `discover_unknown_term(...; regulator_grid = ...)`: a vector of regulator
  values, or a function of the trained model called after training, on
  which the learned rate is sampled; `nothing` (the default) keeps the grid
  of the observed regulator. This is how a regulator that is never observed
  is handled.
- Two model variants for the laccase/ABTS case study
  (`examples/laccase_abts/run_model_variants.jl`): product inhibition
  through the curve's initial substrate as a second state, and an
  unobserved enzyme-activity state with first-order inactivation; their
  outcome is on the laccase page.

### Fixed

- The residuals and the identifiability diagnostic of `discover_unknown_term`
  use each experiment's observation mask, so unobserved entries (`NaN`) no
  longer make them `NaN`; `production_destruction_tradeoff` gains a `mask`
  keyword (all entries observed by default). Fully observed experiments
  give the same numbers as before.

## [0.12.0] - 2026-09-06

### Added

- A sample coordinate design for the four-state fixture of the library
  comparison study: `design = :varying` spreads the target state S over
  the range observed in the training experiments (as the two-state fixture
  already did), `design = :constant` keeps it at 0.4 as the library check
  does. `designed_trained_graph_local_coordinates`,
  `evaluate_trained_graph_local`, `library_comparison_run`, and
  `library_comparison_study` take the keyword; study rows and CSV files
  gain a `design` column (files written by 0.11 still read); the study
  script takes `--design`.
- Benchmarks page: the S-varying run of the four-state study on the same
  15 trained models, which settles the cause of the recall 0.5 of the
  library-check configuration (the constant S column of the sample
  design, not the presence of the target state in the library).
- A test that a network declaring its unknown term as a reaction alone has
  the same graph parents as one that also declares the edge.
- The first case study on measured data, `examples/laccase_abts/`
  (`download_data.jl`, `preprocess.jl`, `run_case_study.jl`,
  `plot_case_study.jl`) and the "Case study: measured data" page: nine
  substrate-depletion progress curves of the laccase-catalysed oxidation of
  ABTS from the EnzymeML paper's repository, one observed state, the
  one-call workflow with the reference defaults. The data are downloaded by
  the script with a checksum and do not ship with the package; the run is
  not part of the tests or CI. On this dataset no rational rate was
  accepted (the denominator-safety check rejected the candidate); the page
  reports the trained model, the learned rate, and why.

### Changed

- The library comparison study's default design for the four-state fixture
  is `:varying`. With it the library-check configuration (`study` variant)
  recovers the true support in 15 of 15 runs with the graph-local library
  (0 of 15 with S constant); the headline reference-configuration rows are
  identical under both designs. `design = :constant` reproduces the 0.11
  rows; `evaluate_trained_graph_local` keeps its constant design and its
  recorded outputs. `local_basis` is unchanged.
- The interaction graph of a `BiologicalNetwork` holds, in addition to the
  declared edges, an edge from each regulator of every reaction with
  `known = false` to the species that reaction changes. A network that
  declares its unknown term as a reaction only (the tutorial's
  `build_hill_recovery_network(known = false)`) therefore gets a graph-local
  library that contains its regulators instead of the target state alone.
  Networks that declare their edges explicitly, including every benchmark
  fixture, are unchanged (graph parents, discovered supports, and study
  rows verified identical). Regulators of known reactions add no edges.

### Added (extra-term study)

- Three settings of the two-state fixture of the library comparison study,
  off by default, for the study of the extra terms `1` and `R` of the
  reference protocol: `fixed_production` (the production rate frozen at
  its true value during training), `normalise_rate` (the learned rate
  samples divided by the fitted production rate before discovery), and
  `n_sample_points` (the density of the regulator grid), with an
  `on_discovery` callback that receives every discovery result (the study
  script records the stability-selection frequencies with `--pruning`).
  Benchmarks page: none of the three removes the terms; the hypothesis
  that they absorb the production/destruction scale is not supported.

### Deprecated

- `BioDynaX.UNIQUE_CLAIM_PROTOCOL`, `BioDynaX.unique_claim_experiment_set`,
  and `BioDynaX.unique_claim_discovery_config` are renamed to
  `REFERENCE_PROTOCOL`, `reference_protocol_experiment_set`, and
  `reference_protocol_discovery_config`. The old names forward to the new
  ones with a deprecation warning and are removed in 0.13.

### Changed (internal names)

- Unexported names that carried milestone or review labels are renamed
  outright: `unique_claim_*` to `reference_protocol_*`,
  `UniqueClaimFingerprint` to `ReferenceProtocolFingerprint`,
  `UniqueClaimProtocolRow` to `ReferenceProtocolRow`; `M4B_PROTOCOL`,
  `M4B_SMOKE`, `M4B_SCOPE_PLAN`, and `m4b_*` to `TRAINED_LIBRARY_COMPARISON`,
  `TRAINED_LIBRARY_COMPARISON_SMOKE`, `TRAINED_LIBRARY_COMPARISON_SCOPE_PLAN`,
  and `trained_library_comparison_*`; `format_q3_q4_side_by_side` to
  `format_diagnostics_side_by_side`; `recovery_suite_section_is_gated`,
  `recovery_suite_all_sections_gated`, the row field `gated`, and the
  `*_gate_row` helpers to `*_is_checked`, `*_all_sections_checked`,
  `checked`, and `*_check_row`; `recovery_hard_named_kpi_contract` and
  `remapped_two_regulator_contract_holds` to `recovery_hard_named_kpi_spec`
  and `remapped_two_regulator_spec_holds`; the `hill_from_nn` row field
  `closed` to `not_attempted`. `src/UniqueClaim.jl` is
  `src/ReferenceProtocol.jl`, `src/CompilerContract.jl` and
  `src/DataGenContract.jl` are `src/CompilerSpec.jl` and
  `src/DataGenSpec.jl`, `test/run_m4_b_protocol.jl` is
  `test/run_trained_library_comparison.jl`, and
  `benchmark/allocation_gate.jl` is `benchmark/allocation_check.jl`.
- The source-reading checks and consistency matrices (`*_source_holds`,
  `*_index_holds`, `*_honesty_*`, `*_sensealg_honesty`, and the fixture
  rows only they used, 76 functions and one constant) moved from `src/` to
  `test/support/source_checks.jl`; they read the package source at test
  time and are test helpers, not package code. The public name list is
  locked in one place, `LOCKED_PUBLIC_EXPORTS`, and the test reads it
  instead of keeping a copy.

### Fixed

- The warm-up training of `discover_unknown_term` now uses the learning
  rate, gradient clip, constraint, solver, and frozen parameters of the
  `training` config (it used only its Adam iterations and log interval);
  the defaults are unchanged.

## [0.11.0] - 2026-09-06

### Added

- `discover_unknown_term(network, experiments; ...)`, a one-call entry
  point that builds the hybrid model, trains it (warm-up on the first
  experiment, then Adam 100 and BFGS 50), samples the learned rate on the
  regulator grid of the training experiments, discovers a rational rate,
  computes the identifiability diagnostic and the residuals on the first,
  the training, and the held-out experiments (`holdout = 2` by default), and
  returns an `UnknownTermResult`; `report_unknown_term(result)` gives the four-section
  report and `show` prints it. It calls the same functions as the reference
  example, in the same order, with the same defaults, and a test checks
  that its result matches the chained calls field by field.
- An optional stability-selection stage for implicit discovery:
  `discover_unknown_rate(...; stability_selection = StabilitySelection())`
  (also on `discover_equations` and `discover_unknown_term`) resamples the
  training rows of the regression, repeats the thresholded fit on every
  resample, and drops candidate terms selected in fewer than a fraction `τ`
  of the resamples. Terms are never added. `stability_selection_report` and
  `format_stability_selection` show the selection frequency of every library
  term. Off by default; with it off, discovery output is unchanged.
- The library comparison study (`BioDynaX.library_comparison_study`,
  unexported): the trained-model library comparison over seeds and
  observation-noise levels, scoring the graph-local, global, and wrong-graph
  libraries on the same trained model, on the four-state fixture and on the
  two-state reference network, with discovery variants that isolate the
  library construction and the bootstrap. `benchmark/library_comparison_study.jl`
  runs it, appends each row to a CSV as it finishes, and resumes from that
  file; `benchmark/plot_library_comparison.jl` draws the figure. The smoke
  configuration runs in the default test suite and the study runs in the
  weekly heavy CI job.
- `evaluate_trained_graph_local` accepts `seed`, `noise_σ`, and
  `stability_selection` keywords; `build_hill_recovery_network` accepts
  `parent`. All defaults reproduce the previous behaviour.
- `format_protocol_result` accepts `residual_train` and `residual_holdout`
  (printed only when given).

### Changed

- Documentation: the README quick start and the first tutorial section use
  `discover_unknown_term`; the tutorial keeps the step-by-step chain as
  "What the one call does". The benchmarks page has a "Library comparison
  study" section with the figure, the summary tables for both networks, the
  investigation of the recall gap between the study and the reference
  protocol, and the stability-selection comparison; the concepts page
  explains the pruning stage; the how-to page has a one-call recipe.

## [0.10.0] - 2026-09-05

This release turns the repository into a publicly presentable package:
rewritten README and documentation, plain-language printed reports, a
formatted tree, a faster default test suite, working extensions, and a
reorganised continuous-integration workflow. Scientific behaviour (numerics,
thresholds, seeds, protocol settings, library construction) is unchanged.

### Added

- Held-out validation for the reference recovery protocol. Nine initial
  conditions are generated once; the model is trained on the first seven,
  and the residual and the neural destruction-rate error are reported on
  the remaining two. These held-out numbers are reported evidence only and
  are not part of the acceptance criteria.
- A practical functional-identifiability diagnostic
  (`BioDynaX.assess_functional_identifiability`, unexported). Independent
  training restarts on five fixed seeds are compared on a shared domain;
  every restart, including failures, is reported. It is a diagnostic, not
  an acceptance criterion and not a structural identifiability proof.
- Trained-model library validation (`BioDynaX.evaluate_trained_graph_local`,
  unexported): one trained model, its learned destruction rate sampled once,
  and symbolic discovery run with the graph-local library, a global library,
  and a wrong-graph library on the same samples. A fast version runs in the
  default tests; the full protocol is `test/run_m4_b_protocol.jl`.
- Observed-trajectory sampling context (`BioDynaX.TrajectoryOccupancy`,
  unexported) for train or held-out experiments.
- Training reuses one compiled model and one solver session across initial
  conditions; the Adam state from the first-experiment warm-up is carried into
  the joint training.
- Discovery library evaluation reuses grow-only workspaces and streams the
  implicit design matrix in row blocks, reducing peak memory for large sample
  counts.
- `generate_experiment_set` compiles the ground-truth model once and generates
  every initial condition from that model through `SciMLBase.ODEProblem`.
- Experiment fingerprints, batched training, and checkpoint/resume
  (`BioDynaX.save_checkpoint`, `BioDynaX.resume_training`, unexported).
- Parameter packing collects `CustomKineticMetadata.rate_param`;
  `unpack_parameters` inverts `pack_parameters`.
- `denominator_violation_count` is split across training, validation, and the
  orthant domain grid.
- The compiler reindexes kept neural destruction terms to `1:n`, so a skipped
  duplicate unknown edge no longer leaves a gap that crashed `ude_system`.
- Aqua runs with its default `test_all` surface, and a separate `standards` job
  runs JET on `train_ude`, `discover_unknown_rate`, and `compose_hybrid_rhs`.

### Changed

- Recovery paths that require exactly one unknown destruction term check this
  before training and fail early on zero or two unknown terms.
  `validate_network` itself does not count unknown terms.
- The printed recovery report has four sections (identifiability, fit,
  discovery, reproduction) whose order matches the `protocol_result` fields.
  Its explanatory lines are now plain English, floating-point values are
  printed with four significant digits, and the last discovery line reads
  `acceptance_criteria: ...` instead of an internal label.
- The functional-identifiability report and the side-by-side scale-warning
  report use plain section headers.
- The documentation is restructured into ten pages: home, getting started,
  tutorial, concepts, how-to recipes, benchmarks, API reference, extensions,
  scope and limitations, and changelog. Internal design notes and lock lists
  are gone; every remaining code block either runs in the docs build or is
  marked illustrative.
- Source comments and docstrings no longer use internal milestone and
  question labels.
- The two multi-minute training-loop testsets in
  `test/test_experiment_checkpoint.jl` run only with
  `BIODYNAX_TEST_HEAVY=1`; the default `Pkg.test()` is correspondingly
  shorter. No test was removed.
- Continuous integration: the default workflow runs the test suite on Julia
  1.10 and the latest 1.x with coverage, a JuliaFormatter check over the whole
  tree, Aqua and JET, the allocation check, the compat check, and the docs
  build (deployed from `main` and version tags). The trained-model recovery
  protocol, the trained-model library comparison, the heavy test tier, and
  the JET standards run in a weekly scheduled job that can also be started by
  hand.
- The whole tree is formatted with JuliaFormatter (SciML style).
- Every benchmark script starts with a header stating its purpose, runtime,
  output, and how to run it.
- Mechanistic models switch from `BacksolveAdjoint` to `InterpolatingAdjoint`
  when the number of observations exceeds 64. Neural terms always use
  `InterpolatingAdjoint`.
- CompatHelper and TagBot request the write scopes they need.

### Fixed

- The ModelingToolkit extension never loaded (invalid syntax and an
  undeclared dependency on Symbolics) and the SBML extensions could not be
  precompiled because they redefined package methods. All extensions now load;
  `BioDynaX.export_mtk_system` was verified against ModelingToolkit 11.
- `benchmark/scale_basis.jl` could not run because it used unexported names
  without importing them.
- `ModelingToolkit` compat widened to `"9, 10, 11"`; with SciMLBase 3 only
  ModelingToolkit 11 resolves.
- The printed-report consistency check compares support recall and F1 with
  the same rounding that prints them.

### Removed

- Editor and agent configuration (`.cursor/`), internal planning documents
  (`docs/research/`), the debug runner `scripts/run_discovery.jl`, the root
  result figure, and `NEWS.md` (this file is the single changelog).
- Documentation pages that described milestone plans, CI lock lists, and
  API-freeze checklists rather than package behaviour.
- Internal helpers whose only purpose was to read documentation files and
  assert their wording, and the tests that called them.
- Fifty unreferenced leftover helpers from the same layer (source-path,
  test-path, fixture-matrix and lock-row functions that nothing called).

## [0.9.2] - 2026-08-14

### Added

- Six-state graph-prior fixture and a wrong-graph negative control
  (`:six_state`, `:six_state_wrong_graph`) in the recovery suite.
- `benchmark/recovery_seeds.jl` (analytical recovery on five seeds; optional
  `--ude` for the trained-model protocol), `benchmark/noise_grid.jl`, and
  `benchmark/ude_f1_attempt.jl`.
- GitHub Pages workflow that publishes the `gh-pages` branch.
- Practical production/destruction scale warning
  (`BioDynaX.production_destruction_tradeoff` and
  `BioDynaX.report_production_destruction_tradeoff`). The Hill recovery CI job
  requires the warning to be raised.
- `TrainingConfig.frozen_phys` to pin known production parameters during
  training.
- Scale-normalized discovery on the same monomial library
  (`normalize_destruction_samples`).
- Partial-observation path: subsampled destruction-rate samples feed the hybrid
  residual. Training on missing states is not supported.

### Changed

- The export list is reduced to the core types and the functions used in the
  tutorial. Recovery fixtures, Fisher identifiability, and the GPU, SBML, and
  ModelingToolkit extensions are unexported.
- The example builds its network with `ReactionSpec` and `HillMetadata`.
- `Zygote.@ignore` replaced by `ChainRulesCore.ignore_derivatives`.
- `train_experiments` refines the joint loss over every experiment with BFGS;
  Adam may still be minibatched.
- Nested subset selection scores BIC on the fit set and uses held-out residual
  sum of squares only as a safety filter.
- Neural-network parameters are promoted to `Float64` after `Lux.setup`.
- Documenter deploys only from `main`.

## [0.9.1] - 2026-08-13

### Added

- Nested subset pruning and predicted-output implicit refit on the same
  monomial library. Analytical Hill recovery at 0.5% noise reaches a combined
  support F1 of 0.99.
- Graph-local versus global library baseline script
  `benchmark/sindy_baseline.jl` (DataDrivenSparse optional).
- Practical production/destruction trade-off report, three-state graph-prior
  test, partial-observation mask, two-regulator competitive unknown term, and a
  synthetic repressilator fixture.
- TagBot, CompatHelper, Documenter `deploydocs`, and `CONTRIBUTING.md`.

### Changed

- `RECOVERY_THRESHOLDS.nn_rate_rmse` tightened to 0.12.
- Unknown-term training uses nine initial conditions and a 50-point horizon.

## [0.9.0] - 2026-08-13

### Added

- Recovery suite (`BioDynaX.run_recovery_suite`, `benchmark/recovery_suite.jl`)
  with thresholds for linear, Michaelis-Menten, Hill, and competitive parameter
  error, an executable discovered right-hand side, and the graph-local versus
  global library comparison.
- `DiscoveryRetcode` on `DiscoveryResult`; `strict = true` rethrows instead of
  returning a failed result.
- CSV experiment I/O (`experiment_from_csv`, `write_experiment_csv`).
- The unknown-inhibition example and the Documenter tutorial, how-to, and API
  pages.
- Optional `DataDrivenSparseSTLSQ` backend (requires DataDrivenSparse.jl) and
  optional `import_sbmltoolkit_network` (requires SBMLToolkit and Catalyst).
- `local_basis(...; scope = :graph | :global)` for library comparisons.
- `CITATION.cff`.

### Changed

- Default synthetic data uses the compiled mechanism. Pass
  `generator = :hill_p53_fixture` for misspecification studies.
- `ude_system(::SVector)` dispatches through the StaticArrays kernel below
  `STATIC_STATE_THRESHOLD`.
- SBML import no longer guesses Michaelis-Menten kinetics from type names;
  explicit kinetic laws compile as unknown neural terms.

### Fixed

- Julia 1.12 docs and quality load path: `SciMLBase` is a direct `docs/`
  dependency.

## [0.8.0] - 2026-08-12

### Added

- Streaming library evaluation (`evaluate_library_range!`, `LibraryChunks`,
  `each_library_chunk`) and blocked STLSQ with buffer reuse.
- Denominator domain safety: an orthant stress grid (`domain_samples`) checked
  on training, validation, and domain samples.
- Raw-data discovery: `estimate_derivatives` and
  `discover_equations(X, times, network)` without a trained model.
- Equation export: `equation_to_latex`, `equation_to_function`, `export_rhs`.
- Model selection: `information_criterion`, `score_candidate`,
  `select_discovery_config`.

## [0.7.0] - 2026-08-06

### Added

- Optimization.jl training path (`build_optimization_problem`,
  `solve_optimization`, `train_via_optimization`).
- `recommend_sensealg` and `auto_sensealg` selecting `BacksolveAdjoint` for
  mechanistic models and `InterpolatingAdjoint` for neural terms.
- Typed `HorizonCurriculum` for horizon training schedules.
- Per-experiment weighting through `experiment_weight` and
  `experiment_noise_scale` metadata.
- Network size presets `:small`, `:medium`, `:large` for `build_ude_nn`.

## [0.6.0] - 2026-08-06

### Added

- Compiler support for `SATURATION` (Michaelis-Menten) and `CUSTOM_KINETIC`
  reactions with `SaturationMetadata` and `CustomKineticMetadata`.
- Multi-head neural networks (`MultiHeadNetwork`, `build_ude_nn(rng; n_heads)`).
- Stoichiometric scaling on all mechanism terms.
- StaticArrays fast path for networks with at most four states.
- Weak-dependency extensions for ModelingToolkit (`export_mtk_system`) and SBML
  (`import_sbml_network`).

## [0.5.0] - 2026-08-06

### Added

- Fisher-information identifiability (`assess_identifiability`,
  `trajectory_jacobian`, `parameter_credible_intervals`,
  `estimate_parameter_uncertainty`).
- `TrainingRetcode` enum and gradient-norm convergence diagnostics.
- Discovery uncertainty reports and multi-trajectory
  `discover_equations(params, model, set::ExperimentSet)`.
- Benchmark networks including a repressilator.

### Changed

- `TrainingResult.retcode` is a `TrainingRetcode` enum (was a `Symbol`).

## [0.4.0] - 2026-08-06

### Added

- SciML integration: `SciMLBase.ODEProblem(model, u0, tspan, p)`,
  `build_ude_function`, `auto_sensealg`, `default_solver_config`, and
  `SciMLBase.solve(model, ...)`.
- Typed kinetic metadata structs with backward compatibility for
  `Dict{Symbol,Any}` metadata.
- `ExplicitSTLSQ` discovery backend and `ExplicitCandidate` results.

### Changed

- `predict_ude` routes through `SciMLBase.ODEProblem` for both AD policies.
- `RunMetadata` defaults to `BioDynaX.PACKAGE_VERSION`.

[Unreleased]: https://github.com/utkuyilmaz1903/HybridKinetics.jl/compare/v0.19.0...HEAD
[0.19.0]: https://github.com/utkuyilmaz1903/HybridKinetics.jl/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/utkuyilmaz1903/HybridKinetics.jl/compare/v0.17.2...v0.18.0
[0.17.2]: https://github.com/utkuyilmaz1903/HybridKinetics.jl/compare/v0.17.1...v0.17.2
[0.17.1]: https://github.com/utkuyilmaz1903/HybridKinetics.jl/compare/v0.17.0...v0.17.1
[0.17.0]: https://github.com/utkuyilmaz1903/HybridKinetics.jl/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/utkuyilmaz1903/HybridKinetics.jl/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/utkuyilmaz1903/HybridKinetics.jl/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/releases/tag/v0.4.0
