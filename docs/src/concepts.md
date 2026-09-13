# Concepts

## Model form

Every HybridKinetics model is a production-destruction system

```math
\frac{du_i}{dt} = P_i(u, p, t) - D_i(u, p, t)\,u_i, \qquad P_i, D_i \ge 0 .
```

`compile_mechanism` lowers reactions and edges into production and
destruction terms. Known kinetics stay symbolic and compiled: mass action,
linear decay, Hill, Michaelis-Menten saturation, competitive binding, an
input drive, or a custom rate expression. Any number of destruction terms,
on distinct nodes, may be marked `known = false`; each becomes a
`NeuralDestructionTerm`, a small Lux multilayer perceptron with a softplus
output that maps the regulator concentrations to a non-negative rate (two
unknown terms on the same node and an unknown production term are refused).
Each unknown term is multiplicative
(`D_i(u) * u_i`), not an additive residual on the right-hand side.

Physical parameters are stored in raw form and mapped through a softplus so
that they stay positive; `HybridKinetics.positive_parameter` applies the map and
`HybridKinetics.pack_parameters` builds the `ComponentVector` with `phys` and `nn` axes that
every solver and trainer takes. States pass through `max(0, x)` inside the
right-hand side. These are architectural choices that keep the model in the
positive orthant in practice; they are not a positivity theorem.

`validate_network` checks names, bounds, stoichiometry, and metadata. It does
not count unknown terms: a network with zero or several unknown destruction
terms still compiles. The recovery workflow checks the count separately
(`HybridKinetics.assert_single_unknown_destruction`) and raises an error for
anything other than one.

## Networks and metadata

A `BiologicalNetwork` holds `NodeSpec`s and either `ReactionSpec`s (with
stoichiometry and regulator indices) or `EdgeSpec`s (source, target, kind).
Each carries a typed metadata struct naming its rate parameters:

| Metadata | Kinetics | Parameters |
|---|---|---|
| `MassActionMetadata` | rate proportional to the product of the regulators | `rate_param` |
| `LinearDecayMetadata` | first-order decay | `rate_param` |
| `HillMetadata` | `vmax * r^n / (K^n + r^n)` | `vmax_param`, `k_param`, `hill_order` |
| `SaturationMetadata` | Michaelis-Menten saturation | `vmax_param`, `k_param` |
| `CompetitiveMetadata` | competitive binding of two regulators | see the API page |
| `InputDriveMetadata` | production driven by an input node | `rate_param`, `input_param`, `input_node` |
| `CustomKineticMetadata` | user-supplied rate expression | `rate_param` |
| `HybridKinetics.EmptyMetadata` | no parameters (used for unknown edges) | none |

`Dict{Symbol,Any}` metadata is still accepted for backward compatibility.

```@example concepts
using HybridKinetics
ReactionSpec(name = :decay, stoichiometry = Dict(1 => -1.0), regulators = Int[],
    metadata = LinearDecayMetadata(rate_param = :k))
```

## Experiments

An `Experiment` is one time series: `times`, an `observations` matrix
(`nstates x n_points`), the initial state `u0`, an optional `mask` of
observed entries, and metadata such as per-experiment weights. An
`ExperimentSet` is a list of experiments with the state names.
`experiment_from_csv` and `write_experiment_csv` read and write one
experiment; `generate_experiment_set` produces synthetic sets from a known
network by compiling the ground-truth model once and integrating every
initial condition from it.

## Training

`train_ude` fits one experiment and `train_experiments` fits a set. Both
minimize the trajectory mean-squared error with Adam and then, optionally,
refine with BFGS on the full loss. A `TrainingConfig` sets the iteration
counts, learning rate, gradient clipping, a horizon curriculum
(`HorizonCurriculum`, which trains on a growing fraction of the time span),
the constraint strategy (`StructuralPositivity` or
`AugmentedLagrangianConfig`), and `frozen_phys`, a list of physical
parameters held fixed. Gradients come from SciMLSensitivity adjoints:
`auto_sensealg` picks `BacksolveAdjoint` for small purely mechanistic models
with at most 64 observations and `InterpolatingAdjoint` otherwise; a model
with a neural term always uses `InterpolatingAdjoint`.

Training reuses one compiled model and one solver session across initial
conditions. The optimizer state is kept on the result
(`TrainingResult.diagnostics.optimizer_state`), so a first-experiment warm-up
can hand its Adam state to the joint fit, and checkpoints can resume without
recompiling (see [How-to](howto.md)).

**Parameter vector.** The packed parameters are a `ComponentVector` with two
blocks: `p.phys`, the physical kinetic parameters in the order of
`parameter_schema(model).phys_names`, stored unconstrained and mapped to
positive values by `HybridKinetics.positive_parameter`, and `p.nn`, the network weights.
With one unknown term `p.nn` holds the layers of that network directly
(`layer_1`, `layer_2`, `layer_3`), as in every release since 0.1; with
several, `p.nn` holds one block per term, `head_1`, `head_2`, …, in the
order of `unknown_terms(network)`, each with the same three layers, so
`result.params.nn.head_2.layer_1.weight` addresses the first layer of the
second term. The same architecture, initialisation and optimizer settings
apply to every term; the initial weights of the first term are the ones a
single-term model would draw from the same seed, and the further terms draw
next from the same generator. Checkpoints store the whole vector, so
saving and resuming work unchanged with several terms.

**Warm-up with several terms.** The first-experiment warm-up trains every
term at once, exactly as it trains one: there is no per-term stage. A staged
warm-up (one term at a time) was not adopted because it would change the
single-term path and because nothing in the two-term study needed it; the
option remains open if a fixture calls for it.

**How long the warm-up is.** Its length was measured at zero, at the
package's value and at twice it, on fully observed data and with half of one
state's observations hidden. The three settings differ by less than the
spread that ten random initialisations of the same data produce, so the
current length is a reasonable default rather than a tuned one; dropping the
warm-up entirely is the only setting in which two seeds disagree about the
discovered support ([Benchmarks](benchmarks.md#How-much-one-run-varies)).

## Identifiability diagnostic

With observed concentrations alone, a production rate and the scale of the
destruction term that follows it trade off against each other.
`HybridKinetics.production_destruction_tradeoff` quantifies this for a trained
model on one trajectory:

- the Fisher information over the physical parameters (neural weights
  excluded) and its condition number, from a finite-difference trajectory
  Jacobian;
- the cosine between the trajectory sensitivity to the production parameter
  (`k_prod` by default) and the sensitivity to a multiplicative rescaling of
  the unknown term.

`unidentifiable_edge` is `true` when the condition number is at least `1e6`
or the cosine is at least 0.95. `coefficients_are_biological_constants` is
its negation. The diagnostic is local (one trajectory, one parameter point)
and asymptotic; it is not a structural identifiability proof. A raised
warning does not stop the workflow. In the reference protocol it is required
output: the ambiguity must be reported, not hidden.

A second, unexported diagnostic, `HybridKinetics.assess_functional_identifiability`,
trains the unknown term independently from five fixed restart seeds
(201 to 205) and compares the learned rate functions pairwise on a shared
domain built from the training and held-out regulator values. It reports
every restart, including failed ones, the scale-normalized disagreement
between rate functions, the agreement between trajectories, and a derived
status. It is a diagnostic; it is not an acceptance criterion. Fisher
information over the physical parameters is also available on its own
through `HybridKinetics.assess_identifiability`.

## Several unknown terms

Since 0.16 a network may have one unknown destruction term on each of
several nodes. Each term is the rate `D_i(u)` of its own node, learned by
its own network, which sees only that term's regulators; the networks are
trained jointly against the same trajectories, so the only thing that ties
the terms together is the data. Each term then gets its own graph-local
library, built from its regulators as for a single term, its own rational
regression, and its own stability selection; the fit numbers of the report
(the residual on the first training experiment, over the training
experiments, and over the held-out experiments) are those of the model with
every discovered rate substituted at once.

**Per-term diagnostic.** The production/destruction scale check of the
[Identifiability diagnostic](@ref) runs for each term against the production
parameter of that term's node (`production_param = :auto` picks it, or a
`Dict` names it per node), and is reported in that term's block.

**Cross-term diagnostic.** Two unknown terms can trade against each other:
if scaling one term up and the other down leaves the observed trajectories
unchanged, the data do not fix either scale, and the coefficients discovered
for both inherit that freedom. `cross_term_collinearity` measures this with
the same construction as the scale check. For each term, the output of its
network is multiplied by `1 ± δ` (`δ = 10⁻³`), the model is simulated, and
the central difference of the observed trajectory entries is that term's
sensitivity vector. For each pair of terms the value is the absolute cosine
between the two sensitivity vectors: zero when the two terms move the
trajectory in unrelated directions, one when a scale change of one term is
indistinguishable, in these data, from a scale change of the other.
`discover_unknown_terms` reports the value for every pair and warns above
`CROSS_TERM_COLLINEARITY_THRESHOLD`.

What it establishes is narrow, and the same caveats as for the single-term
check apply. It is local: computed at the fitted parameters, for the first
training experiment's initial condition and sampling times. It concerns the
scales only: two terms with a low cosine can still be wrong in their
functional form, and a high cosine does not say which of the two absorbed
the other's contribution. It is not a structural identifiability result; a
value near one is a reason to distrust the discovered coefficients of both
terms, not a proof that the terms cannot be separated with other data.

**What the study measured.** The two-term fixtures of the benchmarks page
put two unknown terms on nodes that do not regulate each other's term
(`build_two_term_separate_network`) and on nodes that do
(`build_two_term_coupled_network`), each with the single-unknown control of
the same network, over five seeds and three noise levels (0, 0.02, 0.05).
On non-adjacent nodes the second term cost nothing measurable: the same
support F1 as the control, a learned-rate error within 1.15 times the
control's, cross-term collinearity 0.37 to 0.43. On adjacent nodes the
supports were still recovered, but the term whose regulator is the other
unknown node came out biased low in all fifteen runs (15 to 30%, median
16%), with a rate error 2 to 6 times the control's, and the cross-term
collinearity was 0.95 to 0.96 in every run. No run at any noise level
showed the two terms trading against each other with opposite signs. The
three-term fixture (noise 0 only) cost more: its adjacent pair came out 1.4
and 2.2 times further from the truth than the controls and the third term
lost its support in two of five seeds, at cross-term values of 0.5 to 0.6. The
tables and the figure are on the [Benchmarks](benchmarks.md#Two-unknown-terms)
page.

## Symbolic discovery

Discovery fits an implicit rational form

```math
D(z)\,\dot x - N(z) = 0
```

by sequentially thresholded least squares over a monomial library, with the
constant denominator coefficient anchored to one. `ImplicitSINDyPI` is the
default backend; `ExplicitSTLSQ` fits an explicit polynomial right-hand side.
Both are configured through `DiscoveryConfig` (threshold, maximum degree,
bootstrap resamples, validation fraction, domain samples, chunk size).

The library is graph-local: `local_basis(network, target; scope = :graph)`
builds monomials from the target state and its parents in the interaction
graph. The graph holds the declared edges and, since 0.12, an edge from
each regulator of every reaction with `known = false` to the species that
reaction changes, so a network that declares its unknown term as a
reaction alone (the tutorial's network does) gets the same graph-local
library as one that also declares the edge; in 0.11 and earlier its library
held the target state alone. Regulators of known reactions add no edges.
`scope = :global` uses every dynamic node and is the comparison used in the
benchmarks. For a bounded in-degree `k`, the library size grows with the sum
of `k_i^d` over targets rather than with `n^d`. The target state is always
part of its own library; the library comparison study on the
[Benchmarks](benchmarks.md#Why-the-four-state-recall-is-0.5) page shows
that this costs nothing as long as the samples vary in the target state,
and that a sample design on which the target is constant does.

```@example concepts
net = BiologicalNetwork(
    [NodeSpec(name = :S), NodeSpec(name = :R), NodeSpec(name = :Z)],
    [EdgeSpec(source = 2, target = 1, kind = INHIBITION, known = false,
        family = HILL, metadata = HybridKinetics.EmptyMetadata())])
graph = local_basis(net, 1; degree = 2, scope = :graph)
global_lib = local_basis(net, 1; degree = 2, scope = :global)
(graph.variables, length(graph.numerator), global_lib.variables, length(global_lib.numerator))
```

Two entry points share this machinery:

- `regress_unknown_rate(R, times, D)` regresses sampled values of the learned
  destruction rate on the regulator values. This is the path of the reference
  protocol: the neural term is sampled (`sample_unknown_destruction` along
  trajectories, or `HybridKinetics.sample_unknown_destruction_grid` on a regulator
  grid) and a rate `D(r)` is fitted.
- `discover_equations(X, times, network)` works on state trajectories and
  their derivatives (`estimate_derivatives`) without a trained model.

Candidates are validated on training columns, on held-out columns, and on a
grid of domain samples in the positive orthant; a candidate whose denominator
changes sign or approaches zero on any of them is rejected
(`DenominatorUnsafe`). Bootstrap resampling reports how often each term is
selected. Failures are reported through `DiscoveryRetcode`
(`InsufficientSamples`, `DenominatorUnsafe`, `EmptySupport`,
`SingularLibrary`, `DiscoveryFailed`); with `strict = true` they throw.

A recovered candidate can be exported as LaTeX (`equation_to_latex`), as a
callable (`equation_to_function`), or as a full right-hand side
(`export_rhs`). `compose_hybrid_rhs` swaps the callable in for the neural
term, and `hybrid_data_residual` integrates the result and compares it with
observations.

### Pruning nuisance terms

Discovery on a learned rate often keeps small terms that fit the neural
network's approximation error rather than the mechanism; on the reference
protocol these are a constant and a linear term next to the true Hill
monomials. The optional stability-selection stage,
`regress_unknown_rate(...; stability_selection = StabilitySelection())`,
resamples the training rows of the regression with replacement `n_boot`
times (default 100), repeats the thresholded fit on every resample, and
keeps a term of the fitted candidate only if it was selected in at least a
fraction `τ` of the resamples (default 0.8), refitting the coefficients of
the kept terms afterwards. Terms are never added, the stage is skipped when
it would remove every numerator term or make the denominator unsafe, and
the selection frequency of every library term is available through
`stability_selection_report` and `format_stability_selection`, so a user can
see why a term was kept or dropped. Its cost is `n_boot` thresholded fits on
the training rows, which for the reference protocol is a fraction of a
second. It is off by default, and with it off the discovery output is
unchanged; the [Benchmarks](benchmarks.md#Stability-selection-on-the-library-comparison-study)
page reports what it does on the library comparison study.

## The reference protocol

The recovery benchmarks and the example share one protocol, stored in
`HybridKinetics.REFERENCE_PROTOCOL`:

| Setting | Value |
|---|---|
| seed | 103 |
| initial conditions | 9, generated once |
| points per experiment | 50 over `t in [0, 8]` |
| observation noise | 0 |
| training | Adam 100, then BFGS 50 |
| discovery | bootstrap 8, discovery seed 3 |
| smoke variant | 1 initial condition, 8 points, no BFGS |

In the recovery suite the nine experiments are split 7/2: experiments 1 to 7
are used for training and for deriving the discovery grid; experiments 8 and 9
are held out. The reported quantities are:

| Quantity | Meaning |
|---|---|
| `data_residual` | hybrid residual on training experiment 1 |
| `data_residual_train` | mean of the residuals on experiments 1 to 7 |
| `data_residual_holdout` | mean of the residuals on experiments 8 and 9 |
| `d_rmse_holdout` | error of the neural rate at the observed held-out regulator values |
| `d_rmse_holdout_domain` | error of the neural rate on a band derived from the training data |
| `nn_rate_rmse`, `nn_correlation` | neural rate versus the true rate on the discovery grid |
| `support_recall`, `support_f1` | recovered monomials versus the true Hill support |

The acceptance criteria for the Hill-class recovery are three: the scale
warning is raised, `data_residual` is at most 0.30, and support recall is at
least 0.99. Combined support F1 is reported against a floor of 0.50 and is
not part of the criteria; nuisance terms typically remain. Held-out numbers
are reported and are never compared with a threshold. Michaelis-Menten
unknown terms are checked on the neural-rate error and the residual only.
All thresholds live in `HybridKinetics.RECOVERY_THRESHOLDS`:

```@example concepts
HybridKinetics.RECOVERY_THRESHOLDS
```

## Robustness checks

Two further checks exist beyond the single-seed protocol. Both are
unexported.

- `HybridKinetics.evaluate_trained_graph_local` trains one model (seed 401, three
  initial conditions), samples its learned rate once, and runs discovery
  three times on the same samples with the graph-local library, a global
  library, and a library from a deliberately wrong graph. The graph-local
  run must keep the true regulator; the wrong-graph run must miss it. A fast
  version runs in the default tests; the full version is
  `test/run_trained_library_comparison.jl`.
- `HybridKinetics.TrajectoryOccupancy` collects the observed states of the training
  or held-out experiments as an alternative sampling context for the learned
  rate. It is not used by the functional-identifiability diagnostic or by the
  held-out evaluation.

A multi-seed robustness study of the full protocol is not implemented;
`benchmark/recovery_seeds.jl --ude` runs the trained-model protocol on five
seeds as a report.
