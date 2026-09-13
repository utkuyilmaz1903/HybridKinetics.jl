# How-to recipes

Short recipes for common tasks. Blocks marked `@example` run in the
documentation build; the others are illustrative and assume the `model`,
`trained`, `set`, and `term` objects from the [Tutorial](tutorial.md).

## Run the whole workflow in one call

`discover_unknown_terms(network, experiments)` builds the hybrid model,
trains it (a warm-up on the first experiment, then Adam 100 and BFGS 50 on
the training experiments), samples the learned rate on the regulator grid,
discovers a rational rate, computes the identifiability diagnostic and the
residuals, and prints the four-section report. The last `holdout`
experiments (default 2) are held out of training and reported separately.

```julia
result = discover_unknown_terms(ude_net, set; rng = MersenneTwister(0), holdout = 2)
result.params            # trained parameters
result[:S].discovery     # DiscoveryResult of the unknown term on S
result.residuals         # (data_residual, data_residual_train, data_residual_holdout)
report_unknown_terms(result)  # the report as a string
```

`training = TrainingConfig(...)`, `discovery = DiscoveryConfig(...)`,
`stability_selection = StabilitySelection()`, `warmup = false`, and
`phys_init` change the individual steps; the [Tutorial](tutorial.md) shows
the call on the reference protocol and then the steps one by one.

## Migrating from 0.15

0.16 supports several unknown destruction terms, and the single-term entry
point was renamed to say so. The old names are still defined, but each raises
an error that names its replacement, so a 0.15 script fails at the first call
rather than running with different meaning.

| 0.15 | 0.16 |
|---|---|
| `discover_unknown_term(network, experiments; ...)` | `discover_unknown_terms(network, experiments; ...)`, same keywords except `term` (every unknown term is discovered) |
| `UnknownTermResult` returned by the call | `DiscoveryRun`; the per-term `UnknownTermResult` is `result[:S]` (by node name), `result[1]`, or an element of `unknown_terms(result)` |
| `result.discovery`, `result.identifiability`, `result.samples`, `result.term`, `result.extras` | `result[:S].discovery` and so on; `result.params`, `result.training`, `result.residuals`, `result.settings` stay at the top level |
| `report_unknown_term(result)` | `report_unknown_terms(result)`; with one unknown term the text is identical |
| `symbolic(result)` | unchanged with one term; `symbolic(result; node = :S)` or `symbolic(result[:S])` with several |
| `export_mtk_system(model; discovered = result)` | unchanged; substitutes every term's rate |
| `network_from_reactionsystem(rs; unknown = 2)` | unchanged; `unknown = [2, 4]` marks several |

Before:

```julia
result = discover_unknown_term(net, set; holdout = 2)
result.discovery.equations
report_unknown_term(result)
```

After:

```julia
result = discover_unknown_terms(net, set; holdout = 2)
result[:S].discovery.equations
report_unknown_terms(result)
```

The error a 0.15 call raises:

```
discover_unknown_term was removed in HybridKinetics 0.16. Call
discover_unknown_terms(network, experiments; ...) instead; it takes the same
keywords except `term` (all unknown terms are discovered) and returns an
DiscoveryRun whose per-term contents are result[:node]. Migration:
https://utkuyilmaz1903.github.io/HybridKinetics.jl/stable/howto/#Migrating-from-0.15
```

With one unknown term nothing else changes: the trained parameters, the
sampled rate, the discovered equations and coefficients, the residuals and
the report text are the ones 0.15 produced, and the test suite checks them
against values recorded on 0.15.0 (`test/support/fingerprints_015.toml`).

### Migrating from 0.16

0.17 renames three names and stops exporting six, with no aliases (the
package was not yet registered). The old names are gone, so a 0.16 script
fails at the first use with `UndefVarError` rather than running differently.

| 0.16 | 0.17 |
|---|---|
| `discover_unknown_rate` | `regress_unknown_rate` |
| `ude_system` | `ude_rhs` |
| `UnknownTermsResult` | `DiscoveryRun` (the container; `UnknownTermResult` is still the per-term type) |
| `pack_parameters`, `positive_parameter`, `allocate_cache`, `RECOVERY_THRESHOLDS`, `EmptyMetadata`, `MetadataLike` | unchanged, but no longer exported: write `HybridKinetics.pack_parameters` or `using HybridKinetics: pack_parameters` |

## Marking several terms unknown

Mark each unknown destruction term with `ReactionSpec(known = false)` as
before, or pass `UnknownTerm` specs to the network constructor; both give the
same network. One unknown term per node; a second one on the same node, and
an unknown production term, are errors.

```@example multi
using HybridKinetics, Random
truth = HybridKinetics.build_two_term_coupled_network(; unknown = ())
net = BiologicalNetwork(truth.nodes, EdgeSpec[]; reactions = truth.reactions,
    unknown = [UnknownTerm(:A), UnknownTerm(:B)])
unknown_terms(net)
```

`UnknownTerm(:A; regulators = [:C])` overrides the regulators the term may
depend on (one or two nodes); `UnknownTerm(:A; library = config)` gives that
term its own `DiscoveryConfig` when the specs are passed to
`discover_unknown_terms(...; terms = ...)`.

## Reading a multi-term result

Two result types: `discover_unknown_terms` returns one `DiscoveryRun`, the
whole run (model, trained parameters, residuals, cross-term diagnostic), and
`run[:S]`, `run[1]` or `unknown_terms(run)` give the `UnknownTermResult` of
one term (its identifiability, discovery, samples and extras).

```@example multi
set = generate_experiment_set(MersenneTwister(103); network = truth,
    initial_conditions = [[0.3, 0.2, 0.5], [0.6, 0.4, 0.8], [0.2, 0.7, 0.3]],
    tspan = (0.0, 6.0), n_points = 12, noise_σ = 0.0,
    truth_params = HybridKinetics.TWO_TERM_COUPLED_TRUTH)
result = discover_unknown_terms(net, set; holdout = 1, rng = MersenneTwister(7),
    training = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6),
    verbose = false)
keys(result)                       # [:A, :B]
result[:A].discovery.equations     # the rate discovered for the term on A
result.residuals                   # fit of the model with both rates substituted
result.cross_term                  # one entry per pair of terms
println(report_unknown_terms(result))
```

The report has one shared fit section, one block per term (its
identifiability diagnostic and its discovery), a cross-term section, and the
reproduction section. The cross-term collinearity is explained on the
[Concepts](concepts.md#Several-unknown-terms) page.

## Starting from a Catalyst model

A Catalyst `ReactionSystem` converts to a `BiologicalNetwork` with
`network_from_reactionsystem`. Every reaction Catalyst can express as a
parameter, `k * Y`, `hill(Y, v, K, n)`, or `mm(Y, v, K)` rate on one species
compiles to the matching known term; the reaction named by `unknown` (its
index or its `description` metadata) becomes the unknown destruction term;
anything else raises an error that names the reaction and its rate. The
tutorial network in Catalyst:

```@example catalyst
using HybridKinetics, Catalyst, ModelingToolkit, Symbolics, Random

tutorial = @reaction_network tutorial begin
    k_prod * R, 0 --> S
    hill(R, vmax, K, 2), S --> 0, [description = "unknown"]
    k_rs * S, 0 --> R
    k_r, R --> 0
end

net = network_from_reactionsystem(tutorial; unknown = "unknown")
[node.name for node in net.nodes], length(net.reactions), HybridKinetics.count_unknown_destructions(net)
```

`unknown = nothing` compiles every reaction as known kinetics, which is how
a ground-truth model for synthetic data is built from the same system:

```@example catalyst
truth = network_from_reactionsystem(tutorial; unknown = nothing)
truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
set = generate_experiment_set(MersenneTwister(103); network = truth,
    initial_conditions = [[0.25, 0.20], [0.80, 0.35], [0.40, 1.10]],
    tspan = (0.0, 8.0), n_points = 8, noise_σ = 0.0, truth_params = truth_params)
length(set.experiments), set.state_names
```

From here the workflow is the usual one; a two-step training keeps this
example short (the reference settings are Adam 100 and BFGS 50):

```@example catalyst
result = discover_unknown_terms(net, set;
    training = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6),
    holdout = 0, rng = MersenneTwister(7), verbose = false)
result[:S].discovery.success, result[:S].discovery.equations
```

The discovered rate is available as a `Symbolics` expression in the
network's state names, and the completed model as a ModelingToolkit system
whose states carry the same names, ready for `ODEProblem`:

```@example catalyst
if result[:S].discovery.success
    rate = symbolic(result)
    completed = HybridKinetics.export_mtk_system(result.model; discovered = result)
    rate, ModelingToolkit.unknowns(completed), ModelingToolkit.equations(completed)
end
```

The states of the exported system are `S(t)` and `R(t)`, its parameters are
the Catalyst parameters of the known reactions, and its right-hand side
equals Catalyst's own ODE system for the known reactions
(`Catalyst.ode_model(tutorial)`, or `convert(ODESystem, tutorial)` before Catalyst 16); the test suite checks both. Rates the
converter does not compile: bimolecular mass action (two substrates), a
substrate of stoichiometry above 1, `hillr`, `mmr`, `hillar`, and any rate
law written as a full rate with `=>`.

## Load an experiment from CSV

`experiment_from_csv` reads a table whose first column is time and whose
remaining columns are the observed states. It returns the `Experiment` and
the column names. The file shipped in `examples/data/` is a synthetic
fixture generated from the tutorial network, not a measured series.

```@example howto
using HybridKinetics
path = joinpath(pkgdir(HybridKinetics), "examples", "data", "unknown_inhibition.csv")
experiment, names = experiment_from_csv(path)
(names, length(experiment.times), size(experiment.observations))
```

Several experiments become an `ExperimentSet`:

```@example howto
set = ExperimentSet([experiment], names)
length(set.experiments)
```

`write_experiment_csv(path, experiment; state_names)` writes one back.
`Experiment.mask` can hide a state or time subset from the loss;
`train_experiments` respects it. Training on states that are never observed
is not supported.

## Mark an edge as unknown

Build the network with the public constructors and set `known = false` on the
one reaction (or edge) whose kinetics you do not trust. It compiles to a
`NeuralDestructionTerm` whose inputs are that reaction's regulators.

```@example howto
using Random
network = BiologicalNetwork(
    [NodeSpec(name = :S), NodeSpec(name = :R)],
    EdgeSpec[];
    reactions = [
        ReactionSpec(name = :produce_s, stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg, stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = false, family = HILL,
            metadata = HillMetadata(vmax_param = :vmax, k_param = :K, hill_order = 2)),
        ReactionSpec(name = :produce_r, stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r, stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r))])
model, params = build_ude_model(MersenneTwister(0), network)
HybridKinetics.count_unknown_destructions(model)
```

The step-by-step chain of this section and the reference recovery
protocol work on exactly one unknown destruction term;
`HybridKinetics.assert_single_unknown_destruction(model)` raises an error
otherwise. `discover_unknown_terms` handles several (one per node, see
"Marking several terms unknown"); `validate_network` refuses two unknown
terms on the same node and an unknown production term but does not
otherwise limit the count.

## Generate synthetic data

`generate_experiment_set` compiles the ground-truth model once and
integrates every initial condition from it:

```@example howto
known = BiologicalNetwork(
    [NodeSpec(name = :A), NodeSpec(name = :B)],
    EdgeSpec[];
    reactions = [
        ReactionSpec(name = :b_drives_a, stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_ba)),
        ReactionSpec(name = :a_decay, stoichiometry = Dict(1 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_a)),
        ReactionSpec(name = :b_decay, stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_b))])
synthetic = generate_experiment_set(MersenneTwister(2); network = known,
    truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5),
    initial_conditions = [[0.2, 0.1], [0.5, 0.4]], tspan = (0.0, 4.0),
    n_points = 9, noise_σ = 0.02)
(length(synthetic.experiments), synthetic.metadata[:compiled_once])
```

## Train, discover, resimulate

```julia
trained = train_experiments(params, set, model;
    config = TrainingConfig(adam_iterations = 100, bfgs_iterations = 50))
X = predict_ude(trained.params, u0, tspan, times, model)
R, D, term = sample_unknown_destruction(model, trained.params, X)
discovery = regress_unknown_rate(R, times, D; strict = false)
if discovery.success
    rate_fn = equation_to_function(discovery.candidates[1])
    rhs = compose_hybrid_rhs(model, trained.params, term, rate_fn)
    residual = hybrid_data_residual(model, trained.params, term, rate_fn,
        u0, tspan, times, data)
end
```

`TrainingConfig(frozen_phys = [:k_prod])` pins a known production rate during
training. Use it when the rate is known from a separate assay; it does not
by itself remove the collinearity between the production rate and the scale
of the unknown term.

With `strict = false`, check `discovery.retcode` instead of catching an error:

| `retcode` | Meaning |
|---|---|
| `DiscoverySuccess` | a support was recovered; the hybrid right-hand side can be built |
| `InsufficientSamples` | fewer than 20 sample columns |
| `DenominatorUnsafe` | the denominator changed sign or approached zero on a validation set |
| `EmptySupport` | thresholding removed every term |
| `SingularLibrary` | the design matrix was singular |
| `DiscoveryFailed` | any other error; see `discovery.message` |

`export_rhs` refuses a failed result.

## Discover from raw trajectories

Trajectory data can enter discovery without a trained model. Derivatives are
estimated by central differences, so this path is only reliable at low noise
(the analytical benchmarks succeed up to 2% noise and fail at 5%).

```@example howto
e = first(synthetic.experiments)
dX = estimate_derivatives(e.observations, e.times)
size(dX)
```

```julia
result = discover_equations(X, times, network; derivatives = dX)
rhs = export_rhs(result)
```

## Print the identifiability warning and the report

```julia
ident = HybridKinetics.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = true)
println(HybridKinetics.format_protocol_result(ident; residual = residual,
    equations = discovery.equations))
```

`format_protocol_result` accepts the values to print as keyword arguments;
anything not supplied is printed as `NA` or as "not scored".

```@example howto
ident = (; unidentifiable_edge = true, production_param = :k_prod, collinearity = 0.997)
print(HybridKinetics.format_protocol_result(ident; residual = 0.0017769, seed = 103, n_ics = 9))
```

## Use the SciML solve surface

A `UDEModel` behaves as an `ODEProblem` factory. `remake` works on `p`, `u0`,
and `tspan`, and an in-place problem with a preallocated cache avoids
allocations in the forward pass:

```@example howto
using SciMLBase, OrdinaryDiffEq
p = HybridKinetics.pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5),
    build_ude_model(MersenneTwister(0), known)[2].nn)
m = build_ude_model(MersenneTwister(0), known)[1]
prob = ODEProblem(m, [0.2, 0.1], (0.0, 4.0), p)
cache = HybridKinetics.allocate_cache(m, Float64)
inplace = ODEProblem(m, [0.2, 0.1], (0.0, 4.0), p; inplace = true, cache = cache)
sol = solve(remake(prob; u0 = [0.5, 0.4]), Tsit5(); saveat = [0.0, 2.0, 4.0])
round.(sol[end]; digits = 4)
```

Pair the in-place problem with `ProductionAD()` for training:

```julia
solver_config = default_solver_config(model; ad_policy = ProductionAD())
prediction = predict_ude(params, u0, tspan, times, model;
    solver_config = solver_config, cache = cache)
```

`auto_sensealg(model)` returns the adjoint that `train_ude` will use. A
one-shot Optimization.jl path is available as an unexported alternative to
`train_ude`:

```julia
prob, objective = HybridKinetics.build_optimization_problem(
    model, params, data, times, u0, tspan; config = TrainingConfig())
result = HybridKinetics.train_via_optimization(
    model, params, data, times, u0, tspan; maxiters = 50)
```

## Which right-hand side builder do I want?

Three functions turn a model or a discovery into a callable right-hand side;
they answer different questions.

| You have | You want | Call |
|---|---|---|
| a compiled `UDEModel` with trained parameters `p` | to integrate the hybrid model as it is, neural terms included, through the SciML interface (`ODEProblem`, `remake`, sensitivities) | `build_ude_function(model)` gives the `ODEFunction`; `ude_rhs(model, u, p, t)` or `ude_rhs!(du, u, p, t, cache)` evaluate it directly |
| a `DiscoveryRun` or an `UnknownTermResult` | the same model with the *discovered rational rate* substituted for a neural term, everything known kept from the compiled model | `compose_hybrid_rhs(model, p, term, rate_fn)` (one term) or `compose_hybrid_rhs(model, p, pairs)` (several); `hybrid_data_residual` scores it against data |
| a `DiscoveryResult` of `discover_equations` on every state | a right-hand side made only of the discovered equations, no compiled terms at all | `export_rhs(result)` |

For a completed model as a ModelingToolkit system rather than a Julia
closure, use `export_mtk_system(model; discovered = run)`
([Extensions](extensions.md)).

## Fit from several starting points

A fit begins from one random draw of the neural term, and which draw it is
changes where training ends up: on the reference two-state protocol, five
draws of the neural term on identical data end at final training losses
spanning a factor of 3.9. `restarts` fits that many draws and keeps the one
that reaches the lowest final training loss.

```julia
result = train_ude(p_init, data, times, u0, tspan, model;
    config = TrainingConfig(adam_iterations = 100, bfgs_iterations = 50,
        restarts = 5))
result.metadata.config.restart_losses   # the final loss of every attempt
```

Only the neural term is drawn again; the physical guess and every other
setting stay as given, and the draws come from the training seed, so a set of
restarts repeats exactly. The default is 1, and the first restart is the fit
the package would have done on its own, so the option can only improve the
loss it selects on — at the cost of that many times the training time.

What it buys, measured on the reference protocol, is in
[Benchmarks](benchmarks.md): a lower loss and a closer learned rate, and the
same discovered support. Choosing the lowest-loss fit is not the same as
choosing the one that discovers best.

## Checkpoint and resume

`train_ude` writes a checkpoint every `checkpoint_every` Adam iterations when
`checkpoint_path` is given. A checkpoint stores the parameters, the Optimisers
state, the iteration counter, and the augmented-Lagrangian state, serialized
with Julia's `Serialization`. `resume_training` continues from it without
recompiling the model; BFGS is a terminal stage and restarts after
resumption.

```julia
trained = train_ude(p_init, data, times, u0, tspan, model;
    config = TrainingConfig(adam_iterations = 200),
    checkpoint_path = "run.jls", checkpoint_every = 50)
checkpoint = HybridKinetics.load_checkpoint("run.jls")
resumed = HybridKinetics.resume_training(checkpoint, data, times, u0, tspan, model;
    config = TrainingConfig(adam_iterations = 300))
```

`HybridKinetics.save_result` and `HybridKinetics.load_result` do the same for a finished
`TrainingResult`.

## Run experiments on several threads or processes

`HybridKinetics.execute_experiments(f, set; config = HybridKinetics.ExecutionConfig(backend = :threads))`
maps `f` over the experiments with the serial, threaded, or distributed
backend and returns the results in input order. The `:gpu` backend needs the
CUDA extension and only transfers arrays (see [Extensions](extensions.md)).

## Run part of the recovery suite

`HybridKinetics.run_recovery_suite` takes a `sections` tuple. Sections that are not
requested are not run, so the known-kinetics checks can be run without the
trained-model protocol:

```julia
report = HybridKinetics.run_recovery_suite(MersenneTwister(1);
    sections = (:linear, :mm, :hill, :competitive),
    linear_adam = 1, linear_bfgs = 0, mm_adam = 1, mm_bfgs = 0,
    hill_adam = 1, hill_bfgs = 0, competitive_adam = 1, competitive_bfgs = 0)
haskey(report, :ude_discovery)   # false
```

See [Benchmarks](benchmarks.md) for the list of sections.
