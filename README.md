# HybridKinetics.jl

Hybrid kinetic models for small biochemical networks: learn the unknown rate laws from time-series data, then recover them symbolically.

[![CI](https://github.com/utkuyilmaz1903/HybridKinetics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/utkuyilmaz1903/HybridKinetics.jl/actions/workflows/ci.yml) [![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://utkuyilmaz1903.github.io/HybridKinetics.jl/stable/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Julia](https://img.shields.io/badge/julia-%E2%89%A5%201.10-9558B2.svg)](https://julialang.org) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac) [![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Version 0.17. The public API may still change before 1.0; see [CHANGELOG.md](CHANGELOG.md).

## What HybridKinetics does

HybridKinetics fits hybrid models of small biochemical networks. You give it a
known interaction graph and known kinetics (mass action, linear decay, Hill,
Michaelis-Menten saturation, competitive binding, or a custom rate); it
compiles those into a production-destruction ODE
`du_i/dt = P_i(u) - D_i(u) * u_i`. Any number of destruction terms, on
distinct nodes, may be marked unknown. Each is replaced by its own small
neural network (a universal differential equation), trained jointly on
time-series data from one or more initial conditions, and then approximated
symbolically by sparse rational regression (implicit SINDy) over a library
built only from that node's graph neighbours.

The package reports three things: whether each unknown term is practically
identifiable from the data (a Fisher-information and scale-collinearity
diagnostic), how well the hybrid model reproduces observed and held-out
trajectories, and which symbolic terms are recovered. It is a research tool
for small networks with a known graph (the benchmarks cover 2- to 6-state
networks), not a general-purpose network-inference tool or reaction-network
solver.

## Installation

HybridKinetics requires Julia 1.10 or newer. It is not yet in the General registry;
install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/utkuyilmaz1903/HybridKinetics.jl")
```

or clone the repository and instantiate its environment:

```bash
git clone https://github.com/utkuyilmaz1903/HybridKinetics.jl.git
cd HybridKinetics.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

The block below builds a two-species network, marks one destruction term as
unknown, generates synthetic data from four initial conditions, and calls
`discover_unknown_terms`, which trains the hybrid model on three of them,
prints the identifiability warning, discovers a symbolic rate, and reports
the residuals on the training and on the held-out experiment. It took
about three minutes on a 4-core machine.

```julia
using HybridKinetics, Random

# Two species. S is produced in proportion to R and degraded by a Hill-type
# mechanism driven by R; R is produced from S and decays linearly.
# `known = false` marks the Hill degradation as the unknown term.
function network(; known::Bool)
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R)]
    reactions = [
        ReactionSpec(name = :produce_s, stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :degrade_s, stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = known, family = HILL,
            metadata = HillMetadata(vmax_param = :vmax, k_param = :K, hill_order = 2)),
        ReactionSpec(name = :produce_r, stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r, stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r))]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

rng = MersenneTwister(1)
truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
data = generate_experiment_set(rng; network = network(known = true), truth_params = truth,
    initial_conditions = [[0.2, 0.1], [1.0, 0.5], [0.5, 1.2], [0.3, 0.8]],
    tspan = (0.0, 10.0), n_points = 40, noise_σ = 0.0)

# Trains on the first three experiments, holds out the fourth, and prints the report.
result = discover_unknown_terms(network(known = false), data; rng = rng, holdout = 1,
    training = TrainingConfig(adam_iterations = 100, bfgs_iterations = 20, log_every = 10^6))
```

`discover_unknown_terms` prints a four-section report (identifiability, fit,
discovery, reproduction) and returns a `DiscoveryRun` that holds the
trained model, the identifiability diagnostic, the discovery, and the
residuals; `report_unknown_terms(result)` returns the report as a string. The lines that
matter most, from a run in September 2026:

Several unknown destruction terms, one per node, go through the same call:
mark each with `ReactionSpec(known = false)` or with
`BiologicalNetwork(...; unknown = [UnknownTerm(:A), UnknownTerm(:B)])`, and
read the per-term results as `result[:A]` and `result[:B]`. The report then
adds a cross-term section with the scale collinearity of every pair of
terms; the tutorial has a two-term example and the benchmarks page measures
what a second term costs.

```text
  unidentifiable_edge: true
  coefficients_are_biological_constants: false
  collinearity: 0.9952
  hybrid_data_residual: 0.01272
  hybrid_data_residual_train: 0.009773
  hybrid_data_residual_holdout: 0.02114
dx[1]/dt = (0.21548*1 + -0.83814*x[1] + 2.8172*x[1]^2) / (1 + -1.6151*x[1] + 2.3666*x[1]^2)
```

Read top to bottom: the production rate `k_prod` and the scale of the unknown
term are collinear, so the recovered coefficients are not biological constants;
the hybrid model with the discovered rational rate reproduces the training
trajectories with a root-mean-square residual of about 0.01 and the held-out
trajectory with about 0.02; the discovered rate has the Hill-like
`x^2 / (K + x^2)` structure, with a constant and a linear term remaining.
`x[1]` is the regulator `R`.

The [tutorial](https://utkuyilmaz1903.github.io/HybridKinetics.jl/stable/tutorial/)
shows the same call on the reference protocol and then the step-by-step
version of what it does. The reference example (seed 103, nine initial
conditions, 50 points each, Adam 100 / BFGS 50, discovery on a regulator grid)
takes about 10 to 15 minutes:

```bash
julia --project=. examples/unknown_inhibition.jl
```

It prints the same four-section report. The lines that matter most:

```text
  unidentifiable_edge: true
  coefficients_are_biological_constants: false
  collinearity: 0.997
  hybrid_data_residual: 0.001777
dx[1]/dt = (0.24118*1 + -1.3569*x[1] + 7.7609*x[1]^2) / (1 + -0.3862*x[1] + 4.1863*x[1]^2)
  extras: 1, r
```

![Hybrid model and discovered destruction rate for the unknown-inhibition example](docs/src/assets/unknown_inhibition.png)

## How it works

| Step | What happens | Main functions |
|---|---|---|
| Network specification | Nodes, edges or reactions, and typed kinetic metadata | `BiologicalNetwork`, `NodeSpec`, `ReactionSpec`, `HillMetadata`, ... |
| Compile | Known kinetics become production and destruction terms; each unknown term becomes a neural network with a softplus output | `build_ude_model`, `compile_mechanism` |
| Simulate | The model is an ordinary `ODEProblem` and works with OrdinaryDiffEq solvers | `ODEProblem(model, u0, tspan, p)`, `ude_rhs`, `ude_rhs!` |
| Train | Adam followed by BFGS on the trajectory mean-squared error across experiments, with adjoint sensitivities | `train_ude`, `train_experiments`, `TrainingConfig` |
| Identifiability check | Fisher condition number and the cosine between the production-rate and destruction-scale trajectory Jacobians | `HybridKinetics.report_production_destruction_tradeoff` |
| Symbolic discovery | The learned rate is sampled and fitted by implicit sparse regression over a graph-local rational library | `sample_unknown_destruction`, `regress_unknown_rate`, `local_basis` |
| Resimulate | The discovered rate replaces the neural term and the hybrid model is compared with data | `compose_hybrid_rhs`, `hybrid_data_residual`, `export_rhs` |

Synthetic data for several initial conditions come from
`generate_experiment_set`; measured data can be loaded with
`experiment_from_csv`.

## Scope and limitations

- **Unknown destruction terms only, one per node.** Any number of
  destruction terms on distinct nodes may be unknown. Two unknown terms on
  the same node are refused with an error (two rates multiplying the same
  state are one rate to the data), and an unknown production term is out of
  scope and refused. Two terms on nodes that do not regulate each other's
  term separate cleanly; two terms on adjacent nodes leave one term's learned
  rate 15–30% low, which the cross-term diagnostic flags above 0.46; three
  terms degrade further and lose a support in some seeds (0.16 study,
  benchmarks page). That 15–30% is a property of the search rather than of
  the data: fitted from the true rates the same data leave both terms within
  1% of the truth on the median and 5% at worst, and none of the three
  corrections measured in 0.18 removes it.
  The reference recovery protocol behind the benchmarks
  (`run_recovery_suite`) is a single-term instrument and raises an error for
  zero or several unknown terms.
- **The identifiability diagnostic is local and practical.** It flags an edge
  as unidentifiable when the Fisher condition number exceeds `1e6` or when the
  production-rate and destruction-scale trajectory Jacobians have a cosine of
  at least 0.95, on one trajectory. It is not a structural identifiability
  proof, and a passing check does not certify the recovered mechanism.
- **A good fit is not mechanism recovery.** A small trajectory residual shows
  that the hybrid model reproduces the data. It does not by itself show that
  the true mechanism was found.
- **Coefficients are not biological constants** unless the scale of production
  or destruction is fixed by outside information. With observed concentrations
  alone, the production rate and the scale of the unknown term trade off
  against each other.
- **The discovered form is a rational function, not a canonical Hill law.**
  The reference protocol recovers the true monomials of the Hill term (the
  acceptance criterion is recall of at least 0.99), but extra terms remain, a
  constant and a linear term, which hold the combined support F1 at about
  0.57. That number measures how far the learned rate sits from the true law
  in the library's basis, not a mislabelled truth and not an unreachable
  ceiling: the extra terms carry coefficients of the same order as the true
  ones and describe the network's error rather than the mechanism (0.18
  study, benchmarks page). The F1 threshold in the benchmarks is 0.50; the
  package does not turn the neural term into a canonical Hill expression with
  named parameters.
- **Validation on synthetic data; two measured datasets.** The benchmarks and
  the tests use data generated from the compiled ground-truth mechanism, and
  the CSV in `examples/data/` is synthetic. The primary real-data example is
  the laccase/ABTS case study of `examples/laccase_abts/`: on published
  progress curves, the two-state product-inhibition model yields an accepted
  rational rate, a saturating function of the substrate (the one-state model
  does not). The p53–Mdm2 study of `examples/p53_mdm2/` is a partially
  observed illustration (p53 measured, Mdm2 an unobserved state) with a
  negative discovery result. Each has its own documentation page; the data
  are downloaded by scripts with checksums and do not ship with the package,
  and the runs are not part of the tests or CI.
- **Held-out validation is reported, not enforced.** The reference protocol
  trains on seven of nine initial conditions and reports the residual and the
  destruction-rate error on the other two. Those numbers are evidence; they are
  not part of the acceptance criteria.
- **Robustness validation is partial.** The package includes a diagnostic that
  compares independently trained neural terms across five restart seeds, and a
  check that runs discovery with the graph-local, a global, and a wrong-graph
  library on the same trained model. A multi-seed robustness study of the full
  protocol is not implemented.
- **Partial observation** is limited to discovery from subsampled
  destruction-rate values; training on unobserved states is not supported.
- **Extensions are thin.** The GPU extension only transfers arrays; there is no
  batched GPU training. SBML import reads species, reactions, and
  stoichiometry, but does not parse kinetic laws (MathML); such reactions
  become unknown neural terms.
- Not in scope: inferring the interaction graph, general reaction-network
  solving, several unknown terms at once, and experimental design.

## Optional extensions

Each loads automatically when the trigger package is present. All are
experimental and unexported; call them as `HybridKinetics.name`.

| Trigger package | What it adds |
|---|---|
| `CUDA` | Moves experiment arrays to the GPU (`HybridKinetics.to_device`, `HybridKinetics.gpu_execute`) |
| `Plots` | `HybridKinetics.plot_training` for observations, truth, prediction, and the loss history |
| `ModelingToolkit` | `HybridKinetics.export_mtk_system` converts a compiled model to an `ODESystem` |
| `SBML` | `HybridKinetics.import_sbml_network` builds a network from species and reactions |
| `SBMLToolkit` + `Catalyst` | `HybridKinetics.import_sbmltoolkit_network` with mass-action detection through Catalyst |
| `DataDrivenSparse` | `HybridKinetics.DataDrivenSparseSTLSQ`, an alternative sparse-regression backend |

## Documentation

The documentation is at
[utkuyilmaz1903.github.io/HybridKinetics.jl/dev](https://utkuyilmaz1903.github.io/HybridKinetics.jl/dev/).
Main pages: Getting started, Tutorial (the unknown-inhibition walkthrough),
Concepts (model form, identifiability, discovery, the train/holdout protocol),
How-to recipes, Benchmarks, API reference, and Scope and limitations.

## Development

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                                         # default tests
HYBRIDKINETICS_TEST_HEAVY=1 julia --project=. -e 'using Pkg; Pkg.test()'                    # plus the slow training-loop tests
HYBRIDKINETICS_SMOKE=1 ADAM_ITERS=2 BFGS_ITERS=0 julia --project=. examples/unknown_inhibition.jl  # 1-IC smoke run
julia --project=. examples/unknown_inhibition.jl                                     # full example
julia --project=. test/run_recovery_hard.jl                                          # trained-model recovery checks
julia --project=. benchmark/recovery_suite.jl                                        # fast recovery benchmarks
julia --project=docs docs/instantiate.jl && julia --project=docs docs/make.jl        # build the docs
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the ground rules.

## Citing

Citation metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@software{hybridkinetics,
  author  = {Yılmaz, Utku},
  title   = {HybridKinetics.jl: hybrid kinetic models for small biochemical networks},
  year    = {2026},
  version = {0.15.0},
  url     = {https://github.com/utkuyilmaz1903/HybridKinetics.jl},
  license = {MIT}
}
```

## License

MIT. See [LICENSE](LICENSE).
