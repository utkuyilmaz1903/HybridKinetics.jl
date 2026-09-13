###############################################################################
# One-call entry point: discover_unknown_terms (exported).
#
# Runs the reference workflow of examples/unknown_inhibition.jl with the same
# functions, in the same order, with the same defaults: build the hybrid
# model, warm up on the first experiment, train jointly, then for every
# unknown term sample its learned rate on the regulator grid of the training
# experiments, discover a rational rate and compute the identifiability
# diagnostic; then the hybrid residuals of the model with every discovered
# rate substituted, and, with two or more terms, the cross-term diagnostic.
# With one unknown term every number is the one 0.15's discover_unknown_term
# produced (test/support/fingerprints_015.toml records them).
###############################################################################

"""
    UnknownTermResult

What `discover_unknown_terms` computed for one unknown term; `result[:S]`,
`result[1]` and `unknown_terms(result)` return these:

- `node`: the name of the node whose destruction term this is;
- `network`, `model`, `params`, `training`: the hybrid model, its trained
  parameters and the `TrainingResult` of the joint fit (shared by all terms);
- `term`: the compiled neural destruction term;
- `identifiability`: the production/destruction trade-off report of this
  term on the first training experiment;
- `discovery`: the `DiscoveryResult` of the rational-rate regression;
- `samples`: the regulator grid `R` and the learned rate `D` the discovery
  used;
- `selection_frequency`: how often each library term was selected by the
  bootstrap of the accepted candidate, `nothing` when discovery failed;
- `extras`: the recovered monomials outside the known support, or `nothing`
  when no truth was given;
- `settings`: the values printed in the reproduction section of the report.

`symbolic(result[:S])` and `export_mtk_system(model; discovered = result[:S])`
accept it.
"""
struct UnknownTermResult{M, P, T, I, D, S}
    node::Symbol
    network::BiologicalNetwork
    model::M
    params::P
    training::TrainingResult
    term::T
    identifiability::I
    discovery::D
    samples::S
    selection_frequency::Union{Nothing, Vector{Float64}}
    extras::Union{Nothing, Vector{String}}
    settings::NamedTuple
end

"""
    DiscoveryRun

Everything `discover_unknown_terms` computes. This is the container for the
whole run; `UnknownTermResult` is the per-term type inside it.

- `network`, `model`, `params`, `training`: the hybrid model, its trained
  parameters, and the `TrainingResult` of the joint fit;
- `terms`: one `UnknownTermResult` per unknown term, in node order;
  `result[:S]` selects by node name, `result[i]` by position, and
  `unknown_terms(result)` returns the vector;
- `residuals`: `data_residual` (the hybrid model with every discovered rate
  substituted, against the first training experiment),
  `data_residual_train` (mean over the training experiments), and
  `data_residual_holdout` (mean over the held-out experiments, `NaN` when
  none were held out); `Inf` when a discovery failed;
- `cross_term`: the pairwise cross-term collinearity, one entry per pair of
  unknown terms (`nodes`, `terms`, `collinearity`); empty with one term;
- `training_indices`, `holdout_indices`: which experiments trained the
  model and which were held out;
- `settings`: the values printed in the reproduction section of the report.

`report_unknown_terms(result)` returns the report as a string and `show`
prints it. With one unknown term the report is the four-section report of
0.15; with several it has one shared fit section, one block per term, and a
cross-term section.
"""
struct DiscoveryRun{M, P, V <: AbstractVector, X}
    network::BiologicalNetwork
    model::M
    params::P
    training::TrainingResult
    terms::V
    residuals::NamedTuple{(:data_residual, :data_residual_train, :data_residual_holdout),
        NTuple{3, Float64}}
    cross_term::X
    training_indices::Vector{Int}
    holdout_indices::Vector{Int}
    settings::NamedTuple
end

"""
    unknown_terms(result::DiscoveryRun) -> Vector{UnknownTermResult}

The per-term results, in node order.
"""
unknown_terms(result::DiscoveryRun) = result.terms

Base.length(result::DiscoveryRun) = length(result.terms)
function Base.iterate(result::DiscoveryRun, state = 1)
    state > length(result.terms) ? nothing : (result.terms[state], state + 1)
end
Base.eltype(::Type{DiscoveryRun{M, P, V, X}}) where {M, P, V, X} = eltype(V)
Base.keys(result::DiscoveryRun) = [term.node for term in result.terms]
Base.getindex(result::DiscoveryRun, i::Integer) = result.terms[i]
function Base.getindex(result::DiscoveryRun, node::Symbol)
    for term in result.terms
        term.node == node && return term
    end
    throw(KeyError(node))
end
function Base.haskey(result::DiscoveryRun, node::Symbol)
    any(t -> t.node == node, result.terms)
end

"""
    CROSS_TERM_COLLINEARITY_THRESHOLD

Cross-term collinearity above which `discover_unknown_terms` warns that the
scales of a pair of unknown terms are not separately determined. Set from
the multi-term study of 0.16 (benchmarks page, "Two unknown terms"): over
35 multi-unknown runs (two two-term fixtures at noise 0, 0.02 and 0.05 and a
three-term fixture at noise 0, five seeds each) every run in which the extra
term had no measurable cost had all its pairs at or below 0.425, and every
run with a measurable cost (a learned rate 1.4 to 6.3 times further from the
truth than the single-unknown control, or a lost support) had at least one
pair at or above 0.498; the threshold is the midpoint of that gap.
"""
const CROSS_TERM_COLLINEARITY_THRESHOLD = 0.46

# -- Report -------------------------------------------------------------------

"""
    report_unknown_terms(result::DiscoveryRun) -> String

The report of a `discover_unknown_terms` result. With one unknown term it is
the four-section report (identifiability, fit, discovery, reproduction) that
`discover_unknown_term` printed in 0.15, unchanged. With several terms it
prints one shared fit section, then one block per term with that term's
identifiability and discovery sections, then the cross-term section, then
the reproduction section.
"""
function report_unknown_terms(result::DiscoveryRun)
    settings = result.settings
    residuals = result.residuals
    holdout = isempty(result.holdout_indices) ? nothing : residuals.data_residual_holdout
    if length(result.terms) == 1
        term = only(result.terms)
        return format_protocol_result(term.identifiability;
            residual = residuals.data_residual,
            residual_train = residuals.data_residual_train,
            residual_holdout = holdout,
            equations = term.discovery.equations,
            extras = term.extras,
            unknown_holes = settings.unknown_holes,
            seed = settings.seed,
            n_ics = settings.n_ics,
            n_points = settings.n_points,
            adam_iters = settings.adam_iters,
            bfgs_iters = settings.bfgs_iters,
            bootstrap = settings.bootstrap,
            discovery_seed = settings.discovery_seed)
    end
    io = IOBuffer()
    println(io, "FIT")
    println(io, "  unknown_terms: ", join(string.(keys(result)), ", "))
    println(io, "  hybrid_data_residual: ", _format_protocol_value(residuals.data_residual))
    println(io, "  hybrid_data_residual_train: ",
        _format_protocol_value(residuals.data_residual_train))
    holdout === nothing ||
        println(io, "  hybrid_data_residual_holdout: ", _format_protocol_value(holdout))
    for term in result.terms
        print(io, _format_term_block(term))
    end
    println(io, "CROSS-TERM")
    for pair in result.cross_term
        value = pair.collinearity
        flag = isfinite(value) && value > CROSS_TERM_COLLINEARITY_THRESHOLD
        println(io, "  ", pair.nodes[1], ", ", pair.nodes[2], ": collinearity ",
            _format_protocol_value(value),
            flag ?
            " (above $(CROSS_TERM_COLLINEARITY_THRESHOLD): scaling one term looks like scaling the other in these data; their scales are not separately determined)" :
            "")
    end
    println(io,
        "  local diagnostic (cosine of the two terms' trajectory sensitivities); not a structural identifiability proof")
    println(io, "REPRODUCTION")
    println(io, "  seed: ", _format_protocol_value(settings.seed))
    println(io, "  n_ics: ", _format_protocol_value(settings.n_ics))
    println(io, "  n_points: ", _format_protocol_value(settings.n_points))
    println(io, "  adam_iters: ", _format_protocol_value(settings.adam_iters))
    println(io, "  bfgs_iters: ", _format_protocol_value(settings.bfgs_iters))
    println(io, "  bootstrap: ", _format_protocol_value(settings.bootstrap))
    println(io, "  discovery_seed: ", _format_protocol_value(settings.discovery_seed))
    println(io, "  unknown_terms: ", settings.unknown_holes)
    return String(take!(io))
end

function _format_term_block(term::UnknownTermResult)
    ident = term.identifiability
    edge = ident.unidentifiable_edge
    production = hasproperty(ident, :production_param) ? ident.production_param : :k_prod
    names = [node.name for node in term.network.nodes]
    regulators = join(string.(names[state_nodes(term.network)[term.term.regulators]]), ", ")
    io = IOBuffer()
    println(io, "TERM ", term.node, " (unknown destruction of ", term.node,
        ", regulators ", regulators, ")")
    println(io, "IDENTIFIABILITY")
    println(io, "  unidentifiable_edge: ", edge)
    println(io, "  coefficients_are_biological_constants: ", !edge)
    println(io, "  production_param: ", production)
    if edge
        println(io, "  the production rate (", production,
            ") and the scale of the unknown term are not separately identifiable from these data")
    else
        println(io, "  no scale collinearity warning was raised for ", production)
    end
    if hasproperty(ident, :collinearity) && isfinite(ident.collinearity)
        println(io, "  collinearity: ", _format_protocol_value(float(ident.collinearity)))
    end
    println(io, "DISCOVERY")
    println(io, "  equations:")
    println(io, _format_protocol_equations(term.discovery.equations))
    println(io, "  extras: ", _format_protocol_extras(term.extras))
    return String(take!(io))
end

function Base.show(io::IO, ::MIME"text/plain", result::DiscoveryRun)
    print(io, report_unknown_terms(result))
end

function Base.show(io::IO, result::DiscoveryRun)
    print(io, "DiscoveryRun(unknown terms = ", keys(result),
        ", experiments = ", result.settings.n_ics,
        ", held out = ", length(result.holdout_indices),
        ", discovery ", all(t -> t.discovery.success, result.terms) ? "succeeded" :
                        "failed",
        ", hybrid_data_residual = ",
        _format_protocol_value(result.residuals.data_residual), ")")
end

function Base.show(io::IO, ::MIME"text/plain", term::UnknownTermResult)
    print(io, _format_term_block(term))
end

function Base.show(io::IO, term::UnknownTermResult)
    print(io, "UnknownTermResult(", repr(term.node), ", discovery ",
        term.discovery.success ? "succeeded" : "failed", ")")
end

# -- Migration errors ---------------------------------------------------------

const MIGRATION_SECTION = "https://utkuyilmaz1903.github.io/HybridKinetics.jl/stable/howto/#Migrating-from-0.15"

"""
    discover_unknown_term(args...; kwargs...)

Removed in 0.16. Call `discover_unknown_terms(network, experiments; ...)`,
which returns a `DiscoveryRun`; with one unknown term its numbers
are identical and its report is the same text. Per-term contents are
`result[:S]` (by node) or `result[1]`. See the migration section of the
how-to page.
"""
function discover_unknown_term(args...; kwargs...)
    throw(ErrorException(string(
        "discover_unknown_term was removed in HybridKinetics 0.16. Call ",
        "discover_unknown_terms(network, experiments; ...) instead; it takes the same ",
        "keywords except `term` (all unknown terms are discovered) and returns an ",
        "DiscoveryRun whose per-term contents are result[:node]. ",
        "Migration: ", MIGRATION_SECTION)))
end

"""
    report_unknown_term(args...; kwargs...)

Removed in 0.16. Call `report_unknown_terms(result)`; with one unknown term
it returns the same text. See the migration section of the how-to page.
"""
function report_unknown_term(args...; kwargs...)
    throw(ErrorException(string(
        "report_unknown_term was removed in HybridKinetics 0.16. Call ",
        "report_unknown_terms(result) instead; with one unknown term it prints the ",
        "same report. Migration: ", MIGRATION_SECTION)))
end

# -- The entry point ----------------------------------------------------------

function _unknown_term_subset(set::ExperimentSet, indices)
    return ExperimentSet(
        [set.experiments[i] for i in indices], set.state_names;
        units = set.units, metadata = set.metadata)
end

_per_term(value, node, n) = value
function _per_term(value::AbstractDict, node, n)
    haskey(value, node) && return value[node]
    return nothing
end
function _per_term(value::NamedTuple, node, n)
    n == 1 ? value : (
        haskey(value, node) ? getproperty(value, node) : nothing)
end

function _production_param_for(choice, model::UDEModel, term, node::Symbol, phys_names)
    choice isa Symbol && choice !== :auto && return choice
    if choice isa AbstractDict
        haskey(choice, node) && return choice[node]
        throw(ArgumentError("production_param has no entry for the unknown term on $(node)"))
    end
    :k_prod in phys_names && return :k_prod
    candidates = Symbol[]
    for production in model.compiled.production_terms
        production.target == term.target || continue
        if production isa InputProductionTerm
            push!(candidates, production.rate_param)
        elseif production isa MassActionProductionTerm
            push!(candidates, production.param)
        elseif production isa SaturationProductionTerm
            push!(candidates, production.vmax_param)
        end
    end
    return length(candidates) == 1 ? only(candidates) : :k_prod
end

function _resolve_library(terms::AbstractVector{UnknownTerm}, node, default)
    for spec in terms
        spec.node == node && spec.library !== nothing && return spec.library
    end
    return default
end

"""
    discover_unknown_terms(network, experiments; terms=unknown_terms(network),
                           training=TrainingConfig(adam_iterations=100, bfgs_iterations=50, log_every=10^6),
                           discovery=rate_discovery_config(), holdout=2,
                           rng=MersenneTwister(0), phys_init=nothing, warmup=true,
                           known_support=nothing, stability_selection=nothing,
                           strict=false, seed=nothing, regulator_grid=nothing,
                           verbose=true)

Train the hybrid model of `network` on `experiments`, discover a rational
expression for each of its unknown destruction terms, and return an
`DiscoveryRun`. The steps are those of `examples/unknown_inhibition.jl`,
in the same order and with the same defaults:

1. `build_ude_model(rng, network)`; the network must have at least one
   unknown destruction term (`unknown_terms(network)` lists them; one per
   node). With one term every number below is the one 0.15's
   `discover_unknown_term` produced.
2. Physical parameters start from a flat guess of 0.8 (`phys_init` overrides).
3. A warm-up `train_ude` on the first training experiment, of every unknown
   term at once, with the training config's settings (Adam iterations and
   learning rate, gradient clip, constraint, solver, frozen parameters), no
   BFGS, and the horizon curriculum 35%, 70%, 100% (`warmup = false` skips
   it).
4. `train_experiments` on the training experiments with `training`
   (default: Adam 100 then BFGS 50). One network per unknown term is
   trained jointly; each sees only its own regulators.
5. For each unknown term, in node order: its learned rate is sampled on the
   regulator grid of the training experiments
   (`sample_unknown_destruction_grid`) and `regress_unknown_rate` fits a
   rational rate with `discovery` (default: the reference protocol's
   configuration, bootstrap 8, discovery seed 3), or with the term's own
   `library` when `terms` carries one. `stability_selection` and `strict`
   are passed through, per term. `regulator_grid` replaces the grid: a vector
   or range (one term), a `Dict` from node name to vector or range, or a
   function `(model, params, training_set, term) -> grid` called per term
   after training, which is the way to sample a regulator that is never
   observed; `nothing` keeps the observed grid.
6. `report_production_destruction_tradeoff` per term on the first training
   experiment, and with two or more terms `cross_term_collinearity` on the
   same experiment; a pair above `CROSS_TERM_COLLINEARITY_THRESHOLD` is
   reported with a warning.
7. `hybrid_data_residual` of the model with every discovered rate substituted
   against the first training experiment, the mean over the training
   experiments, and the mean over the held-out experiments.

`holdout` is the number of experiments at the end of `experiments` that are
held out of training and used only for the held-out residual (the reference
protocol holds out 2 of 9). Residuals and the identifiability diagnostics use
each experiment's observation mask, so unobserved entries do not count.
`known_support`, the true implicit support when the data are synthetic (for
example `HybridKinetics.hill_rate_support(2)`), is used only to list the
extra terms in the report; with several terms give a `Dict` from node name
to support. `production_param` names the production rate parameter the per-term
identifiability diagnostic compares each term's scale against: a `Symbol`
for every term, a `Dict` from node name to `Symbol`, or `:auto` (the
default), which is `:k_prod` when the model has that parameter (the 0.15
behaviour) and otherwise the rate parameter of the one production reaction
of the term's node, when there is exactly one. `seed` is recorded in the
report and not used otherwise; `verbose` prints training progress and the
report.
"""
function discover_unknown_terms(network::BiologicalNetwork, experiments::ExperimentSet;
        terms::AbstractVector{UnknownTerm} = unknown_terms(network),
        training::TrainingConfig = TrainingConfig(
            adam_iterations = REFERENCE_PROTOCOL.adam_iterations,
            bfgs_iterations = REFERENCE_PROTOCOL.bfgs_iterations,
            log_every = 10^6),
        discovery::DiscoveryConfig = rate_discovery_config(),
        holdout::Integer = 2,
        rng::AbstractRNG = MersenneTwister(0),
        phys_init = nothing,
        warmup::Bool = true,
        known_support = nothing,
        stability_selection::Union{Nothing, StabilitySelection} = nothing,
        strict::Bool = false,
        seed = nothing,
        regulator_grid = nothing,
        production_param = :auto,
        verbose::Bool = true)
    n = length(experiments.experiments)
    n ≥ 1 || throw(ArgumentError("experiments must contain at least one experiment"))
    0 ≤ holdout ≤ n - 1 || throw(ArgumentError(
        "holdout must be between 0 and $(n - 1) for $(n) experiments; got $(holdout)"))
    specs = unknown_terms(network)
    isempty(specs) && throw(ArgumentError(
        "the network has no unknown destruction term; mark one with ReactionSpec(known = false) or BiologicalNetwork(...; unknown = [UnknownTerm(:node)])"))
    for spec in terms
        any(s -> s.node == spec.node, specs) || throw(ArgumentError(
            "terms names $(spec.node), which is not an unknown term of the network ($(join(string.(getfield.(specs, :node)), ", ")))"))
    end
    training_indices = collect(1:(n - holdout))
    holdout_indices = collect((n - holdout + 1):n)
    train_set = holdout == 0 ? experiments :
                _unknown_term_subset(experiments, training_indices)
    holdout_set = holdout == 0 ? nothing :
                  _unknown_term_subset(experiments, holdout_indices)

    model, p0 = build_ude_model(rng, network)
    neural = neural_destruction_terms(model)
    length(neural) == length(specs) || throw(ErrorException(
        "the compiled model has $(length(neural)) neural terms but the network lists $(length(specs)) unknown terms"))
    n_terms = length(neural)
    if n_terms > 1 && known_support !== nothing && !(known_support isa AbstractDict)
        throw(ArgumentError(
            "with $(n_terms) unknown terms known_support must be a Dict from node name to support"))
    end
    if n_terms > 1 && regulator_grid !== nothing &&
       !(regulator_grid isa AbstractDict || regulator_grid isa Function)
        throw(ArgumentError(
            "with $(n_terms) unknown terms regulator_grid must be a Dict from node name to grid, or a function"))
    end
    phys_names = Tuple(parameter_schema(model).phys_names)
    guess = phys_init === nothing ?
            NamedTuple{phys_names}(ntuple(_ -> 0.8, length(phys_names))) : phys_init
    ude_init = pack_parameters(guess, p0.nn)
    first_exp = first(train_set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    start = ude_init
    if warmup
        warm = train_ude(
            ude_init, first_exp.observations, first_exp.times, first_exp.u0, tspan, model;
            config = TrainingConfig(training; bfgs_iterations = 0,
                horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0])),
            verbose = verbose, mask = first_exp.mask)
        start = warm.params
    end
    trained = train_experiments(start, train_set, model; config = training,
        verbose = verbose)
    isfinite(trained.final_loss) ||
        @warn "the joint training loss is not finite; the discovered rates will not be reliable" final_loss=trained.final_loss

    node_names = [node.name for node in network.nodes]
    state_ids = state_nodes(network)
    backend = discovery.backend
    settings = (;
        unknown_holes = count_unknown_destructions(model),
        seed = seed,
        n_ics = n,
        n_points = size(first_exp.observations, 2),
        adam_iters = training.adam_iterations,
        bfgs_iters = training.bfgs_iterations,
        bootstrap = backend isa ImplicitSINDyPI ? backend.bootstrap_samples : nothing,
        discovery_seed = Int(discovery.seed),
        holdout = holdout,
        warmup = warmup,
        regulator_grid = regulator_grid === nothing ? :observed :
                         regulator_grid isa Function ? :function : :given)

    per_term = UnknownTermResult[]
    pairs = Pair{Any, Any}[]
    all_found = true
    for chosen in neural
        node = node_names[state_ids[chosen.target]]
        config = _resolve_library(terms, node, discovery)
        grid = _per_term(regulator_grid, node, n_terms)
        r_range = grid === nothing ? _regulator_grid(train_set, chosen) :
                  grid isa Function ? grid(model, trained.params, train_set, chosen) : grid
        R, D, chosen = sample_unknown_destruction_grid(model, trained.params, chosen;
            r_range = r_range)
        times_grid = collect(range(0.0, 1.0; length = size(R, 2)))
        found = regress_unknown_rate(R, times_grid, D;
            config = config, verbose = false, strict = strict,
            stability_selection = stability_selection)
        prod_param = _production_param_for(
            production_param, model, chosen, node, phys_names)
        ident = report_production_destruction_tradeoff(
            model, trained.params, first_exp.observations, first_exp.times,
            first_exp.u0, tspan; term = chosen, verbose = false, mask = first_exp.mask,
            production_param = prod_param)
        extras = nothing
        frequency = nothing
        if found.success && !isempty(found.candidates)
            candidate = found.candidates[1]
            push!(pairs, chosen => equation_to_function(candidate))
            support = _per_term(known_support, node, n_terms)
            support === nothing || (extras = discovered_support_extras(
                candidate, support.numerator, support.denominator))
            candidate isa ImplicitCandidate &&
                (frequency = collect(Float64, candidate.selection_frequency))
        else
            all_found = false
            output = D
            all(iszero, output) &&
                @warn "the learned rate of the unknown term on $(node) is zero everywhere on its grid"
        end
        push!(per_term,
            UnknownTermResult(node, network, model, trained.params, trained,
                chosen, ident, found, (; R, D), frequency, extras, settings))
    end

    residual = Inf
    residual_train = Inf
    residual_holdout = holdout == 0 ? NaN : Inf
    if all_found
        function residual_of(e)
            hybrid_data_residual(model, trained.params, pairs,
                e.u0, (first(e.times), last(e.times)), e.times, e.observations;
                mask = e.mask)
        end
        residual = residual_of(first_exp)
        residual_train = mean(residual_of(e) for e in train_set.experiments)
        holdout == 0 || (residual_holdout = mean(residual_of(e)
        for e in holdout_set.experiments))
    end

    # Empty with one unknown term (no simulation is run), so the single-term
    # path stays exactly the 0.15 sequence of calls.
    cross = cross_term_collinearity(model, trained.params, first_exp.u0, tspan,
        first_exp.times; mask = first_exp.mask)
    for pair in cross
        if isfinite(pair.collinearity) &&
           pair.collinearity > CROSS_TERM_COLLINEARITY_THRESHOLD
            @warn string("the unknown terms on ", pair.nodes[1], " and ", pair.nodes[2],
                " have cross-term collinearity ", _format_protocol_value(pair.collinearity),
                " (above ", CROSS_TERM_COLLINEARITY_THRESHOLD,
                "): in these data scaling one looks like scaling the other, so their ",
                "scales, and the coefficients discovered for them, are not separately determined")
        end
    end

    result = DiscoveryRun(
        network, model, trained.params, trained, per_term,
        (; data_residual = Float64(residual),
            data_residual_train = Float64(residual_train),
            data_residual_holdout = Float64(residual_holdout)),
        cross, training_indices, holdout_indices, settings)
    verbose && println(report_unknown_terms(result))
    return result
end
