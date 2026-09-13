"""Recovery fixtures and metrics for the scientific wedge (graph-local UDE + discovery)."""

function relative_parameter_error(estimate, truth::NamedTuple)
    names = collect(keys(truth))
    rel = Dict{Symbol, Float64}()
    squares = Float64[]
    for name in names
        fitted = positive_parameter(getproperty(estimate.phys, name))
        target = float(getproperty(truth, name))
        err = abs(fitted - target) / max(abs(target), eps(target))
        rel[name] = err
        push!(squares, abs2(err))
    end
    return sqrt(mean(squares)), rel
end

"""
    locked_ude_kpis(ude)

Named locked UDE claim fields: hybrid `data_residual`, true-monomial
`support_recall`, and `unidentifiable_edge`. Combined F1 is skeleton
context, not the claim. Not exported.
"""
function locked_ude_kpis(ude)
    ident = hasproperty(ude, :identifiability) ? ude.identifiability : nothing
    edge = ident !== nothing && hasproperty(ident, :unidentifiable_edge) ?
           ident.unidentifiable_edge : false
    extras = hasproperty(ude, :extras) ? getproperty(ude, :extras) : nothing
    f1 = hasproperty(ude, :support_f1) ? ude.support_f1 : 0.0
    return (;
        data_residual = ude.data_residual,
        support_recall = ude.support_recall,
        unidentifiable_edge = edge,
        support_f1 = f1,
        extras,
        claim = :recall_plus_data_residual)
end

function _format_protocol_value(value)
    value === nothing && return "NA"
    value isa AbstractFloat && return string(round(value; sigdigits = 4))
    return string(value)
end

function _format_protocol_extras(extras)
    extras === nothing && return "NA"
    extras isa AbstractString && return extras
    isempty(extras) && return "(none)"
    return join(string.(extras), ", ")
end

function _format_protocol_equations(equations)
    equations === nothing && return "(none)"
    return rstrip(string(equations))
end

"""Plain-language summary of the three acceptance criteria of the reference protocol."""
function _acceptance_criteria_label()
    t = RECOVERY_THRESHOLDS
    return string("scale warning raised, hybrid residual at most ", t.data_residual,
        ", support recall at least ", t.support_recall)
end

"""
    format_protocol_result(ident; kwargs...) -> String

Format the printed recovery report. The four sections are printed in the
order identifiability, fit, discovery, reproduction, matching the field order
of `build_protocol_result`. Floating-point values are shown with four
significant digits. Not exported.
"""
function format_protocol_result(ident;
        residual = nothing,
        residual_train = nothing,
        residual_holdout = nothing,
        equations = nothing,
        extras = nothing,
        support_f1 = nothing,
        support_recall = nothing,
        unknown_holes::Integer = 1,
        seed = nothing,
        n_ics = nothing,
        adam_iters = nothing,
        bfgs_iters = nothing,
        bootstrap = nothing,
        discovery_seed = nothing,
        n_points = nothing,
        protocol_kind = nothing,
        smoke::Bool = false)
    edge = ident.unidentifiable_edge
    production = hasproperty(ident, :production_param) ?
                 ident.production_param : :k_prod
    kind = protocol_kind === nothing ? (smoke ? :smoke : :protocol) : protocol_kind
    io = IOBuffer()
    println(io, "IDENTIFIABILITY")
    println(io, "  unknown_holes: ", unknown_holes)
    println(io, "  unidentifiable_edge: ", edge)
    println(io, "  coefficients_are_biological_constants: ", !edge)
    println(io, "  production_param: ", production)
    if edge
        println(io, "  the production rate (", production,
            ") and the scale of the unknown term are not separately identifiable from these data")
    else
        println(io, "  no scale collinearity warning was raised for ", production)
    end
    println(io,
        "  local diagnostic (Fisher condition number and scale collinearity); not a structural identifiability proof")
    if hasproperty(ident, :collinearity) && isfinite(ident.collinearity)
        println(io, "  collinearity: ", _format_protocol_value(float(ident.collinearity)))
    end
    println(io, "FIT")
    println(io, "  hybrid_data_residual: ", _format_protocol_value(residual))
    residual_train === nothing ||
        println(
            io, "  hybrid_data_residual_train: ", _format_protocol_value(residual_train))
    residual_holdout === nothing ||
        println(io, "  hybrid_data_residual_holdout: ",
            _format_protocol_value(residual_holdout))
    recall = support_recall === nothing ?
             "not scored (needs the synthetic ground truth)" :
             _format_protocol_value(support_recall)
    println(io, "  support_recall: ", recall)
    println(io, "DISCOVERY")
    println(io, "  equations:")
    println(io, _format_protocol_equations(equations))
    f1 = support_f1 === nothing ?
         "not scored (needs the synthetic ground truth)" :
         _format_protocol_value(support_f1)
    println(io, "  support_f1: ", f1)
    println(io, "  extras: ", _format_protocol_extras(extras))
    println(io, "  canonical_hill_from_nn: false")
    println(io, "  acceptance_criteria: ", _acceptance_criteria_label())
    println(io, "REPRODUCTION")
    println(io, "  seed: ", _format_protocol_value(seed))
    println(io, "  n_ics: ", _format_protocol_value(n_ics))
    println(io, "  n_points: ", _format_protocol_value(n_points))
    println(io, "  adam_iters: ", _format_protocol_value(adam_iters))
    println(io, "  bfgs_iters: ", _format_protocol_value(bfgs_iters))
    println(io, "  bootstrap: ", _format_protocol_value(bootstrap))
    println(io, "  discovery_seed: ", _format_protocol_value(discovery_seed))
    println(io, "  protocol_kind: ", kind)
    println(io, "  smoke: ", smoke)
    return String(take!(io))
end

"""
    build_protocol_result(ude)

Reference-protocol result object. Field order is the claim: identifiability
first, then fit, then discovery. Not exported. Does not replace
`locked_ude_kpis` or existing `ude_discovery` fields.
"""
function build_protocol_result(ude; unknown_holes::Integer = 1)
    ident = hasproperty(ude, :identifiability) ? ude.identifiability : nothing
    edge = ident !== nothing && hasproperty(ident, :unidentifiable_edge) ?
           ident.unidentifiable_edge : false
    extras = hasproperty(ude, :extras) ? getproperty(ude, :extras) : nothing
    f1 = hasproperty(ude, :support_f1) ? ude.support_f1 : 0.0
    holes = unknown_holes
    return (;
        unknown_holes = holes,
        unidentifiable_edge = edge,
        coefficients_are_biological_constants = !edge,
        data_residual = ude.data_residual,
        support_recall = ude.support_recall,
        support_f1 = f1,
        extras,
        canonical_hill_from_nn = false,
        claim = :recall_plus_data_residual)
end

"""
    reference_protocol_kpis_hold(kpis) -> Bool

True when the locked UDE claim holds: `unidentifiable_edge`, hybrid residual
versus data, and true-monomial recall. Combined F1 is not an acceptance criterion.
Not exported.
"""
function reference_protocol_kpis_hold(kpis)
    return kpis.unidentifiable_edge === true &&
           kpis.data_residual ≤ RECOVERY_THRESHOLDS.data_residual &&
           kpis.support_recall ≥ RECOVERY_THRESHOLDS.support_recall
end

"""
    assert_reference_protocol_residual(residual)

Reference-example residual threshold. Same number as `RECOVERY_THRESHOLDS.data_residual`.
Not exported.
"""
function assert_reference_protocol_residual(residual)
    residual ≤ RECOVERY_THRESHOLDS.data_residual ||
        throw(ErrorException(
            "hybrid residual $(residual) exceeds RECOVERY_THRESHOLDS.data_residual"))
    return residual
end

"""
    RECOVERY_THRESHOLDS

Recovery thresholds. Loosening a threshold is a breaking change.
UDE combined F1 is not the analytical Hill threshold; that threshold is `support_f1_clean`
on exact/noisy analytical `D`. A same-library F1 attempt
(`benchmark/ude_f1_attempt.jl`) left extras `1` and `r`. The UDE claim is
recall + data residual until a new major version.
"""
const RECOVERY_THRESHOLDS = (
    nn_correlation = 0.90,
    nn_rate_rmse = 0.12,
    support_f1_clean = 0.99,
    support_f1_ude = 0.50,
    support_f1_noisy = 0.50,
    support_recall = 0.99,
    discovered_rate_rmse = 0.20,
    data_residual = 0.30
)

"""
    REFERENCE_PROTOCOL

Hyperparameters shared by `examples/unknown_inhibition.jl` and the UDE
recovery job. Not a threshold table. Not exported.

Reproduction fingerprint: seed 103, `n_ics = 9`, `n_points = 50`.
Smoke is a different object (`smoke_n_ics = 1`, `smoke_n_points = 8`)
and is not this protocol.
"""
const REFERENCE_PROTOCOL = (
    seed = 103,
    adam_iterations = 100,
    bfgs_iterations = 50,
    n_points = 50,
    smoke_n_points = 8,
    tspan = (0.0, 8.0),
    bootstrap = 8,
    discovery_seed = 3,
    n_ics = 9,
    smoke_n_ics = 1,
    observation_noise = 0.0
)

term_key(term::MonomialTerm) = (Tuple(term.variables), Tuple(term.powers))

function active_support(candidate::ImplicitCandidate; atol::Real = 1e-8)
    num = Set{Tuple{Tuple{Vararg{Int}}, Tuple{Vararg{Int}}}}()
    den = Set{Tuple{Tuple{Vararg{Int}}, Tuple{Vararg{Int}}}}()
    spec = candidate.specification
    for (coefficient, term) in zip(candidate.numerator_coefficients, spec.numerator)
        abs(coefficient) > atol && push!(num, term_key(term))
    end
    for (coefficient, term) in zip(candidate.denominator_coefficients, spec.denominator)
        abs(coefficient) > atol && push!(den, term_key(term))
    end
    return (numerator = num, denominator = den)
end

function active_support(candidate::ExplicitCandidate; atol::Real = 1e-8)
    num = Set{Tuple{Tuple{Vararg{Int}}, Tuple{Vararg{Int}}}}()
    for (coefficient, term) in zip(candidate.coefficients,
        candidate.specification.numerator)
        abs(coefficient) > atol && push!(num, term_key(term))
    end
    return (numerator = num, denominator = Set{eltype(num)}())
end

function support_f1(recovered::Set, truth::Set)
    isempty(truth) && isempty(recovered) &&
        return (; precision = 1.0, recall = 1.0, f1 = 1.0, tp = 0, fp = 0, fn = 0)
    tp = length(intersect(recovered, truth))
    fp = length(setdiff(recovered, truth))
    fn = length(setdiff(truth, recovered))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
    return (; precision, recall, f1, tp, fp, fn)
end

function support_f1(candidate, truth_num::Set, truth_den::Set; atol::Real = 1e-8)
    recovered = active_support(candidate; atol = atol)
    combined_rec = union(
        Set((:n, key) for key in recovered.numerator),
        Set((:d, key) for key in recovered.denominator))
    combined_truth = union(
        Set((:n, key) for key in truth_num),
        Set((:d, key) for key in truth_den))
    return (;
        numerator = support_f1(recovered.numerator, truth_num),
        denominator = support_f1(recovered.denominator, truth_den),
        combined = support_f1(combined_rec, combined_truth))
end

function rate_rel_rmse(estimate, truth)
    estimate_vec = vec(Float64.(estimate))
    truth_vec = vec(Float64.(truth))
    length(estimate_vec) == length(truth_vec) ||
        throw(DimensionMismatch("estimate and truth must have the same length"))
    scale = max(sqrt(mean(abs2, truth_vec)), eps(Float64))
    return sqrt(mean(abs2, estimate_vec .- truth_vec)) / scale
end

function denominator_violation_count(candidate::ImplicitCandidate, X;
        floor::Real = 1e-8)
    _, denominator = _evaluate_candidate(
        candidate.specification,
        candidate.numerator_coefficients,
        candidate.denominator_coefficients, X)
    return count(<(floor), denominator)
end

"""Explicit candidates have no rational denominator; the count is 0."""
function denominator_violation_count(::ExplicitCandidate, X;
        floor::Real = 1e-8)
    return 0
end

"""Failed discovery has no candidate; treat as an unbounded violation."""
function denominator_violation_count(::Nothing, X; floor::Real = 1e-8)
    return typemax(Int)
end

"""
    denominator_split_counts(candidate, train_X, val_X, domain_X; floor=1e-8)

Count denominator violations on the train slice, the validation slice,
and the orthant domain grid separately. Combined F1 is not scored here.
"""
function denominator_split_counts(candidate, train_X, val_X, domain_X;
        floor::Real = 1e-8)
    train = denominator_violation_count(candidate, train_X; floor = floor)
    val = denominator_violation_count(candidate, val_X; floor = floor)
    domain = denominator_violation_count(candidate, domain_X; floor = floor)
    return (;
        train, val, domain,
        total = train + val + domain,
        any = (train + val + domain) > 0)
end

"""
    ude_extras_denominator_row(candidate, R_grid; extras, floor, domain_samples)

Apply `denominator_split_counts` on the UDE extras path. Live extras
do not skip the train / val / orthant grid. Hardcoded F1-attempt
extras strings stay rejected.
"""
function ude_extras_denominator_row(candidate, R_grid;
        extras = String[], floor::Real = 1e-8,
        domain_samples::Int = 32, seed::Integer = 42)
    n = size(R_grid, 2)
    n_val = n ≤ 2 ? 0 : clamp(round(Int, 0.2 * n), 1, n - 1)
    train_X = n_val == 0 ? R_grid : @view R_grid[:, 1:(n - n_val)]
    val_X = n_val == 0 ? R_grid : @view R_grid[:, (n - n_val + 1):n]
    domain_X = _denominator_domain_grid(R_grid; n = domain_samples, seed = seed)
    split = denominator_split_counts(
        candidate, train_X, val_X, domain_X; floor = floor)
    extras_label = extras_print_label(extras)
    return merge(split,
        (;
            extras,
            extras_label,
            extras_live = extras !== nothing && !isempty(extras),
            hardcoded = extras_print_is_hardcoded_attempt(extras_label),
            n_train = size(train_X, 2),
            n_val = size(val_X, 2),
            n_domain = size(domain_X, 2),
            holds = extras_print_is_hardcoded_attempt(extras_label) == false))
end

function support_uses_variable(candidate; variable::Int, atol::Real = 1e-8)
    recovered = active_support(candidate; atol = atol)
    keys = union(recovered.numerator, recovered.denominator)
    return any(key -> variable in key[1], keys)
end

"""True when a discovered candidate support contains `variable`."""
function local_has_true_parent_check(candidate; variable::Int, atol::Real = 1e-8)
    candidate === nothing && return false
    return support_uses_variable(candidate; variable = variable, atol = atol)
end

"""True when a discovered candidate support contains any of `variables`."""
function local_has_false_parent_check(candidate; variables, atol::Real = 1e-8)
    candidate === nothing && return false
    return any(v -> support_uses_variable(candidate; variable = Int(v), atol = atol),
        variables)
end

"""True implicit support for `D = vmax r^n / (K^n + r^n)` on variable `variable`."""
function hill_rate_support(order::Int; variable::Int = 1)
    key = ((variable,), (order,))
    return (numerator = Set([key]), denominator = Set([key]))
end

"""True implicit support for `D = vmax r / (km + r)` on variable `variable`."""
function mm_rate_support(; variable::Int = 1)
    key = ((variable,), (1,))
    return (numerator = Set([key]), denominator = Set([key]))
end

hill_rate_truth(r; vmax, K, n) = vmax .* (r .^ n) ./ (K^n .+ r .^ n)
mm_rate_truth(r; vmax, km) = vmax .* r ./ (km .+ r)

"""Human label for an implicit monomial key `((vars,), (powers,))`."""
function monomial_key_label(key)
    vars, powers = key
    isempty(vars) && return "1"
    parts = String[]
    for (variable, power) in zip(vars, powers)
        name = variable == 1 ? "r" : string("x", variable)
        push!(parts, power == 1 ? name : string(name, "^", power))
    end
    return join(parts, "*")
end

"""
    discovered_support_extras(candidate, truth_num, truth_den; atol=1e-8)

Monomial keys present in `candidate` but not in the true Hill/MM support.
Labels match the locked UDE extras (`1`, `r`). Not exported.
"""
function discovered_support_extras(candidate, truth_num, truth_den;
        atol::Real = 1e-8)
    recovered = active_support(candidate; atol = atol)
    truth_keys = union(truth_num, truth_den)
    extras = String[]
    for key in sort!(collect(union(recovered.numerator, recovered.denominator));
        by = string)
        key in truth_keys && continue
        push!(extras, monomial_key_label(key))
    end
    return extras
end

function neural_destruction_terms(model::UDEModel)
    [term for term in model.compiled.destruction_terms if term isa NeuralDestructionTerm]
end

"""
    assert_single_unknown_destruction(model) -> 1

The reference protocol is a single-unknown-term workflow. Throws
`ErrorException` unless `model` has exactly one `NeuralDestructionTerm`.
Not exported. Does not change `validate_network`.
"""
function assert_single_unknown_destruction(model::UDEModel)
    n = length(neural_destruction_terms(model))
    n == 1 || throw(ErrorException(
        "reference protocol requires exactly one unknown destruction D(z); got $n"))
    return n
end

function only_unknown_destruction(model::UDEModel)
    assert_single_unknown_destruction(model)
    return only(neural_destruction_terms(model))
end

"""
    sample_unknown_destruction(model, p, X; term=nothing)

Evaluate compiled neural destruction `D` at trajectory columns of `X`
(the rate used inside `du = P - D·u`, not a raw Lux call).
"""
function sample_unknown_destruction(model::UDEModel, p, X::AbstractMatrix;
        term = nothing)
    terms = neural_destruction_terms(model)
    chosen = term === nothing ? only(terms) : term
    _note_sample_unknown_destruction(model, p, X, chosen)
    regs = chosen.regulators
    n = size(X, 2)
    rates = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        x = @view X[:, j]
        rates[j] = _destruction_contribution(
            chosen, chosen.target, x, p, model.nn, model.st)
    end
    R = Float64.(X[regs, :])
    return R, reshape(rates, 1, :), chosen
end

function sample_unknown_destruction_grid(model::UDEModel, p, term;
        r_range = range(0.05, 2.0; length = 80),
        fill_value = 0.3)
    _note_sample_unknown_destruction_grid(r_range)
    nstates = model.compiled.nstates
    r = collect(r_range)
    X = fill(float(fill_value), nstates, length(r))
    X[term.regulator, :] .= r
    R, D, chosen = sample_unknown_destruction(model, p, X; term = term)
    _note_sample_unknown_destruction_result(r_range, R, D, chosen, p)
    return R, D, chosen
end

"""Single-state network used to discover a scalar rate `D(r)`."""
build_rate_discovery_network() = BiologicalNetwork([NodeSpec(name = :r)], EdgeSpec[])

"""Two-state rate network: `r` plus unused distractor `z` for global ablations."""
build_rate_ablation_network() = BiologicalNetwork(
    [NodeSpec(name = :r), NodeSpec(name = :z)], EdgeSpec[])

function reference_protocol_discovery_config(; kwargs...)
    return rate_discovery_config(;
        bootstrap = REFERENCE_PROTOCOL.bootstrap,
        seed = REFERENCE_PROTOCOL.discovery_seed,
        kwargs...)
end

function rate_discovery_config(; threshold = 1e-3, degree = 2,
        bootstrap = REFERENCE_PROTOCOL.bootstrap,
        scope::Symbol = :graph,
        seed = REFERENCE_PROTOCOL.discovery_seed)
    return DiscoveryConfig(
        backend = ImplicitSINDyPI(
            threshold = threshold, max_degree = degree, max_hill_degree = degree,
            bootstrap_samples = bootstrap, validation_fraction = 0.2,
            domain_samples = 32, chunk_size = 32),
        include_interactions = false, seed = seed, basis_scope = scope)
end

function _permute_rate_samples(X::AbstractMatrix, dX::AbstractMatrix, seed)
    n = size(X, 2)
    perm = randperm(MersenneTwister(UInt64(seed) ⊻ 0x9e3779b97f4a7c15), n)
    return X[:, perm], dX[:, perm]
end

function _rate_network_from_samples(R::AbstractMatrix)
    n = size(R, 1)
    n ≥ 1 || throw(ArgumentError("rate samples must have at least one regulator row"))
    nodes = [NodeSpec(name = Symbol(:r, i)) for i in 1:n]
    edges = [EdgeSpec(source = i, target = 1, kind = INHIBITION,
                 family = HILL, known = false,
                 metadata = EmptyMetadata()) for i in 2:n]
    return _trusted_rate_network(nodes, edges)
end

function _trusted_rate_network(nodes::Vector{NodeSpec}, edges::Vector{EdgeSpec})
    g = SimpleDiGraph(length(nodes))
    interactions = Dict{Tuple{Int, Int}, EdgeSpec}()
    for edge in edges
        key = (edge.source, edge.target)
        add_edge!(g, edge.source, edge.target)
        interactions[key] = edge
    end
    names = Dict(i => String(node.name) for (i, node) in pairs(nodes))
    kinds = Dict(key => edge.kind for (key, edge) in interactions)
    return BiologicalNetwork(g, nodes, interactions, ReactionSpec[], names, kinds)
end

"""
    regress_unknown_rate(R, times, D; network, config, ...)

Discover a scalar destruction rate `D(r)` with graph-local implicit SINDy-PI.
`R` is `n_regulators × n` and `D` is `1 × n`. `stability_selection =
StabilitySelection()` enables the optional pruning stage (off by default).
"""
function regress_unknown_rate(R::AbstractMatrix, times, D::AbstractMatrix;
        network = nothing,
        config::DiscoveryConfig = rate_discovery_config(),
        verbose::Bool = false, strict::Bool = false,
        stability_selection::Union{Nothing, StabilitySelection} = nothing)
    observed = _note_rate_discovery_entry(R, times, D, config)
    observed !== nothing && return observed
    net = network === nothing ? _rate_network_from_samples(R) : network
    nreg = size(R, 1)
    D_full = if size(D, 1) == nreg
        D
    else
        padded = zeros(eltype(D), nreg, size(R, 2))
        padded[1, :] .= vec(D)
        padded
    end
    R_perm, D_perm = _permute_rate_samples(R, D_full, config.seed)
    return discover_equations(
        R_perm, times, net; derivatives = D_perm, targets = 1,
        config = config, verbose = verbose, strict = strict,
        stability_selection = stability_selection)
end

"""
    compose_hybrid_rhs(model, p, term, rate_fn)

ODE right-hand side that keeps compiled known terms and replaces neural
destruction `term` with `rate_fn` of the regulator vector.
"""
function compose_hybrid_rhs(model::UDEModel, p, term::NeuralDestructionTerm, rate_fn)
    return function (u, _, t)
        du = ude_rhs(u, p, t, model)::typeof(u)
        nn_D = _destruction_contribution(
            term, term.target, u, p, model.nn, model.st)
        hat_D = rate_fn(_hybrid_regulator_vector(u, term.regulators))
        du[term.target] += (nn_D - hat_D) * u[term.target]
        return du
    end
end

"""
    compose_hybrid_rhs(model, p, pairs)

Right-hand side that keeps the compiled known terms and replaces every
neural destruction term in `pairs` (a vector of `term => rate_fn`) with its
rate function of the regulator vector. With one pair this is the single-term
`compose_hybrid_rhs`.
"""
function compose_hybrid_rhs(model::UDEModel, p, pairs::AbstractVector)
    length(pairs) == 1 &&
        return compose_hybrid_rhs(model, p, first(pairs)[1], first(pairs)[2])
    return function (u, _, t)
        du = ude_rhs(u, p, t, model)::typeof(u)
        for (term, rate_fn) in pairs
            nn_D = _destruction_contribution(
                term, term.target, u, p, model.nn, model.st)
            hat_D = rate_fn(_hybrid_regulator_vector(u, term.regulators))
            du[term.target] += (nn_D - hat_D) * u[term.target]
        end
        return du
    end
end

@inline function _hybrid_regulator_vector(u, regulators)
    n = length(regulators)
    T = eltype(u)
    n == 1 && return T[u[regulators[1]]]
    n == 2 && return T[u[regulators[1]], u[regulators[2]]]
    return T[u[r] for r in regulators]
end

"""
    hybrid_data_residual(model, p, term, rate_fn, u0, tspan, times, data; mask)

RMSE of `compose_hybrid_rhs` versus observations (not versus UDE `ẋ`).
This is a locked UDE claim field.
"""
function hybrid_data_residual(model, p, term, rate_fn, u0, tspan, times, data;
        mask = nothing)
    rhs = compose_hybrid_rhs(model, p, term, rate_fn)
    prob = SciMLBase.ODEProblem(rhs, u0, tspan)
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return Inf
    pred = Array(sol)
    size(pred) == size(data) || return Inf
    if mask === nothing
        return sqrt(mean(abs2, pred .- data))
    end
    size(mask) == size(data) || return Inf
    n = count(mask)
    n == 0 && return Inf
    residual = ifelse.(mask, pred .- data, zero(eltype(pred)))
    return sqrt(sum(abs2, residual) / n)
end

"""
    hybrid_data_residual(model, p, pairs, u0, tspan, times, data; mask)

RMSE of the hybrid right-hand side with several unknown terms (`pairs` is a
vector of `(term, rate_fn)` tuples) versus observations. One pair reduces to the
single-term method.
"""
function hybrid_data_residual(model, p, pairs::AbstractVector, u0, tspan, times, data;
        mask = nothing)
    length(pairs) == 1 && return hybrid_data_residual(
        model, p, first(pairs)[1], first(pairs)[2], u0, tspan, times, data; mask = mask)
    return _hybrid_data_residual(compose_hybrid_rhs(model, p, pairs), u0, tspan, times,
        data, mask)
end

function _hybrid_data_residual(rhs, u0, tspan, times, data, mask)
    prob = SciMLBase.ODEProblem(rhs, u0, tspan)
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return Inf
    pred = Array(sol)
    size(pred) == size(data) || return Inf
    mask === nothing && return sqrt(mean(abs2, pred .- data))
    size(mask) == size(data) || return Inf
    n = count(mask)
    n == 0 && return Inf
    residual = ifelse.(mask, pred .- data, zero(eltype(pred)))
    return sqrt(sum(abs2, residual) / n)
end

"""Scale a sampled destruction rate so `max|D| = 1` (same monomial library; no new atoms)."""
function normalize_destruction_samples(D)
    values = Float64.(D)
    scale = max(maximum(abs, values), eps(Float64))
    return values ./ scale, scale
end

function _unknown_edge_ics()
    ics = [
        [0.25, 0.20], [0.80, 0.35], [0.40, 1.10], [1.20, 0.70], [0.15, 0.90],
        [0.50, 0.15], [0.90, 1.50], [0.20, 0.50], [1.50, 1.20]
    ]
    length(ics) == REFERENCE_PROTOCOL.n_ics || throw(ErrorException(
        "reference-protocol IC table must have $(REFERENCE_PROTOCOL.n_ics) rows; got $(length(ics))"))
    return ics
end

function _train_unknown_edge(rng, ude_model, ude_p0, truth_net, truth_params;
        adam, bfgs, noise_σ, tspan, n_points,
        frozen_phys::Vector{Symbol} = Symbol[],
        phys_init = nothing)
    _note_train_unknown_edge()
    set = generate_recovery_experiments(
        rng, truth_net, truth_params;
        tspan = tspan, n_points = n_points, noise_σ = noise_σ)
    split = reference_protocol_experiment_split(set)
    fit = fit_unknown_destruction(
        ude_model, ude_p0, split.train;
        adam = adam, bfgs = bfgs,
        frozen_phys = frozen_phys, phys_init = phys_init)
    return fit, set
end

function _regulator_grid(set::ExperimentSet, term; npoints::Int = 80)
    # Only observed, finite regulator values set the grid: a masked or NaN
    # entry would otherwise make the whole grid NaN.
    r = term.regulator
    values = Float64[]
    for exp in set.experiments
        obs = exp.observations
        for j in axes(obs, 2)
            exp.mask[r, j] && isfinite(obs[r, j]) && push!(values, Float64(obs[r, j]))
        end
    end
    isempty(values) && throw(ArgumentError(
        "regulator state $(r) has no finite observed values in the training experiments"))
    lo, hi = extrema(values)
    span = max(hi - lo, 0.1)
    start = max(0.05, lo - 0.1 * span)
    stop = hi + 0.1 * span
    return range(start, stop; length = npoints)
end

"""
    _reference_protocol_rate_recovery(ude_model, ude_params, term, truth_rate, set;
                               order, family, noise_σ, data_residual_fn)

Reference-protocol production domain + composer step. The discovery domain is
exactly `_regulator_grid(split.train, term)`. The composer signature is
unchanged and does not receive `split` or `holdout`. Not exported.
"""
function _reference_protocol_rate_recovery(ude_model, ude_params, term, truth_rate, set;
        order, family::Symbol, noise_σ,
        data_residual_fn)
    split = reference_protocol_experiment_split(set)
    return _evaluate_unknown_rate_recovery(
        ude_model, ude_params, term, truth_rate;
        order = order, family = family, noise_σ = noise_σ,
        r_range = _regulator_grid(split.train, term),
        data_residual_fn = data_residual_fn)
end

function _evaluate_unknown_rate_recovery(ude_model, ude_params, term, truth_rate;
        order, family::Symbol, noise_σ,
        data_residual_fn,
        r_range = range(0.05, 2.0; length = 80))
    observed = _note_evaluate_unknown_rate_recovery_range(r_range)
    observed !== nothing && return observed
    R_grid, D_nn, _ = sample_unknown_destruction_grid(
        ude_model, ude_params, term; r_range = r_range)
    r = vec(R_grid)
    D_true = truth_rate(r)
    nn_corr = cor(vec(D_nn), D_true)
    nn_corr = isnan(nn_corr) ? 0.0 : nn_corr
    nn_rmse = rate_rel_rmse(D_nn, D_true)
    training_ok = nn_corr ≥ RECOVERY_THRESHOLDS.nn_correlation &&
                  nn_rmse ≤ RECOVERY_THRESHOLDS.nn_rate_rmse
    if !training_ok
        return (;
            nn_correlation = nn_corr,
            nn_rate_rmse = nn_rmse,
            success = false,
            retcode = DiscoveryFailed,
            message = "training did not identify the unknown edge",
            support_f1 = 0.0,
            support_recall = 0.0,
            discovered_rate_rmse = Inf,
            data_residual = Inf,
            denominator_violations = typemax(Int),
            normalized_support_f1 = 0.0,
            normalized_support_recall = 0.0,
            extras = String[],
            discovery = nothing,
            term = term)
    end
    times = collect(range(0.0, 1.0; length = length(r)))
    truth_support = family === :hill ? hill_rate_support(order) : mm_rate_support()
    discovery = regress_unknown_rate(
        R_grid, times, D_nn;
        config = reference_protocol_discovery_config(),
        verbose = false, strict = false)
    D_norm, _ = normalize_destruction_samples(D_nn)
    discovery_norm = regress_unknown_rate(
        R_grid, times, reshape(vec(D_norm), size(D_nn));
        config = reference_protocol_discovery_config(),
        verbose = false, strict = false)
    metrics = evaluate_recovery(
        R_grid, D_nn, discovery, discovery_norm, truth_rate, truth_support,
        data_residual_fn)
    return (;
        nn_correlation = nn_corr,
        nn_rate_rmse = nn_rmse,
        success = discovery.success,
        retcode = discovery.retcode,
        message = discovery.message,
        support_f1 = metrics.support_f1,
        support_recall = metrics.support_recall,
        discovered_rate_rmse = metrics.discovered_rate_rmse,
        data_residual = metrics.data_residual,
        denominator_violations = metrics.denominator_violations,
        normalized_support_f1 = metrics.normalized_support_f1,
        normalized_support_recall = metrics.normalized_support_recall,
        extras = metrics.extras,
        extras_denominator = metrics.extras_denominator,
        discovery = discovery,
        term = term)
end

"""Fully known Michaelis–Menten production network (two states)."""
function build_mm_test_network()::BiologicalNetwork
    nodes = [NodeSpec(name = :S), NodeSpec(name = :E)]
    reactions = [
        ReactionSpec(name = :sat_prod,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            known = true, family = SATURATION,
            metadata = SaturationMetadata(
                vmax_param = :vmax, km_param = :km)),
        ReactionSpec(name = :s_decay,
            stoichiometry = Dict(1 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_s)),
        ReactionSpec(name = :e_prod,
            stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_se)),
        ReactionSpec(name = :e_decay,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_e))
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""
Two-state Hill degradation of `S` by `R`. Set `known=false` to replace the Hill
edge with a neural unknown for the UDE → discovery path.
"""
function build_hill_recovery_network(; known::Bool = true,
        hill_order::Int = 2, parent::Int = 2)::BiologicalNetwork
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R)]
    reactions = [
        ReactionSpec(name = :produce_s,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg,
            stoichiometry = Dict(1 => -1.0), regulators = [parent],
            known = known, family = HILL,
            metadata = HillMetadata(
                vmax_param = :vmax, k_param = :K,
                hill_order = hill_order)),
        ReactionSpec(name = :produce_r,
            stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r))
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""Two-state MM degradation of `S` by `R`. `known=false` compiles a neural unknown."""
function build_mm_recovery_network(; known::Bool = true)::BiologicalNetwork
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R)]
    reactions = [
        ReactionSpec(name = :produce_s,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :mm_deg,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = known, family = SATURATION,
            metadata = SaturationMetadata(
                vmax_param = :vmax, km_param = :km)),
        ReactionSpec(name = :produce_r,
            stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r))
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""Competitive-inhibition destruction of enzyme `E` by substrate `S` and inhibitor `I`."""
function build_competitive_test_network(; known::Bool = true)::BiologicalNetwork
    nodes = [NodeSpec(name = :E), NodeSpec(name = :S), NodeSpec(name = :I)]
    edges = [
        EdgeSpec(source = 2, target = 1, kind = INHIBITION,
            family = COMPETITIVE, known = known),
        EdgeSpec(source = 3, target = 1, kind = INHIBITION,
            family = COMPETITIVE, known = known)
    ]
    reactions = [
        ReactionSpec(name = :source,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_in)),
        ReactionSpec(name = :competitive,
            stoichiometry = Dict(1 => -1.0),
            regulators = [2, 3], known = known, family = COMPETITIVE,
            metadata = CompetitiveMetadata(
                vmax_param = :vmax, km_param = :km, ki_param = :ki)),
        ReactionSpec(name = :substrate_decay,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_s)),
        ReactionSpec(name = :inhibitor_decay,
            stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_i))
    ]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

"""Chain plus a distractor node for graph-local vs global library ablation."""
function build_distractor_network()::BiologicalNetwork
    nodes = [NodeSpec(name = :x), NodeSpec(name = :reg), NodeSpec(name = :z)]
    edges = [
        EdgeSpec(source = 2, target = 1, kind = INHIBITION, family = HILL)
    ]
    reactions = [
        ReactionSpec(name = :z_decay,
        stoichiometry = Dict(3 => -1.0), regulators = Int[],
        metadata = LinearDecayMetadata(rate_param = :k_z))
    ]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

"""Three dynamic states (S, R, Q) with optional unused distractor Z and one unknown Hill edge.
`parent` is the graph source of the S-edge (2 = true parent R; 3 = wrong parent Q)."""
function build_three_state_unknown_network(; known::Bool = false,
        with_distractor::Bool = true,
        parent::Int = 2)::BiologicalNetwork
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R), NodeSpec(name = :Q)]
    with_distractor && push!(nodes, NodeSpec(name = :Z))
    edges = [
        EdgeSpec(source = parent, target = 1, kind = INHIBITION,
        family = HILL, known = known)
    ]
    reactions = [
        ReactionSpec(name = :produce_s,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg,
            stoichiometry = Dict(1 => -1.0), regulators = [parent],
            known = known, family = HILL,
            metadata = HillMetadata(
                vmax_param = :vmax, k_param = :K, hill_order = 2)),
        ReactionSpec(name = :produce_r,
            stoichiometry = Dict(2 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_rq)),
        ReactionSpec(name = :decay_r,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r)),
        ReactionSpec(name = :produce_q,
            stoichiometry = Dict(3 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_qs)),
        ReactionSpec(name = :decay_q,
            stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_q))
    ]
    if with_distractor
        push!(reactions,
            ReactionSpec(
                name = :decay_z,
                stoichiometry = Dict(4 => -1.0), regulators = Int[],
                metadata = LinearDecayMetadata(rate_param = :k_z)))
    end
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

"""Negative control: graph claims Q→S while the sampled rate is still D(R)."""
build_wrong_graph_unknown_network(; known::Bool = false,
with_distractor::Bool = true) = build_three_state_unknown_network(; known = known,
    with_distractor = with_distractor,
    parent = 3)

"""Six dynamic states (S, R, Q, T, U, Z) with one unknown Hill edge on S.

`parent` is the graph source of that edge (2 = true parent R; 3 = wrong parent Q).
Known production/decay on the remaining states stay compiled. Combined F1 is
not the 1D Hill analytical threshold (the target state sits in `local_basis`).
"""
function build_six_state_unknown_network(; known::Bool = false,
        parent::Int = 2)::BiologicalNetwork
    nodes = [
        NodeSpec(name = :S), NodeSpec(name = :R), NodeSpec(name = :Q),
        NodeSpec(name = :T), NodeSpec(name = :U), NodeSpec(name = :Z)
    ]
    edges = [
        EdgeSpec(source = parent, target = 1, kind = INHIBITION,
        family = HILL, known = known)
    ]
    reactions = [
        ReactionSpec(name = :produce_s,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg,
            stoichiometry = Dict(1 => -1.0), regulators = [parent],
            known = known, family = HILL,
            metadata = HillMetadata(
                vmax_param = :vmax, k_param = :K, hill_order = 2)),
        ReactionSpec(name = :produce_r,
            stoichiometry = Dict(2 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_rq)),
        ReactionSpec(name = :decay_r,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r)),
        ReactionSpec(name = :produce_q,
            stoichiometry = Dict(3 => 1.0), regulators = [4],
            metadata = MassActionMetadata(rate_param = :k_qt)),
        ReactionSpec(name = :decay_q,
            stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_q)),
        ReactionSpec(name = :produce_t,
            stoichiometry = Dict(4 => 1.0), regulators = [5],
            metadata = MassActionMetadata(rate_param = :k_tu)),
        ReactionSpec(name = :decay_t,
            stoichiometry = Dict(4 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_t)),
        ReactionSpec(name = :produce_u,
            stoichiometry = Dict(5 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_us)),
        ReactionSpec(name = :decay_u,
            stoichiometry = Dict(5 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_u)),
        ReactionSpec(name = :decay_z,
            stoichiometry = Dict(6 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_z))
    ]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

"""Negative control: graph claims Q→S on the 6-state fixture; sampled rate is D(R)."""
build_six_state_wrong_graph_network(; known::Bool = false) = build_six_state_unknown_network(;
    known = known, parent = 3)

competitive_rate_truth(s, i; vmax, km, ki) = vmax .* s ./ (km .* (1 .+ i ./ ki) .+ s)

"""Elowitz & Leibler, Nature 403:335–338 (2000). Dimensionless synthetic fixture, not experimental CSV."""
elowitz_repressilator_parameters() = (
    basal = 1.0, unit = 1.0,
    vmax_a = 216.0, vmax_b = 216.0, vmax_c = 216.0,
    K_a = 1.0, K_b = 1.0, K_c = 1.0, γ = 1.0)

function _rhs_correlation(rhs, X, dX)
    pred = reduce(hcat, (rhs(X[:, j]) for j in axes(X, 2)))
    c = cor(vec(pred), vec(dX))
    return isnan(c) ? 0.0 : c
end

"""
    run_recovery_suite(rng=MersenneTwister(1); kwargs...)

Scientific recovery report used by CI and `benchmark/recovery_suite.jl`.
"""
function run_recovery_suite(rng::AbstractRNG = MersenneTwister(1);
        linear_adam::Int = 40,
        linear_bfgs::Int = 20,
        mm_adam::Int = 50,
        mm_bfgs::Int = 25,
        ude_adam::Int = REFERENCE_PROTOCOL.adam_iterations,
        ude_bfgs::Int = REFERENCE_PROTOCOL.bfgs_iterations,
        hill_adam::Int = 40,
        hill_bfgs::Int = 20,
        competitive_adam::Int = 40,
        competitive_bfgs::Int = 20,
        ude_noise_σ::Float64 = 0.0,
        sections = (:linear, :mm, :hill, :competitive,
            :ude_discovery, :mm_unknown, :ablation))
    report = Dict{Symbol, Any}()
    wanted = Set(sections)

    if :linear in wanted
        linear_net = build_linear_test_network()
        linear_model, linear_p0 = build_ude_model(rng, linear_net)
        linear_truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
        linear_true = pack_parameters(linear_truth, linear_p0.nn)
        u0 = [0.35, 0.25]
        tspan = (0.0, 8.0)
        times, clean, _, _ = generate_data(
            rng; network = linear_net, u0 = u0, tspan = tspan,
            n_points = 40, noise_σ = 0.0, truth_params = linear_true)
        linear_init = pack_parameters((k_ba = 1.15, k_a = 0.85, k_b = 0.75), linear_p0.nn)
        linear_fit = train_ude(
            linear_init, clean, times, u0, tspan, linear_model;
            adam_iters = linear_adam, bfgs_iters = linear_bfgs, verbose = false)
        linear_rmse, linear_rel = relative_parameter_error(linear_fit.params, linear_truth)
        report[:linear] = (; rmse = linear_rmse, rel = linear_rel,
            final_loss = linear_fit.final_loss)
    end

    if :mm in wanted
        mm_net = build_mm_test_network()
        mm_model, mm_p0 = build_ude_model(rng, mm_net)
        mm_truth = (vmax = 1.6, km = 0.45, k_s = 0.7, k_se = 0.9, k_e = 0.55)
        mm_true = pack_parameters(mm_truth, mm_p0.nn)
        mm_u0 = [0.4, 0.3]
        mm_tspan = (0.0, 8.0)
        mm_times, mm_clean, _, _ = generate_data(
            rng; network = mm_net, u0 = mm_u0, tspan = mm_tspan,
            n_points = 50, noise_σ = 0.0, truth_params = mm_true)
        mm_init = pack_parameters(
            (vmax = 1.1, km = 0.7, k_s = 1.0, k_se = 0.6, k_e = 0.8), mm_p0.nn)
        mm_fit = train_ude(
            mm_init, mm_clean, mm_times, mm_u0, mm_tspan, mm_model;
            adam_iters = mm_adam, bfgs_iters = mm_bfgs, verbose = false)
        mm_rmse, mm_rel = relative_parameter_error(mm_fit.params, mm_truth)
        report[:mm] = (; rmse = mm_rmse, rel = mm_rel, final_loss = mm_fit.final_loss)
    end

    if :hill in wanted
        hill_net = build_hill_recovery_network(; known = true, hill_order = 2)
        hill_model, hill_p0 = build_ude_model(rng, hill_net)
        hill_truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
        hill_true = pack_parameters(hill_truth, hill_p0.nn)
        hill_u0 = [0.3, 0.25]
        hill_tspan = (0.0, 8.0)
        hill_times, hill_clean, _, _ = generate_data(
            rng; network = hill_net, u0 = hill_u0, tspan = hill_tspan,
            n_points = 50, noise_σ = 0.0, truth_params = hill_true)
        hill_init = pack_parameters(
            (k_prod = 0.7, vmax = 1.2, K = 0.8, k_rs = 0.75, k_r = 0.85), hill_p0.nn)
        hill_fit = train_ude(
            hill_init, hill_clean, hill_times, hill_u0, hill_tspan, hill_model;
            adam_iters = hill_adam, bfgs_iters = hill_bfgs, verbose = false)
        hill_rmse, hill_rel = relative_parameter_error(hill_fit.params, hill_truth)
        report[:hill] = (; rmse = hill_rmse, rel = hill_rel,
            final_loss = hill_fit.final_loss)
    end

    if :competitive in wanted
        comp_net = build_competitive_test_network()
        comp_model, comp_p0 = build_ude_model(rng, comp_net)
        comp_truth = (k_in = 0.9, vmax = 1.5, km = 0.4, ki = 0.6, k_s = 0.8, k_i = 0.5)
        comp_true = pack_parameters(comp_truth, comp_p0.nn)
        comp_u0 = [0.25, 0.45, 0.2]
        comp_tspan = (0.0, 8.0)
        comp_times, comp_clean, _, _ = generate_data(
            rng; network = comp_net, u0 = comp_u0, tspan = comp_tspan,
            n_points = 55, noise_σ = 0.0, truth_params = comp_true)
        comp_init = pack_parameters(
            (k_in = 0.65, vmax = 1.1, km = 0.65, ki = 0.9, k_s = 1.05, k_i = 0.75),
            comp_p0.nn)
        comp_fit = train_ude(
            comp_init, comp_clean, comp_times, comp_u0, comp_tspan, comp_model;
            adam_iters = competitive_adam, bfgs_iters = competitive_bfgs,
            verbose = false)
        comp_rmse, comp_rel = relative_parameter_error(comp_fit.params, comp_truth)
        report[:competitive] = (; rmse = comp_rmse, rel = comp_rel,
            final_loss = comp_fit.final_loss)
    end

    if :ude_discovery in wanted
        truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
        ude_net = admit_recovery_suite_network(:ude_discovery)
        # Consume the same RNG stream as known-kinetics fixtures so UDE init stays stable.
        consume_shared_suite_rng!(rng, truth_net)
        hill_truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
        ude_model, ude_p0 = build_ude_model(rng, ude_net)
        ude_fit, ude_set = _train_unknown_edge(
            rng, ude_model, ude_p0, truth_net, hill_truth;
            adam = ude_adam, bfgs = ude_bfgs, noise_σ = ude_noise_σ,
            tspan = REFERENCE_PROTOCOL.tspan,
            n_points = REFERENCE_PROTOCOL.n_points)
        term = only_unknown_destruction(ude_model)
        ref_exp = first(ude_set.experiments)
        split = reference_protocol_experiment_split(ude_set)
        truth_rate = r -> hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2)
        evaled = _evaluate_unknown_rate_recovery(
            ude_model, ude_fit.params, term, truth_rate;
            order = 2, family = :hill, noise_σ = ude_noise_σ,
            r_range = _regulator_grid(split.train, term),
            data_residual_fn = d_hat -> hybrid_data_residual(
                ude_model, ude_fit.params, term, d_hat,
                ref_exp.u0, (first(ref_exp.times), last(ref_exp.times)),
                ref_exp.times, ref_exp.observations))
        ident_ude = report_production_destruction_tradeoff(
            ude_model, ude_fit.params, ref_exp.observations, ref_exp.times,
            ref_exp.u0, (first(ref_exp.times), last(ref_exp.times));
            term = term, verbose = false)
        if evaled.discovery === nothing
            holdout = nothing
        else
            holdout = evaluate_holdout(
                split, evaled, ude_model, ude_fit.params, term, truth_rate)
        end
        report[:ude_discovery] = report_recovery(
            evaled, ident_ude;
            model = ude_model, params = ude_fit.params, experiments = ude_set,
            split = split, holdout = holdout)
    end

    if :mm_unknown in wanted
        truth_net = build_mm_recovery_network(; known = true)
        ude_net = admit_recovery_suite_network(:mm_unknown)
        consume_shared_suite_rng!(rng, truth_net)
        mm_truth = (k_prod = 0.9, vmax = 1.6, km = 0.45, k_rs = 1.0, k_r = 0.6)
        ude_model, ude_p0 = build_ude_model(rng, ude_net)
        ude_fit, ude_set = _train_unknown_edge(
            rng, ude_model, ude_p0, truth_net, mm_truth;
            adam = ude_adam, bfgs = ude_bfgs, noise_σ = ude_noise_σ,
            tspan = REFERENCE_PROTOCOL.tspan,
            n_points = REFERENCE_PROTOCOL.n_points)
        term = only_unknown_destruction(ude_model)
        ref_exp = first(ude_set.experiments)
        split = reference_protocol_experiment_split(ude_set)
        truth_rate = r -> mm_rate_truth(r; vmax = 1.6, km = 0.45)
        evaled = _evaluate_unknown_rate_recovery(
            ude_model, ude_fit.params, term, truth_rate;
            order = 1, family = :mm, noise_σ = ude_noise_σ,
            r_range = _regulator_grid(split.train, term),
            data_residual_fn = d_hat -> hybrid_data_residual(
                ude_model, ude_fit.params, term, d_hat,
                ref_exp.u0, (first(ref_exp.times), last(ref_exp.times)),
                ref_exp.times, ref_exp.observations))
        ident_mm = report_production_destruction_tradeoff(
            ude_model, ude_fit.params, ref_exp.observations, ref_exp.times,
            ref_exp.u0, (first(ref_exp.times), last(ref_exp.times));
            term = term, verbose = false)
        if evaled.discovery === nothing
            holdout = nothing
        else
            holdout = evaluate_holdout(
                split, evaled, ude_model, ude_fit.params, term, truth_rate)
        end
        report[:mm_unknown] = report_recovery(
            evaled, ident_mm;
            model = ude_model, params = ude_fit.params, experiments = ude_set,
            split = split, holdout = holdout)
    end

    if :ablation in wanted
        r = collect(range(0.1, 2.0; length = 180))
        rng_ab = MersenneTwister(104)
        vmax, k = 1.7, 0.6
        D = hill_rate_truth(r; vmax = vmax, K = k, n = 2)
        amp = max(maximum(abs, D), eps(Float64))
        D_noisy = D .+ 0.005 .* amp .* randn(rng_ab, length(r))
        # Near-alias of the true Hill monomial on a distractor channel.
        z = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(rng_ab, length(r))
        X_ab = permutedims(hcat(r, z))
        dX_ab = vcat(reshape(D_noisy, 1, :), reshape(-0.5 .* z, 1, :))
        X_ab, dX_ab = _permute_rate_samples(X_ab, dX_ab, 104)
        times_ab = collect(range(0.0, 1.0; length = length(r)))
        net_ab = build_rate_ablation_network()
        truth = hill_rate_support(2; variable = 1)
        local_time = @elapsed local_disc = discover_equations(
            X_ab, times_ab, net_ab; derivatives = dX_ab, targets = 1,
            config = rate_discovery_config(scope = :graph, bootstrap = 8, seed = 4),
            verbose = false, strict = false)
        global_time = @elapsed global_disc = discover_equations(
            X_ab, times_ab, net_ab; derivatives = dX_ab, targets = 1,
            config = rate_discovery_config(scope = :global, bootstrap = 8, seed = 4),
            verbose = false, strict = false)
        local_idx = local_disc.success ?
                    findfirst(c -> c.target == 1, local_disc.candidates) : nothing
        local_cand = local_idx === nothing ? nothing : local_disc.candidates[local_idx]
        global_idx = global_disc.success ?
                     findfirst(c -> c.target == 1, global_disc.candidates) : nothing
        global_cand = global_idx === nothing ? nothing : global_disc.candidates[global_idx]
        local_f1 = local_cand === nothing ? 0.0 :
                   support_f1(local_cand, truth.numerator, truth.denominator).combined.f1
        global_f1 = global_cand === nothing ? 0.0 :
                    support_f1(global_cand, truth.numerator, truth.denominator).combined.f1
        local_fp = local_cand !== nothing &&
                   support_uses_variable(local_cand; variable = 2)
        global_fp = global_cand !== nothing &&
                    support_uses_variable(global_cand; variable = 2)
        local_den = local_cand === nothing ? typemax(Int) :
                    denominator_violation_count(local_cand, X_ab)
        global_den = global_cand === nothing ? typemax(Int) :
                     denominator_violation_count(global_cand, X_ab)
        local_spec = local_basis(
            net_ab, 1; degree = 2, include_interactions = false, scope = :graph)
        global_spec = local_basis(
            net_ab, 1; degree = 2, include_interactions = false, scope = :global)
        local_rate_rmse = local_cand === nothing ? Inf :
                          rate_rel_rmse(
            [equation_to_function(local_cand)(X_ab[:, j])
             for j in axes(X_ab, 2)],
            vec(dX_ab[1, :]))
        global_rate_rmse = global_cand === nothing ? Inf :
                           rate_rel_rmse(
            [equation_to_function(global_cand)(X_ab[:, j])
             for j in axes(X_ab, 2)],
            vec(dX_ab[1, :]))
        report[:ablation] = (;
            local_terms = candidate_count(local_spec),
            global_terms = candidate_count(global_spec),
            local_variables = copy(local_spec.variables),
            global_variables = copy(global_spec.variables),
            local_success = local_disc.success,
            global_success = global_disc.success,
            local_f1 = local_f1,
            global_f1 = global_f1,
            local_false_parent = local_fp,
            global_false_parent = global_fp,
            local_denominator_violations = local_den,
            global_denominator_violations = global_den,
            local_rate_rmse = local_rate_rmse,
            global_rate_rmse = global_rate_rmse,
            local_time = local_time,
            global_time = global_time)
    end

    if :three_state in wanted
        r = collect(range(0.1, 2.0; length = 180))
        rng_3 = MersenneTwister(204)
        D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
        amp = max(maximum(abs, D), eps(Float64))
        D_noisy = D .+ 0.005 .* amp .* randn(rng_3, length(r))
        s = fill(0.4, length(r))
        q = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(rng_3, length(r))
        z = r .+ 0.10 .* (maximum(r) - minimum(r)) .* randn(rng_3, length(r))
        X3 = permutedims(hcat(s, r, q, z))
        dX3 = vcat(reshape(D_noisy, 1, :), zeros(3, length(r)))
        X3, dX3 = _permute_rate_samples(X3, dX3, 204)
        times3 = collect(range(0.0, 1.0; length = length(r)))
        net3 = build_three_state_unknown_network()
        truth3 = hill_rate_support(2; variable = 2)
        local3 = discover_equations(
            X3, times3, net3; derivatives = dX3, targets = 1,
            config = rate_discovery_config(scope = :graph, bootstrap = 0, seed = 5),
            verbose = false, strict = false)
        global3 = discover_equations(
            X3, times3, net3; derivatives = dX3, targets = 1,
            config = rate_discovery_config(scope = :global, bootstrap = 0, seed = 5),
            verbose = false, strict = false)
        lc = local3.success ? local3.candidates[1] : nothing
        gc = global3.success ? global3.candidates[1] : nothing
        parents = candidate_parents(net3, 1)
        report[:three_state] = (;
            graph_parents = parents,
            local_success = local3.success,
            global_success = global3.success,
            local_f1 = lc === nothing ? 0.0 :
                       support_f1(lc, truth3.numerator, truth3.denominator).combined.f1,
            global_f1 = gc === nothing ? 0.0 :
                        support_f1(gc, truth3.numerator, truth3.denominator).combined.f1,
            local_has_true_parent = local_has_true_parent_check(lc; variable = 2),
            local_false_parent = local_has_false_parent_check(lc; variables = (3, 4)),
            global_false_parent = local_has_false_parent_check(gc; variables = (3, 4)))
    end

    if :wrong_graph in wanted
        r = collect(range(0.1, 2.0; length = 180))
        rng_w = MersenneTwister(214)
        D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
        amp = max(maximum(abs, D), eps(Float64))
        D_noisy = D .+ 0.005 .* amp .* randn(rng_w, length(r))
        s = fill(0.4, length(r))
        q = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(rng_w, length(r))
        z = r .+ 0.10 .* (maximum(r) - minimum(r)) .* randn(rng_w, length(r))
        Xw = permutedims(hcat(s, r, q, z))
        dXw = vcat(reshape(D_noisy, 1, :), zeros(3, length(r)))
        Xw, dXw = _permute_rate_samples(Xw, dXw, 214)
        times_w = collect(range(0.0, 1.0; length = length(r)))
        net_w = build_wrong_graph_unknown_network()
        truth_w = hill_rate_support(2; variable = 2)
        local_w = discover_equations(
            Xw, times_w, net_w; derivatives = dXw, targets = 1,
            config = rate_discovery_config(scope = :graph, bootstrap = 0, seed = 8),
            verbose = false, strict = false)
        lcw = local_w.success ? local_w.candidates[1] : nothing
        parents_w = candidate_parents(net_w, 1)
        report[:wrong_graph] = (;
            graph_parents = parents_w,
            local_success = local_w.success,
            local_f1 = lcw === nothing ? 0.0 :
                       support_f1(lcw, truth_w.numerator, truth_w.denominator).combined.f1,
            local_has_true_parent = local_has_true_parent_check(lcw; variable = 2),
            local_false_parent = local_has_false_parent_check(lcw; variables = (3, 4)))
    end

    if :identifiability in wanted
        hill_net = build_hill_recovery_network(; known = true, hill_order = 2)
        hill_model, hill_p0 = build_ude_model(rng, hill_net)
        hill_truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
        hill_p = pack_parameters(hill_truth, hill_p0.nn)
        u0 = [0.3, 0.25]
        tspan = (0.0, 8.0)
        times, data, _, _ = generate_data(
            rng; network = hill_net, u0 = u0, tspan = tspan,
            n_points = 40, noise_σ = 0.0, truth_params = hill_p)
        trade = production_destruction_tradeoff(
            hill_model, hill_p, data, times, u0, tspan)
        report[:identifiability] = trade
    end

    if :ident_interventions in wanted
        r = collect(range(0.1, 2.0; length = 120))
        D_h = hill_rate_truth(r; vmax = 1.8, K = 0.6, n = 2)
        D_n, _ = normalize_destruction_samples(D_h)
        result_n = regress_unknown_rate(
            reshape(r, 1, :), collect(range(0.0, 1.0; length = length(r))),
            reshape(vec(D_n), 1, :);
            config = rate_discovery_config(bootstrap = 0, seed = 9),
            verbose = false, strict = false)
        truth_n = hill_rate_support(2)
        metrics_n = result_n.success ?
                    support_f1(
            result_n.candidates[1], truth_n.numerator, truth_n.denominator) :
                    nothing
        ude_net = admit_recovery_suite_network(:ident_interventions)
        ude_model, ude_p0 = build_ude_model(rng, ude_net)
        term = only_unknown_destruction(ude_model)
        p_nom = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), ude_p0.nn)
        u0 = [0.3, 0.25]
        tspan = (0.0, 2.0)
        times = collect(range(first(tspan), last(tspan); length = 20))
        data = predict_ude(p_nom, u0, tspan, times, ude_model)
        trade_nom = production_destruction_tradeoff(
            ude_model, p_nom, data, times, u0, tspan; term = term)
        k_prod_raw = p_nom.phys.k_prod
        freeze_fit = train_ude(
            p_nom, data, times, u0, tspan, ude_model;
            config = TrainingConfig(
                adam_iterations = 8, bfgs_iterations = 0, log_every = 10^6,
                frozen_phys = [:k_prod]),
            verbose = false)
        trade_freeze = production_destruction_tradeoff(
            ude_model, freeze_fit.params, data, times, u0, tspan; term = term)
        p_hi = pack_parameters((k_prod = 1.8, k_rs = 1.0, k_r = 0.6), ude_p0.nn)
        data_hi = predict_ude(p_hi, u0, tspan, times, ude_model)
        trade_hi = production_destruction_tradeoff(
            ude_model, p_hi, data_hi, times, u0, tspan; term = term)
        report[:ident_interventions] = (;
            normalized_analytical_f1 = metrics_n === nothing ? 0.0 : metrics_n.combined.f1,
            normalized_analytical_recall = metrics_n === nothing ? 0.0 :
                                           metrics_n.combined.recall,
            nominal_collinearity = trade_nom.collinearity,
            nominal_unidentifiable = trade_nom.unidentifiable_edge,
            frozen_k_prod_unchanged = freeze_fit.params.phys.k_prod ≈ k_prod_raw,
            freeze_collinearity = trade_freeze.collinearity,
            freeze_unidentifiable = trade_freeze.unidentifiable_edge,
            perturbation_collinearity = trade_hi.collinearity,
            perturbation_unidentifiable = trade_hi.unidentifiable_edge,
            tradeoff_broken = false)
    end

    if :partial_obs in wanted
        r = collect(range(0.1, 2.0; length = 180))
        D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
        keep = trues(length(r))
        keep[2:2:end] .= false
        result = regress_unknown_rate(
            reshape(r[keep], 1, :), collect(range(0.0, 1.0; length = count(keep))),
            reshape(D[keep], 1, :);
            config = rate_discovery_config(bootstrap = 0, seed = 6),
            verbose = false, strict = false)
        truth = hill_rate_support(2)
        metrics = result.success ?
                  support_f1(result.candidates[1], truth.numerator, truth.denominator) :
                  nothing
        linear_net = build_linear_test_network()
        linear_model, linear_p0 = build_ude_model(rng, linear_net)
        linear_true = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), linear_p0.nn)
        u0 = [0.35, 0.25]
        tspan = (0.0, 6.0)
        times, clean, _, _ = generate_data(
            rng; network = linear_net, u0 = u0, tspan = tspan,
            n_points = 30, noise_σ = 0.0, truth_params = linear_true)
        mask = trues(size(clean))
        mask[1, 10:end] .= false
        exp = Experiment(:masked, times, clean, u0; mask = mask)
        masked_set = ExperimentSet([exp], [:A, :B])
        init = pack_parameters((k_ba = 1.1, k_a = 0.9, k_b = 0.7), linear_p0.nn)
        fit = train_experiments(
            init, masked_set, linear_model;
            config = TrainingConfig(adam_iterations = 25, bfgs_iterations = 0,
                log_every = 10^6),
            verbose = false)
        hill_net = build_hill_recovery_network(; known = true, hill_order = 2)
        ude_net = admit_recovery_suite_network(:partial_obs)
        _, hill_p0 = build_ude_model(rng, hill_net)
        ude_model, ude_p0 = build_ude_model(rng, ude_net)
        hill_truth = (k_prod = 0.9, vmax = 1.7, K = 0.6, k_rs = 1.0, k_r = 0.6)
        hill_p = pack_parameters(hill_truth, hill_p0.nn)
        u0_h = [0.3, 0.25]
        tspan_h = (0.0, 8.0)
        times_h, data_h, _, _ = generate_data(
            rng; network = hill_net, u0 = u0_h, tspan = tspan_h,
            n_points = 40, noise_σ = 0.0, truth_params = hill_p)
        p_hybrid = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), ude_p0.nn)
        term = only_unknown_destruction(ude_model)
        closed_residual = Inf
        masked_residual = Inf
        obs_mask = trues(size(data_h))
        obs_mask[1, 2:2:end] .= false
        if result.success
            d_hat = equation_to_function(result.candidates[1])
            closed_residual = hybrid_data_residual(
                ude_model, p_hybrid, term, d_hat,
                u0_h, tspan_h, times_h, data_h)
            masked_residual = hybrid_data_residual(
                ude_model, p_hybrid, term, d_hat,
                u0_h, tspan_h, times_h, data_h; mask = obs_mask)
        end
        report[:partial_obs] = (;
            subsample_success = result.success,
            subsample_f1 = metrics === nothing ? 0.0 : metrics.combined.f1,
            subsample_recall = metrics === nothing ? 0.0 : metrics.combined.recall,
            mask_used = !all(exp.mask),
            masked_train_loss = fit.final_loss,
            masked_train_finite = isfinite(fit.final_loss),
            closed_loop_residual = closed_residual,
            closed_loop_masked_residual = masked_residual,
            closed_loop_vs_data = closed_residual ≤ RECOVERY_THRESHOLDS.data_residual,
            ude_mask_train_claimed = false)
    end

    if :competitive_unknown in wanted
        rng_c = MersenneTwister(304)
        ns, ni = 24, 24
        svals = collect(range(0.15, 1.8; length = ns))
        ivals = collect(range(0.15, 1.8; length = ni))
        S = repeat(svals, inner = ni)
        I = repeat(ivals, outer = ns)
        D = competitive_rate_truth(S, I; vmax = 1.5, km = 0.4, ki = 0.6)
        z = D .+ 0.15 .* maximum(abs, D) .* randn(rng_c, length(D))
        R = permutedims(hcat(S, I, z))
        dX = vcat(reshape(D, 1, :), zeros(2, length(D)))
        R, dX = _permute_rate_samples(R, dX, 304)
        times_c = collect(range(0.0, 1.0; length = length(D)))
        result = regress_unknown_rate(
            R[1:2, :], times_c, dX[1:1, :];
            config = rate_discovery_config(bootstrap = 0, seed = 7),
            verbose = false, strict = false)
        net_ab = BiologicalNetwork(
            [NodeSpec(name = :s), NodeSpec(name = :i), NodeSpec(name = :z)],
            [EdgeSpec(source = 2, target = 1, kind = INHIBITION, family = HILL)])
        local_c = discover_equations(
            R, times_c, net_ab; derivatives = dX, targets = 1,
            config = rate_discovery_config(scope = :graph, bootstrap = 0, seed = 7),
            verbose = false, strict = false)
        global_c = discover_equations(
            R, times_c, net_ab; derivatives = dX, targets = 1,
            config = rate_discovery_config(scope = :global, bootstrap = 0, seed = 7),
            verbose = false, strict = false)
        cand = result.success ? result.candidates[1] : nothing
        lc = local_c.success ? local_c.candidates[1] : nothing
        gc = global_c.success ? global_c.candidates[1] : nothing
        compiled = compile_mechanism(build_competitive_test_network(; known = false))
        nn_terms = [t for t in compiled.destruction_terms if t isa NeuralDestructionTerm]
        report[:competitive_unknown] = (;
            compiled_regulators = isempty(nn_terms) ? Int[] : first(nn_terms).regulators,
            two_parent_success = result.success,
            has_substrate = cand !== nothing && support_uses_variable(cand; variable = 1),
            has_inhibitor = cand !== nothing && support_uses_variable(cand; variable = 2),
            local_false_parent = lc !== nothing && support_uses_variable(lc; variable = 3),
            global_false_parent = gc !== nothing && support_uses_variable(gc; variable = 3),
            canonical_f1_claimed = false)
    end

    if :six_state in wanted
        r = collect(range(0.1, 2.0; length = 180))
        rng_6 = MersenneTwister(224)
        D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
        amp = max(maximum(abs, D), eps(Float64))
        D_noisy = D .+ 0.005 .* amp .* randn(rng_6, length(r))
        s = fill(0.4, length(r))
        q = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(rng_6, length(r))
        tvals = fill(0.35, length(r))
        uvals = fill(0.25, length(r))
        z = r .+ 0.10 .* (maximum(r) - minimum(r)) .* randn(rng_6, length(r))
        X6 = permutedims(hcat(s, r, q, tvals, uvals, z))
        dX6 = vcat(reshape(D_noisy, 1, :), zeros(5, length(r)))
        X6, dX6 = _permute_rate_samples(X6, dX6, 224)
        times6 = collect(range(0.0, 1.0; length = length(r)))
        net6 = build_six_state_unknown_network()
        truth6 = hill_rate_support(2; variable = 2)
        local_spec6 = local_basis(net6, 1; degree = 2, include_interactions = false,
            scope = :graph)
        global_spec6 = local_basis(net6, 1; degree = 2, include_interactions = false,
            scope = :global)
        local6 = discover_equations(
            X6, times6, net6; derivatives = dX6, targets = 1,
            config = rate_discovery_config(scope = :graph, bootstrap = 0, seed = 10),
            verbose = false, strict = false)
        global6 = discover_equations(
            X6, times6, net6; derivatives = dX6, targets = 1,
            config = rate_discovery_config(scope = :global, bootstrap = 0, seed = 10),
            verbose = false, strict = false)
        lc6 = local6.success ? local6.candidates[1] : nothing
        gc6 = global6.success ? global6.candidates[1] : nothing
        parents6 = candidate_parents(net6, 1)
        report[:six_state] = (;
            nstates = length(net6.nodes),
            graph_parents = parents6,
            local_variables = copy(local_spec6.variables),
            global_variables = copy(global_spec6.variables),
            local_success = local6.success,
            global_success = global6.success,
            local_f1 = lc6 === nothing ? 0.0 :
                       support_f1(lc6, truth6.numerator, truth6.denominator).combined.f1,
            global_f1 = gc6 === nothing ? 0.0 :
                        support_f1(gc6, truth6.numerator, truth6.denominator).combined.f1,
            local_has_true_parent = local_has_true_parent_check(lc6; variable = 2),
            local_false_parent = local_has_false_parent_check(
                lc6; variables = (3, 4, 5, 6)),
            global_false_parent = local_has_false_parent_check(
                gc6; variables = (3, 4, 5, 6)),
            distractor_in_local = 6 ∈ local_spec6.variables,
            distractor_in_global = 6 ∈ global_spec6.variables,
            Z_in_local_library = 6 ∈ local_spec6.variables,
            Z_in_global_library = 6 ∈ global_spec6.variables)
    end

    if :six_state_wrong_graph in wanted
        r = collect(range(0.1, 2.0; length = 180))
        rng_6w = MersenneTwister(234)
        D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
        amp = max(maximum(abs, D), eps(Float64))
        D_noisy = D .+ 0.005 .* amp .* randn(rng_6w, length(r))
        s = fill(0.4, length(r))
        q = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(rng_6w, length(r))
        tvals = fill(0.35, length(r))
        uvals = fill(0.25, length(r))
        z = r .+ 0.10 .* (maximum(r) - minimum(r)) .* randn(rng_6w, length(r))
        X6w = permutedims(hcat(s, r, q, tvals, uvals, z))
        dX6w = vcat(reshape(D_noisy, 1, :), zeros(5, length(r)))
        X6w, dX6w = _permute_rate_samples(X6w, dX6w, 234)
        times_6w = collect(range(0.0, 1.0; length = length(r)))
        net_6w = build_six_state_wrong_graph_network()
        truth_6w = hill_rate_support(2; variable = 2)
        local_6w = discover_equations(
            X6w, times_6w, net_6w; derivatives = dX6w, targets = 1,
            config = rate_discovery_config(scope = :graph, bootstrap = 0, seed = 11),
            verbose = false, strict = false)
        lc6w = local_6w.success ? local_6w.candidates[1] : nothing
        parents_6w = candidate_parents(net_6w, 1)
        report[:six_state_wrong_graph] = (;
            nstates = length(net_6w.nodes),
            graph_parents = parents_6w,
            local_success = local_6w.success,
            local_f1 = lc6w === nothing ? 0.0 :
                       support_f1(
                lc6w, truth_6w.numerator, truth_6w.denominator).combined.f1,
            local_has_true_parent = local_has_true_parent_check(lc6w; variable = 2),
            local_false_parent = local_has_false_parent_check(
                lc6w; variables = (3, 4, 5, 6)))
    end

    if :literature in wanted
        net = build_repressilator_network(; hill_order = 2)
        params = elowitz_repressilator_parameters()
        model, p0 = build_ude_model(rng, net)
        truth = pack_parameters(params, p0.nn)
        u0 = [0.2, 0.1, 0.3]
        tspan = (0.0, 5.0)
        times, clean, _, _ = generate_data(
            rng; network = net, u0 = u0, tspan = tspan,
            n_points = 40, noise_σ = 0.0, truth_params = truth)
        report[:literature] = (;
            source = "Elowitz & Leibler, Nature 403:335–338 (2000)",
            experimental_csv = false,
            reference_protocol_protocol = false,
            licensed_experimental_series = false,
            nstates = size(clean, 1),
            finite_trajectory = all(isfinite, clean),
            nonnegative = all(≥(0), clean))
    end
    return report
end
