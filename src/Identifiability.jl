"""
    IdentifiabilityReport

**Practical** Fisher-information summary for compiled physical parameters at a
fitted trajectory. This is not structural identifiability
(see StructuralIdentifiability.jl). Rank and asymptotic Fisher intervals are local,
finite-difference Gauss–Newton estimates.
"""
struct IdentifiabilityReport
    parameter_names::Vector{Symbol}
    fisher_information::Matrix{Float64}
    condition_number::Float64
    identifiable::BitVector
    correlation_matrix::Matrix{Float64}
    residual_variance::Float64
end

function _perturb_phys_parameter(p, name::Symbol, delta::Real)
    raw = getproperty(p.phys, name)
    pairs = [(sym, sym == name ? raw + delta : getproperty(p.phys, sym))
             for sym in propertynames(p.phys)]
    return ComponentVector(phys = NamedTuple(pairs), nn = p.nn)
end

"""
    trajectory_jacobian(model, p, u0, tspan, times; rel_step=1e-4)

Finite-difference Jacobian of the predicted trajectory with respect to
physical kinetic parameters.
"""
function trajectory_jacobian(model::UDEModel, p, u0, tspan, times;
        rel_step::Real = 1e-4,
        solver_config::SolverConfig = SolverConfig())
    schema = parameter_schema(model)
    names = schema.phys_names
    base = predict_ude(
        p, u0, tspan, times, model; solver_config = solver_config)
    n_obs = length(times) * size(base, 1)
    jacobian = Matrix{Float64}(undef, n_obs, length(names))
    for (col, name) in pairs(names)
        raw = getproperty(p.phys, name)
        delta = max(abs(raw), oneunit(raw)) * rel_step
        plus = _perturb_phys_parameter(p, name, delta)
        minus = _perturb_phys_parameter(p, name, -delta)
        forward = predict_ude(
            plus, u0, tspan, times, model; solver_config = solver_config)
        backward = predict_ude(
            minus, u0, tspan, times, model; solver_config = solver_config)
        jacobian[:, col] = vec((forward .- backward) ./ (2delta))
    end
    return jacobian, names
end

"""
    fisher_information_matrix(model, p, data, t_data, u0, tspan; kwargs...)

Gauss–Newton Fisher information matrix `J'J / σ²` for physical parameters,
over the observed entries of `data` (`mask`, all by default).
"""
function fisher_information_matrix(model::UDEModel, p, data, t_data, u0, tspan;
        mask = trues(size(data)),
        residual_variance = nothing,
        kwargs...)
    jacobian, names = trajectory_jacobian(
        model, p, u0, tspan, t_data; kwargs...)
    residual = ifelse.(mask, data .- predict_ude(
            p, u0, tspan, t_data, model; kwargs...),
        zero(eltype(data)))
    σ² = residual_variance === nothing ?
         max(eps(), sum(abs2, residual) / max(1, count(mask))) :
         float(residual_variance)
    # Only observed entries carry information: drop the Jacobian rows of
    # unobserved (masked) entries before forming J'J.
    observed = vec(mask)
    J = all(observed) ? jacobian : jacobian[observed, :]
    information = (J' * J) ./ σ²
    return information, names, σ²
end

"""
    assess_identifiability(model, p, data, t_data, u0, tspan; threshold=1e-8)

Rank-based **practical** identifiability from the Fisher information at this
fit. Neural parameters are excluded. Not a substitute for structural analysis.
"""
function assess_identifiability(model::UDEModel, p, data, t_data, u0, tspan;
        threshold::Real = 1e-8, kwargs...)
    information, names, σ² = fisher_information_matrix(
        model, p, data, t_data, u0, tspan; kwargs...)
    eigenvalues = eigvals(Symmetric(information))
    positive = eigenvalues[eigenvalues .> threshold * maximum(eigenvalues)]
    condition = isempty(positive) ? Inf :
                maximum(positive) / max(minimum(positive), threshold)
    identifiable = begin
        covariance = pinv(information)
        variances = diag(covariance)
        .!isinf.(variances) .& (variances .< 1e8)
    end
    correlation = _correlation_from_information(information)
    return IdentifiabilityReport(
        names, information, condition, BitVector(identifiable),
        correlation, σ²)
end

function _correlation_from_information(information::AbstractMatrix)
    n = size(information, 1)
    correlation = Matrix{Float64}(I, n, n)
    for i in 1:n, j in (i + 1):n
        denom = sqrt(information[i, i] * information[j, j])
        value = denom == 0 ? zero(denom) : information[i, j] / denom
        correlation[i, j] = value
        correlation[j, i] = value
    end
    return correlation
end

function _z_score(level::Real)
    level ≈ 0.90 && return 1.6448536269514722
    level ≈ 0.95 && return 1.959963984540054
    level ≈ 0.99 && return 2.5758293035489004
    throw(ArgumentError(
        "unsupported nominal coverage $level; use 0.90, 0.95, or 0.99"))
end

"""
    parameter_credible_intervals(report, level=0.95)

Asymptotic Fisher intervals from the inverse Fisher information at the
requested nominal coverage.
"""
function parameter_credible_intervals(report::IdentifiabilityReport,
        estimate;
        level::Real = 0.95)
    z = _z_score(level)
    information = report.fisher_information
    variances = diag(pinv(information))
    intervals = Dict{Symbol, Tuple{Float64, Float64}}()
    for (name, variance) in zip(report.parameter_names, variances)
        center = positive_parameter(getproperty(estimate.phys, name))
        half = z * sqrt(max(variance, zero(variance)))
        intervals[name] = (max(0.0, center - half), center + half)
    end
    return intervals
end

"""
    estimate_parameter_uncertainty(model, params, data, t_data, u0, tspan; level=0.95)

Asymptotic uncertainty for physical parameters from the Fisher information matrix.
"""
function estimate_parameter_uncertainty(model::UDEModel, params, data, t_data,
        u0, tspan; level::Real = 0.95, kwargs...)
    report = assess_identifiability(
        model, params, data, t_data, u0, tspan; kwargs...)
    intervals = parameter_credible_intervals(report, params; level = level)
    names = report.parameter_names
    estimates = Float64[positive_parameter(getproperty(params.phys, name))
                        for name in names]
    lower = Float64[intervals[name][1] for name in names]
    upper = Float64[intervals[name][2] for name in names]
    return ParameterUncertainty(names, estimates, lower, upper, float(level), :fisher)
end

function _simulate_trajectory(rhs, u0, tspan, times)
    prob = SciMLBase.ODEProblem(rhs, u0, tspan)
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return nothing
    return Array(sol)
end

"""
    production_destruction_tradeoff(model, p, data, t_data, u0, tspan; kwargs...)

Practical collinearity between a production parameter (default `k_prod`) and a
multiplicative scale on unknown neural destruction `D(z)`. Reports Fisher
condition number and trajectory-Jacobian cosine. This is not structural
identifiability. `mask` marks the observed entries of `data` (all by
default); the Fisher information and the trajectory-Jacobian cosine are
computed over the observed entries only, so an unobserved state (a `false`
row) contributes nothing to either.
"""
function production_destruction_tradeoff(
        model::UDEModel, p, data, t_data, u0, tspan;
        production_param::Symbol = :k_prod,
        term = nothing,
        rel_step::Real = 1e-3,
        collinearity_threshold::Real = 0.95,
        condition_threshold::Real = 1e6,
        mask = trues(size(data)))
    fisher = assess_identifiability(model, p, data, t_data, u0, tspan; mask = mask)
    names = fisher.parameter_names
    prod_idx = findfirst(==(production_param), names)
    production_correlation = if prod_idx === nothing || length(names) < 2
        NaN
    else
        others = [fisher.correlation_matrix[prod_idx, j]
                  for j in eachindex(names) if j != prod_idx]
        maximum(abs, others; init = 0.0)
    end
    nn_terms = neural_destruction_terms(model)
    chosen = term === nothing ?
             (isempty(nn_terms) ? nothing : first(nn_terms)) : term
    collinearity = NaN
    if chosen isa NeuralDestructionTerm
        δ = rel_step
        function scaled_rhs(sign)
            return function (u, _, t)
                du = ude_rhs(u, p, t, model)
                D = _destruction_contribution(
                    chosen, chosen.target, u, p, model.nn, model.st)
                du[chosen.target] -= sign * δ * D * u[chosen.target]
                return du
            end
        end
        plus = _simulate_trajectory(scaled_rhs(1.0), u0, tspan, t_data)
        minus = _simulate_trajectory(scaled_rhs(-1.0), u0, tspan, t_data)
        j_d = plus === nothing || minus === nothing ?
              nothing : vec((plus .- minus) ./ (2δ))
        j_p = nothing
        if prod_idx !== nothing && j_d !== nothing
            jacobian, _ = trajectory_jacobian(
                model, p, u0, tspan, t_data; rel_step = rel_step)
            j_p = jacobian[:, prod_idx]
        end
        if j_d !== nothing && j_p !== nothing
            # The cosine is taken over the observed entries only.
            observed = vec(mask)
            jp = all(observed) ? j_p : j_p[observed]
            jd = all(observed) ? j_d : j_d[observed]
            denom = norm(jp) * norm(jd)
            collinearity = denom == 0 ? 0.0 : abs(dot(jp, jd)) / denom
        end
    end
    unidentifiable_edge = unidentifiable_edge_from_fisher(;
        condition_number = fisher.condition_number,
        collinearity,
        condition_threshold,
        collinearity_threshold)
    return (;
        production_param,
        condition_number = fisher.condition_number,
        production_correlation,
        collinearity,
        unidentifiable_edge,
        fisher)
end

"""
    cross_term_collinearity(model, p, u0, tspan, times; mask, rel_step = 1e-3)

Pairwise diagnostic for models with several unknown destruction terms. For
each term the output of its network is scaled by `1 ± rel_step`, the model is
simulated on `times`, and the central finite difference of the trajectory is
the term's sensitivity vector over the observed entries (`mask`, all by
default). For each pair of terms the value is the absolute cosine between the
two sensitivity vectors. A value near one means that, in these data, scaling
one term looks like scaling the other, so the two scales are not separately
determined at this fit. It is the same construction as the scale collinearity
of `production_destruction_tradeoff`, applied between terms rather than
between a term and the production rate, and like it a local statement at `p`,
not a structural identifiability result; it says nothing about the terms'
functional form beyond their scale.

Returns a vector with one entry per pair, `(; nodes, terms, collinearity)`,
empty when the model has fewer than two unknown terms.
"""
function cross_term_collinearity(model::UDEModel, p, u0, tspan, times;
        mask = nothing, rel_step::Real = 1e-3)
    terms = neural_destruction_terms(model)
    length(terms) ≥ 2 || return NamedTuple{(:nodes, :terms, :collinearity),
        Tuple{Tuple{Symbol, Symbol}, Tuple{Int, Int}, Float64}}[]
    names = [node.name for node in model.network.nodes]
    state_nodes_list = state_nodes(model.network)
    δ = rel_step
    function scaled_rhs(term, sign)
        return function (u, _, t)
            du = ude_rhs(u, p, t, model)
            D = _destruction_contribution(term, term.target, u, p, model.nn, model.st)
            du[term.target] -= sign * δ * D * u[term.target]
            return du
        end
    end
    sensitivities = Vector{Union{Nothing, Vector{Float64}}}(undef, length(terms))
    for (i, term) in pairs(terms)
        plus = _simulate_trajectory(scaled_rhs(term, 1.0), u0, tspan, times)
        minus = _simulate_trajectory(scaled_rhs(term, -1.0), u0, tspan, times)
        sensitivities[i] = plus === nothing || minus === nothing ? nothing :
                           vec((plus .- minus) ./ (2δ))
    end
    observed = mask === nothing ? nothing : vec(mask)
    out = NamedTuple{(:nodes, :terms, :collinearity),
        Tuple{Tuple{Symbol, Symbol}, Tuple{Int, Int}, Float64}}[]
    for i in 1:(length(terms) - 1), j in (i + 1):length(terms)
        a, b = sensitivities[i], sensitivities[j]
        value = if a === nothing || b === nothing
            NaN
        else
            aa = observed === nothing ? a : a[observed]
            bb = observed === nothing ? b : b[observed]
            denom = norm(aa) * norm(bb)
            denom == 0 ? 0.0 : abs(dot(aa, bb)) / denom
        end
        node_i = names[state_nodes_list[terms[i].target]]
        node_j = names[state_nodes_list[terms[j].target]]
        push!(out, (; nodes = (node_i, node_j), terms = (i, j), collinearity = value))
    end
    return out
end

"""
    unidentifiable_edge_from_fisher(; condition_number, collinearity, ...)

Fisher condition-number **or** trajectory-Jacobian cosine flag.
`coefficients_are_biological_constants` is `!edge`. This is not
StructuralIdentifiability.jl and is not a theorem. Not exported.
"""
function unidentifiable_edge_from_fisher(; condition_number, collinearity,
        condition_threshold::Real = 1e6,
        collinearity_threshold::Real = 0.95)
    cond_hit = isfinite(condition_number) &&
               condition_number ≥ condition_threshold
    cosine_hit = isfinite(collinearity) &&
                 collinearity ≥ collinearity_threshold
    return cond_hit || cosine_hit
end

"""Human-readable practical warning for `production_destruction_tradeoff` (not exported)."""
function format_production_destruction_warning(report)
    cosine = report.collinearity
    cosine_str = isfinite(cosine) ? string(round(cosine; digits = 3)) : "NA"
    if report.unidentifiable_edge
        return "Practical warning: production ($(report.production_param)) and unknown D(z) scale are collinear (cosine=$cosine_str). Observed concentrations do not pin that scale. This is not structural identifiability."
    end
    return "Practical $(report.production_param)↔D collinearity cosine=$cosine_str (below threshold)."
end

"""Run `production_destruction_tradeoff` and optionally print the warning."""
function report_production_destruction_tradeoff(
        model::UDEModel, p, data, t_data, u0, tspan;
        verbose::Bool = false, kwargs...)
    report = production_destruction_tradeoff(
        model, p, data, t_data, u0, tspan; kwargs...)
    verbose && println(format_production_destruction_warning(report))
    return report
end
