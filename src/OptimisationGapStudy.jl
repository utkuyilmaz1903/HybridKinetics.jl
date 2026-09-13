# The optimisation gap (0.19).
#
# 0.18 measured a solution the training path does not reach: pre-trained to the
# true rates, the same data leave both terms of the coupled fixture within 1 per
# cent of the truth on the median, at about a tenth of the final training loss,
# in 5 of 5 seeds. Two of the package's headline weaknesses follow from that gap
# -- the extra terms of the discovered support and the adjacent-node bias.
#
# This file measures the gap rather than closing it:
#
#   the floor    what selecting among the initialisations already run buys
#   the ceiling  what the workflow produces from the true-rate start
#   the path     the training loss along the straight line between the two
#   the map      what support the reference discovery gives for a rate of a
#                given accuracy, so that the reachable accuracy can be read
#                against the accuracy the support actually needs
#
# Nothing here changes a default or touches the discovery code: the study calls
# the package, the package does not call the study. Not exported.

"""Seeds of the optimisation-gap study, the library comparison study's own."""
const GAP_SEEDS = LIBRARY_STUDY_SEEDS

"""Mixing fractions between the true rate and a trained rate."""
const GAP_ALPHAS = collect(0.0:0.05:1.0)

"""Starting points the study can train from."""
const GAP_INITIALISATIONS = (:standard, :true_rate, :analytic, :analytic_scale)

const GAP_COLUMNS = (:fixture, :seed, :init_seed, :noise, :initialisation,
    :adam_iterations, :bfgs_iterations, :warmup, :initial_loss, :final_loss,
    :retcode, :converged, :final_gradient_norm, :nn_rate_rmse, :support,
    :support_f1, :support_recall, :n_terms, :holdout_residual, :train_time_s)

# -- Initialisation ------------------------------------------------------------

"""
The data of `params` carrying ones on the neural term's own parameters and
zeros everywhere else. One unknown term gives a single network whose parameters
are all of `.nn`; several give a multi-head network with one block each. The
physical parameters stay where the package's own path puts them.
"""
function _gap_network_mask(params, term)
    selector = zero(params)
    name = Symbol("head_$(term.nn_index)")
    if hasproperty(selector.nn, name)
        getproperty(selector.nn, name) .= 1.0
    else
        selector.nn .= 1.0
    end
    return ComponentArrays.getdata(selector)
end

"""
`params` with the neural term pre-trained to `target` over the regulator values
`r`, leaving every other parameter alone. The rate is read through
`_destruction_contribution`, which carries a gradient;
`sample_unknown_destruction` fills a vector by index and returns none, so a
loss built on it would train nothing.
"""
function gap_pretrain_network(model, params, term, r, target;
        iterations::Int = 800, learning_rate::Float64 = 0.02,
        fill_value::Real = 0.3)
    nstates = model.compiled.nstates
    inputs = [begin
                  x = fill(float(fill_value), nstates)
                  x[term.regulator] = value
                  x
              end
              for value in r]
    y = collect(float.(target))
    weights = 1.0 ./ (y .+ 0.05 * maximum(y))
    mask = _gap_network_mask(params, term)
    axes = ComponentArrays.getaxes(params)
    rate = (p, x) -> _destruction_contribution(
        term, term.target, x, p, model.nn, model.st)
    loss = p -> sum(i -> abs2(weights[i] * (rate(p, inputs[i]) - y[i])),
        eachindex(y)) / length(y)
    current = copy(params)
    state = Optimisers.setup(Optimisers.Adam(learning_rate), current)
    for _ in 1:iterations
        value, back = Zygote.pullback(loss, current)
        gradient = back(one(value))[1]
        gradient === nothing && break
        data = ComponentArrays.getdata(gradient) .* mask
        all(isfinite, data) || break
        state, current = Optimisers.update(state, current,
            ComponentArrays.ComponentVector(data, axes))
    end
    return current
end

"""The true Hill rate of the two-state fixture over a regulator grid."""
function gap_true_rate_on(r, truth)
    return hill_rate_truth(collect(float.(r)); vmax = truth.vmax, K = truth.K,
        n = truth.n)
end

# -- A rate estimated from the data alone --------------------------------------

"""
A crude destruction rate read straight off the data. The two-state fixture is
`dS/dt = k_prod R - D(R) S`, so `D(R) = (k_prod R - dS/dt) / S` at every
observed point, with `dS/dt` a central difference. `k_prod` is not known to the
fit either, and the production rate and the destruction scale trade off against
each other -- the identifiability limit the package already documents -- so the
estimate is only good up to a scale. It uses the same flat guess the package's
own path starts from and adds no outside information.

Returns the sampled pairs and a smoothed rate on `grid`: the points are binned
by regulator value and each bin takes its median, which is robust to the
outliers that dividing by a small target state produces.
"""
function gap_crude_rate(set::ExperimentSet, grid; k_prod_guess::Real = 0.8,
        target_row::Int = 1, regulator_row::Int = 2, bins::Int = 12,
        floor_value::Real = 1.0e-3)
    regulators = Float64[]
    rates = Float64[]
    for experiment in set.experiments
        observations = experiment.observations
        times = experiment.times
        for i in 2:(length(times) - 1)
            target = observations[target_row, i]
            regulator = observations[regulator_row, i]
            (isfinite(target) && isfinite(regulator)) || continue
            target > 1.0e-6 || continue
            ahead = observations[target_row, i + 1]
            behind = observations[target_row, i - 1]
            (isfinite(ahead) && isfinite(behind)) || continue
            derivative = (ahead - behind) / (times[i + 1] - times[i - 1])
            isfinite(derivative) || continue
            push!(regulators, regulator)
            push!(rates, (k_prod_guess * regulator - derivative) / target)
        end
    end
    values = collect(float.(grid))
    isempty(regulators) &&
        return (; regulators, rates, smoothed = fill(NaN, length(values)))
    lo, hi = extrema(regulators)
    edges = collect(range(lo, hi; length = bins + 1))
    centres = Float64[]
    medians = Float64[]
    for b in 1:bins
        left = edges[b]
        right = b == bins ? edges[end] + eps() : edges[b + 1]
        inside = [rates[k] for k in eachindex(rates) if left ≤ regulators[k] < right]
        isempty(inside) && continue
        push!(centres, (left + min(right, edges[end])) / 2)
        push!(medians, median(inside))
    end
    length(centres) ≥ 2 || return (; regulators, rates,
        smoothed = fill(max(float(floor_value), median(rates)), length(values)))
    smoothed = map(values) do value
        if value ≤ first(centres)
            first(medians)
        elseif value ≥ last(centres)
            last(medians)
        else
            k = findlast(≤(value), centres)
            if k === nothing
                first(medians)
            elseif k ≥ length(centres)
                last(medians)
            else
                span = centres[k + 1] - centres[k]
                weight = span == 0 ? 0.0 : (value - centres[k]) / span
                (1 - weight) * medians[k] + weight * medians[k + 1]
            end
        end
    end
    return (; regulators, rates, smoothed = max.(float(floor_value), smoothed))
end

# -- One training --------------------------------------------------------------

"""
One two-state reference-protocol training with the starting point and the
iteration budget under the caller's control.

`initialisation` is `:standard`, the package's own flat physical guess with a
random network, or `:true_rate`, the same guess with the network pre-trained to
the true Hill rate over the regulator grid. The second is not something a user
can do: it is the ceiling, what the rest of the workflow produces when training
starts where training is trying to get to.

`adam_scale` and `bfgs_scale` multiply the reference budgets, and `warmup`
turns the first-experiment warm-up off. At the defaults this is the package's
own path exactly, which `gap_matches_reliability` checks.
"""
function gap_train(; seed::Integer, init_seed::Integer = seed, noise_σ::Real = 0.0,
        initialisation::Symbol = :standard, warmup::Bool = true,
        adam_scale::Real = 1, bfgs_scale::Real = 1, kind::Symbol = :protocol,
        pretrain_iterations::Int = 800)
    initialisation in GAP_INITIALISATIONS || throw(ArgumentError(
        "initialisation must be one of $(GAP_INITIALISATIONS); got $(initialisation)"))
    budget = kind === :smoke ? LIBRARY_STUDY_TWO_STATE_BUDGET.smoke :
             LIBRARY_STUDY_TWO_STATE_BUDGET.protocol
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    started = time()
    set = generate_recovery_experiments(
        MersenneTwister(seed), truth_net, LIBRARY_STUDY_TWO_STATE_PARAMS;
        tspan = budget.tspan, n_points = budget.n_points,
        noise_σ = Float64(noise_σ))
    split = reference_protocol_experiment_split(set)
    model, p0 = build_ude_model(MersenneTwister(init_seed), ude_net)
    names = Tuple(parameter_schema(model).phys_names)
    start = pack_parameters(
        NamedTuple{names}(ntuple(_ -> 0.8, length(names))), p0.nn)
    term = only_unknown_destruction(model)
    truth = LIBRARY_STUDY_TWO_STATE_TRUTH
    if initialisation !== :standard
        r = collect(_regulator_grid(split.train, term))
        target = if initialisation === :true_rate
            gap_true_rate_on(r, truth)
        else
            crude = gap_crude_rate(split.train, r)
            initialisation === :analytic_scale ?
            fill(median(crude.smoothed), length(r)) : crude.smoothed
        end
        start = gap_pretrain_network(model, start, term, r, target;
            iterations = pretrain_iterations)
    end
    locked = lock_training_config(model,
        reference_protocol_training_config(;
            adam_iterations = max(1, round(Int, adam_scale * budget.adam_iterations)),
            bfgs_iterations = max(0, round(Int, bfgs_scale * budget.bfgs_iterations)),
            model = model))
    session = training_solve_session(model, split.train, start; solver = locked.solver)
    trained = if warmup && length(split.train) > 1
        warm = warmup_first_experiment(start, split.train, model;
            config = locked, verbose = false, seed = 0)
        train_experiments(warm.params, split.train, model;
            config = locked, verbose = false, seed = 0,
            optimizer_state = warm.optimizer_state, session = session)
    else
        train_experiments(start, split.train, model;
            config = locked, verbose = false, seed = 0, session = session)
    end
    X = _library_study_two_state_coordinates(
        split.train, term, budget.n_sample_points, budget.x_seed)
    (_, D, _) = sample_unknown_destruction(model, trained.params, X)
    return (; fixture = :two_state, model, params = trained.params, term, start,
        initialisation, warmup, X = Matrix{Float64}(X), D = Matrix{Float64}(D),
        training = trained, config = locked, session,
        train_set = split.train, holdout_set = split.holdout,
        parent_rows = (; true_parent = [2]), truth,
        seed = Int(seed), init_seed = Int(init_seed), noise = Float64(noise_σ),
        adam_iterations = locked.adam_iterations,
        bfgs_iterations = locked.bfgs_iterations,
        train_time = time() - started)
end

"""
Does `:standard` reproduce the package's own path? The ceiling is only worth
reading against a floor produced the same way, so this compares the learned
rate and the parameters against `reliability_train`, which 0.18 already checked
against `fit_unknown_destruction`.
"""
function gap_matches_reliability(; seed::Integer = 103, noise_σ::Real = 0.0,
        kind::Symbol = :protocol)
    own = gap_train(; seed, init_seed = seed, noise_σ, kind)
    other = reliability_train(; seed, init_seed = seed, noise_σ, kind)
    difference = maximum(abs, vec(own.D) .- vec(other.D))
    scale = max(maximum(abs, vec(other.D)), eps())
    return (; rate_difference = difference, relative = difference / scale,
        parameters_equal = vec(own.params) == vec(other.params),
        same = difference / scale < 1.0e-10)
end

# -- Discovery on a rate the caller supplies -----------------------------------

"""
The reference discovery, unchanged, on a rate vector the caller supplies, over
the regulator coordinates of `trained`. This is how the study asks what support
a rate of a given accuracy produces without training anything.
"""
function gap_discover(trained, rate::AbstractVector)
    rows = trained.parent_rows.true_parent
    R = Matrix(trained.X[rows, :])
    result = regress_unknown_rate(R, dummy_trained_graph_local_times(size(R, 2)),
        reshape(collect(Float64, rate), 1, :);
        config = rate_discovery_config(), verbose = false)
    truth = _library_study_truth(trained.truth, rows)
    scores = library_study_scores(result, truth; names = [:S, :R])
    candidate = scores.candidate
    support = candidate isa ImplicitCandidate ?
              _reliability_support_label(candidate) : ""
    n_terms = candidate isa ImplicitCandidate ?
              count(!=(0.0), candidate.numerator_coefficients) +
              count(!=(0.0), candidate.denominator_coefficients) : 0
    return (; scores, candidate, support, n_terms,
        success = scores.success,
        support_f1 = scores.support_f1, support_recall = scores.support_recall)
end

"""The true Hill rate over the sample grid of `trained`."""
gap_true_rate(trained) = gap_true_rate_on(
    vec(trained.X[trained.truth.variable, :]), trained.truth)

"""
How accurate must a rate be before the discovered support changes? The rate is
mixed from the true law towards the one a real training produced, so the error
being scanned has the shape a network actually makes rather than a synthetic
one. `alpha = 0` is the true rate and `alpha = 1` the trained rate.
"""
function gap_alpha_sweep(trained; alphas = GAP_ALPHAS)
    truth_rate = gap_true_rate(trained)
    learned = vec(trained.D)
    rows = NamedTuple[]
    for alpha in alphas
        rate = truth_rate .+ Float64(alpha) .* (learned .- truth_rate)
        discovery = gap_discover(trained, rate)
        push!(rows,
            (; trained.seed, trained.noise, alpha = Float64(alpha),
                rate_error = rate_rel_rmse(rate, truth_rate),
                discovery.support, discovery.support_f1,
                discovery.support_recall, discovery.n_terms))
    end
    return rows
end

# -- The loss between two solutions --------------------------------------------

"""
The loss the optimiser minimises, at an arbitrary parameter vector: each
experiment's masked mean squared error weighted by its metadata weight and
divided by the square of its noise scale, averaged over the weights. Same
composition as the objective inside `train_experiments`.
"""
function gap_training_loss(model, set::ExperimentSet, params; config,
        session = nothing)
    total = 0.0
    weight_sum = 0.0
    for experiment in set.experiments
        weight = experiment_weight(experiment)
        scale = experiment_noise_scale(experiment)
        value = loss_mse(params, experiment.observations, experiment.times,
            experiment.u0,
            (first(experiment.times), last(experiment.times)), model;
            constraint = config.constraint, solver_config = config.solver,
            mask = experiment.mask, session = session)
        total += weight * value / (scale^2)
        weight_sum += weight
    end
    return total / max(weight_sum, 1.0)
end

"""
The training loss along the straight line from `from` to `to`, at `n` evenly
spaced points including both ends. A rise in the middle would mean the better
solution is behind a barrier; a fall all the way would mean it is downhill and
the optimiser stopped before reaching it.
"""
function gap_interpolate(model, set::ExperimentSet, from, to; n::Int = 20, config,
        session = nothing)
    data_from = ComponentArrays.getdata(from)
    data_to = ComponentArrays.getdata(to)
    axes = ComponentArrays.getaxes(from)
    rows = NamedTuple[]
    for t in range(0.0, 1.0; length = n)
        params = ComponentArrays.ComponentVector(
            data_from .+ t .* (data_to .- data_from), axes)
        loss = try
            gap_training_loss(model, set, params; config, session)
        catch
            Inf
        end
        push!(rows, (; t = Float64(t), loss = Float64(loss)))
    end
    return rows
end

"""
Where the gap between two solutions lives. Three partial moves from `from`
towards `to`: the physical parameters alone, the network alone, and a single
scalar on the network's output fitted by least squares to the rate at `to`.
Each is reported by the training loss it reaches and by the error of the rate
it produces against the true law.
"""
function gap_decompose(trained_from, trained_to)
    model = trained_from.model
    term = trained_from.term
    set = trained_from.train_set
    config = trained_from.config
    truth_rate = gap_true_rate(trained_from)
    X = trained_from.X
    measure = function (params)
        (_, D, _) = sample_unknown_destruction(model, params, X)
        loss = try
            gap_training_loss(model, set, params; config)
        catch
            Inf
        end
        return (; loss = Float64(loss),
            rate_error = rate_rel_rmse(vec(D), truth_rate))
    end
    from = trained_from.params
    to = trained_to.params
    physical = copy(from)
    physical.phys .= to.phys
    network = copy(from)
    network.nn .= to.nn
    # the best single scalar on this network's output, in the least-squares
    # sense against the rate the other solution produces
    (_, D_from, _) = sample_unknown_destruction(model, from, X)
    (_, D_to, _) = sample_unknown_destruction(model, to, X)
    numerator = sum(vec(D_from) .* vec(D_to))
    denominator = max(sum(abs2, vec(D_from)), eps())
    factor = numerator / denominator
    scaled_rate = factor .* vec(D_from)
    return (; seed = trained_from.seed,
        from = measure(from), to = measure(to),
        physical_only = measure(physical), network_only = measure(network),
        output_scale = (; factor,
            rate_error = rate_rel_rmse(scaled_rate, truth_rate)))
end

# -- Convergence ---------------------------------------------------------------

"""
Did training stop at a minimum or run out of iterations? Reads the loss history
and the gradient-norm history the training path already records, so it measures
the package's own run rather than a replica of it. `tail_fraction` is the share
of the history counted as the end of the run.
"""
function gap_convergence(trained; tail_fraction::Real = 0.1)
    history = collect(Float64, trained.training.history)
    length(history) ≥ 2 || return (; iterations = length(history),
        total_decrease = NaN, tail_decrease = NaN, tail_share = NaN,
        final_gradient_norm = NaN, converged = trained.training.converged)
    tail = max(2, ceil(Int, tail_fraction * length(history)))
    total = first(history) - last(history)
    tail_drop = history[end - tail + 1] - last(history)
    diagnostics = trained.training.diagnostics
    norm = hasproperty(diagnostics, :final_gradient_norm) ?
           Float64(diagnostics.final_gradient_norm) : NaN
    return (; iterations = length(history), total_decrease = total,
        tail_decrease = tail_drop,
        tail_share = total > 0 ? tail_drop / total : NaN,
        final_gradient_norm = norm, converged = trained.training.converged)
end

# -- Rows, CSV and summaries ---------------------------------------------------

"""One row per training, with the discovery its rate produces."""
function gap_row(trained, discovery)
    holdout = NaN
    if discovery.candidate !== nothing
        rate_fn = try
            equation_to_function(discovery.candidate)
        catch
            nothing
        end
        if rate_fn !== nothing
            residuals = Float64[]
            for experiment in trained.holdout_set.experiments
                value = try
                    _reliability_residual(trained.model, trained.params,
                        trained.term, rate_fn, trained.parent_rows.true_parent,
                        experiment)
                catch
                    Inf
                end
                push!(residuals, value)
            end
            holdout = isempty(residuals) ? NaN : mean(residuals)
        end
    end
    truth_rate = gap_true_rate(trained)
    return (; trained.fixture, trained.seed, trained.init_seed, trained.noise,
        initialisation = String(trained.initialisation),
        trained.adam_iterations, trained.bfgs_iterations,
        warmup = trained.warmup,
        initial_loss = Float64(trained.training.initial_loss),
        final_loss = Float64(trained.training.final_loss),
        retcode = string(trained.training.retcode),
        converged = trained.training.converged,
        final_gradient_norm = Float64(
            trained.training.diagnostics.final_gradient_norm),
        nn_rate_rmse = rate_rel_rmse(vec(trained.D), truth_rate),
        discovery.support, discovery.support_f1, discovery.support_recall,
        discovery.n_terms, holdout_residual = holdout,
        train_time_s = round(trained.train_time; digits = 1))
end

function gap_csv_line(row)
    values = map(GAP_COLUMNS) do column
        value = getproperty(row, column)
        value isa AbstractString || value isa Symbol ?
        replace(string(value), "," => ";") :
        value isa Bool ? string(value) :
        value isa Integer ? string(value) : string(Float64(value))
    end
    return join(values, ",")
end

function append_gap_row(path::AbstractString, row)
    mkpath(dirname(path))
    fresh = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        fresh && println(io, join(string.(GAP_COLUMNS), ","))
        println(io, gap_csv_line(row))
    end
    return path
end

function read_gap_csv(path::AbstractString)
    isfile(path) || return NamedTuple[]
    lines = readlines(path)
    length(lines) ≤ 1 && return NamedTuple[]
    header = Symbol.(split(lines[1], ","))
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ","; limit = length(header))
        length(fields) == length(header) || continue
        values = map(zip(header, fields)) do (column, text)
            column in (:fixture, :initialisation, :retcode, :support) ? String(text) :
            column in (:seed, :init_seed, :adam_iterations, :bfgs_iterations,
                :n_terms) ? parse(Int, text) :
            column in (:warmup, :converged) ? (text == "true") :
            parse(Float64, text)
        end
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end
    return rows
end

"""The floor and the ceiling side by side, per initialisation."""
function gap_summary(rows)
    out = NamedTuple[]
    for initialisation in unique(row.initialisation for row in rows)
        group = [row for row in rows if row.initialisation == initialisation]
        isempty(group) && continue
        push!(out,
            (; initialisation, n = length(group),
                final_loss = median(row.final_loss for row in group),
                rate_error = median(row.nn_rate_rmse for row in group),
                support_f1 = median(row.support_f1 for row in group),
                recall = median(row.support_recall for row in group),
                supports = length(unique(row.support for row in group)),
                holdout = median(row.holdout_residual for row in group)))
    end
    return out
end

function format_gap_summary(rows)
    isempty(rows) && return "no rows"
    io = IOBuffer()
    println(io,
        "| start | runs | final loss median | rate error median | F1 median | recall median | distinct supports | held-out median |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for row in gap_summary(rows)
        println(io, "| $(row.initialisation) | $(row.n) | ",
            round(row.final_loss; sigdigits = 3), " | ",
            round(row.rate_error; digits = 4), " | ",
            round(row.support_f1; digits = 3), " | ",
            round(row.recall; digits = 3), " | ", row.supports, " | ",
            round(row.holdout; digits = 5), " |")
    end
    return String(take!(io))
end
