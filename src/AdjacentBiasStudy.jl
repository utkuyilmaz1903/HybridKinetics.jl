# The adjacent-node bias (0.18, hypothesis B).
#
# The 0.16 study found that with two unknown destruction terms on adjacent
# nodes — each node regulates the other's term — the downstream term's learned
# rate comes out 15 to 30 per cent low, in every seed. This study asks whether
# that is a property of the data or of the optimiser's search, and measures
# three candidate corrections against the same baseline:
#
#   :joint       the 0.16 path: both terms unknown, the standard
#                initialisation, trained jointly. The baseline.
#   :true_init   both networks pre-trained to their true rates, then the same
#                joint training. If the bias returns, it is a property of the
#                data; if it does not, the standard initialisation's search
#                was finding a biased minimum.
#   :sequential  the upstream term trained with the downstream mechanism
#                known, then frozen at the rate it learned while the
#                downstream term trains. The freeze is real: the frozen rate
#                is compiled as a custom kinetic term and has no
#                representation in the parameter vector.
#   :weighted    each state's residual weighted by the inverse of its observed
#                variance. Expressed with the per-experiment `:weight`
#                metadata the trainer already honours, by splitting each
#                experiment into one single-state replica per state.
#
# `adjacent_scale_scan` measures the fourth question — whether the two terms'
# scales trade off against each other — differently from the way it was
# posed, because a term's `scale` field is a compile-time constant from the
# reaction stoichiometry and not a trained parameter, so "constrain the
# product of the two scales" has nothing to constrain. Instead the two fitted
# rates are rescaled after the fit and the hybrid residual is mapped over the
# rescaling plane, which is the same question asked of the loss surface.
#
# Data, seeds, initial conditions, the split and the training settings are the
# 0.16 study's, so every bias here is comparable with the numbers on the
# benchmarks page.
#
# Not exported.

const ADJACENT_SETTINGS = (:joint, :true_init, :sequential, :weighted)
const ADJACENT_COLUMNS = (:fixture, :seed, :noise, :setting, :node, :role,
    :nn_rate_bias, :nn_rate_rmse, :final_loss, :collinearity, :cross_term_max,
    :data_residual, :holdout_residual, :train_time_s, :note)

"""The data of one cell, generated exactly as the 0.16 multi-term study does."""
function adjacent_data(; fixture::Symbol = :coupled, seed::Integer, noise_σ::Real,
        n_points::Int = REFERENCE_PROTOCOL.n_points,
        n_ics::Int = REFERENCE_PROTOCOL.n_ics)
    fx = multi_term_fixture(fixture)
    truth_net = fx.build(; unknown = ())
    ics = multi_term_study_ics(fx.nstates; n_ics = n_ics)
    set = generate_experiment_set(MersenneTwister(seed); network = truth_net,
        initial_conditions = ics, tspan = REFERENCE_PROTOCOL.tspan,
        n_points = n_points, noise_σ = Float64(noise_σ), truth_params = fx.truth)
    holdout = 2
    train_set = ExperimentSet(set.experiments[1:(end - holdout)], set.state_names)
    holdout_set = ExperimentSet(set.experiments[(end - holdout + 1):end], set.state_names)
    return (; fx, set, train_set, holdout_set, truth_net)
end

"""The reference training config of the multi-term study."""
adjacent_training_config() = TrainingConfig(
    adam_iterations = REFERENCE_PROTOCOL.adam_iterations,
    bfgs_iterations = REFERENCE_PROTOCOL.bfgs_iterations, log_every = 10^6)

"""
The warm-up and joint fit that `discover_unknown_terms` performs, with the
starting parameters given, so that a pre-trained initialisation can be used.
"""
function adjacent_fit(model, start, train_set; config = adjacent_training_config(),
        warmup::Bool = true, verbose::Bool = false)
    first_exp = first(train_set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    params = start
    if warmup
        warm = train_ude(params, first_exp.observations, first_exp.times,
            first_exp.u0, tspan, model;
            config = TrainingConfig(config; bfgs_iterations = 0,
                horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0])),
            verbose = verbose, mask = first_exp.mask)
        params = warm.params
    end
    return train_experiments(params, train_set, model; config = config, verbose = verbose)
end

"""A flat physical guess of 0.8, the package's own starting point."""
function adjacent_initial_parameters(model, p0)
    names = Tuple(parameter_schema(model).phys_names)
    return pack_parameters(NamedTuple{names}(ntuple(_ -> 0.8, length(names))), p0.nn)
end

# -- Measuring one term --------------------------------------------------------

"""
The learned rate of one term on the grid `discover_unknown_terms` would use,
with the signed relative bias and the relative error against the true Hill
rate. Same measurement as the 0.16 study's `nn_rate_bias` / `nn_rate_rmse`.
"""
function adjacent_term_scores(model, params, term, train_set, fx, node)
    r_range = _regulator_grid(train_set, term)
    R, D, _ = sample_unknown_destruction_grid(model, params, term; r_range = r_range)
    vmax_name, K_name = fx.hill[node]
    truth = hill_rate_truth(vec(R); vmax = getproperty(fx.truth, vmax_name),
        K = getproperty(fx.truth, K_name), n = 2)
    return (; R, D,
        nn_rate_bias = _relative_bias(vec(D), truth),
        nn_rate_rmse = rate_rel_rmse(vec(D), truth))
end

"""Node names of a compiled model's neural terms, in term order."""
function adjacent_term_nodes(model)
    node_names = [node.name for node in model.network.nodes]
    state_ids = state_nodes(model.network)
    return [node_names[state_ids[term.target]]
            for term in neural_destruction_terms(model)]
end

"""
Whether each term's regulator is the other unknown node.

In the coupled fixture BOTH terms are: `A`'s rate is a Hill term in `B` and
`B`'s is a Hill term in `A`. The topology is symmetric, so the interaction
graph does not say which term is "upstream" — what breaks the symmetry is the
parameters (`vmax_a = 1.5, K_a = 0.5` against `vmax_b = 1.2, K_b = 0.6`, with
different production rates), which a user does not know for the unknown
terms. A user therefore cannot read the ordering off the graph, and the
sequential experiment runs both orderings rather than assuming one.
"""
function adjacent_roles(model)
    nodes = adjacent_term_nodes(model)
    node_names = [node.name for node in model.network.nodes]
    state_ids = state_nodes(model.network)
    roles = Dict{Symbol, String}()
    terms = neural_destruction_terms(model)
    unknown_nodes = Set(nodes)
    for (term, node) in zip(terms, nodes)
        regulators = [node_names[state_ids[r]] for r in term.regulators]
        roles[node] = any(in(unknown_nodes), regulators) ?
                      "regulated by the other unknown" : "regulated by a known node"
    end
    return roles
end

"""Is the ordering of the two terms readable from the interaction graph alone?"""
function adjacent_ordering_is_graph_readable(model)
    roles = adjacent_roles(model)
    return length(unique(values(roles))) > 1
end

# -- The settings --------------------------------------------------------------

"""
Pre-train one head so that its rate matches `target` on the grid `r`, leaving
the physical parameters and every other head exactly as they were.

The loss calls `_destruction_contribution` rather than
`sample_unknown_destruction`, because the latter fills a plain vector by
index and Zygote cannot differentiate through it — a loss built on it returns
no gradient and the fitting loop would silently do nothing. The gradient is
also masked to this head's block, so the other coordinates are untouched by
construction and not merely by the gradient happening to vanish.
"""
function adjacent_pretrain_head!(model, params, term, r, target;
        iterations::Int = 800, learning_rate::Float64 = 0.02, fill_value::Real = 0.3)
    nstates = model.compiled.nstates
    inputs = [begin
                  x = fill(float(fill_value), nstates)
                  x[term.regulator] = value
                  x
              end
              for value in r]
    y = collect(float.(target))
    weights = 1.0 ./ (y .+ 0.05 * maximum(y))
    selector = zero(params)
    getproperty(selector.nn, Symbol("head_$(term.nn_index)")) .= 1.0
    mask = ComponentArrays.getdata(selector)
    axes = ComponentArrays.getaxes(params)
    rate = (p, x) -> _destruction_contribution(term, term.target, x, p, model.nn, model.st)
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

"""Both heads pre-trained to their true rates, leaving the physical guess alone."""
function adjacent_true_initial_parameters(model, p0, train_set, fx)
    params = adjacent_initial_parameters(model, p0)
    terms = neural_destruction_terms(model)
    nodes = adjacent_term_nodes(model)
    for (term, node) in zip(terms, nodes)
        r = collect(_regulator_grid(train_set, term))
        vmax_name, K_name = fx.hill[node]
        target = hill_rate_truth(r; vmax = getproperty(fx.truth, vmax_name),
            K = getproperty(fx.truth, K_name), n = 2)
        params = adjacent_pretrain_head!(model, params, term, r, target)
    end
    return params
end

"""
One experiment split into one replica per state, each replica observing only
that state and carrying the weight given. The trainer already multiplies each
experiment's loss by `metadata[:weight]`, so this expresses a per-state weight
without touching the trainer.
"""
function adjacent_weighted_set(set::ExperimentSet, weights::AbstractVector)
    experiments = Experiment[]
    for experiment in set.experiments
        for state in eachindex(weights)
            mask = falses(size(experiment.observations))
            mask[state, :] .= experiment.mask[state, :]
            any(mask) || continue
            metadata = copy(experiment.metadata)
            metadata[:weight] = weights[state]
            push!(experiments,
                Experiment(Symbol(experiment.name, :_, state), experiment.times,
                    experiment.observations, experiment.u0;
                    mask = mask, metadata = metadata))
        end
    end
    return ExperimentSet(experiments, set.state_names)
end

"""Inverse observed variance of each state over the training experiments."""
function adjacent_inverse_variance_weights(set::ExperimentSet)
    nstates = length(set.state_names)
    weights = Float64[]
    for state in 1:nstates
        values = Float64[]
        for experiment in set.experiments
            for j in axes(experiment.observations, 2)
                experiment.mask[state, j] && isfinite(experiment.observations[state, j]) &&
                    push!(values, experiment.observations[state, j])
            end
        end
        variance = length(values) > 1 ? var(values) : 1.0
        push!(weights, variance > 0 ? 1 / variance : 1.0)
    end
    return weights ./ maximum(weights)
end

"""
A network in which `node`'s destruction is a fixed custom rate and the other
unknown node's term is still neural. The frozen rate has no representation in
the parameter vector, so this is a freeze and not a gradient mask.
"""
function adjacent_frozen_network(fx, frozen_node::Symbol, unknown_node::Symbol, rate)
    network = fx.build(; unknown = (frozen_node, unknown_node))
    reactions = ReactionSpec[]
    node_index = Dict(node.name => index for (index, node) in pairs(network.nodes))
    frozen_index = node_index[frozen_node]
    for reaction in network.reactions
        is_frozen_decay = !reaction.known &&
                          get(reaction.stoichiometry, frozen_index, 0.0) < 0
        if is_frozen_decay
            push!(reactions,
                ReactionSpec(name = reaction.name,
                    stoichiometry = reaction.stoichiometry,
                    regulators = reaction.regulators,
                    known = true, family = CUSTOM_KINETIC,
                    metadata = CustomKineticMetadata(evaluator = rate)))
        else
            push!(reactions, reaction)
        end
    end
    return BiologicalNetwork(network.nodes, EdgeSpec[]; reactions = reactions)
end

# -- One run -------------------------------------------------------------------

"""
One cell of the study at one setting. Returns one row per unknown term, with
the signed relative bias of its learned rate, its role in the graph, and the
cross-term collinearity of the fit.
"""
function adjacent_run(; setting::Symbol, seed::Integer, noise_σ::Real = 0.0,
        fixture::Symbol = :coupled, sequential_first::Union{Nothing, Symbol} = nothing,
        verbose::Bool = false)
    setting in ADJACENT_SETTINGS || throw(ArgumentError(
        "setting must be one of $(ADJACENT_SETTINGS); got $(setting)"))
    started = time()
    data = adjacent_data(; fixture, seed, noise_σ)
    fx = data.fx
    note = ""
    ude_net = fx.build(; unknown = Tuple(fx.nodes))
    model, p0 = build_ude_model(MersenneTwister(seed), ude_net)
    train_set = data.train_set
    trained = if setting === :joint
        adjacent_fit(model, adjacent_initial_parameters(model, p0), train_set)
    elseif setting === :true_init
        start = adjacent_true_initial_parameters(model, p0, train_set, fx)
        adjacent_fit(model, start, train_set)
    elseif setting === :weighted
        weights = adjacent_inverse_variance_weights(train_set)
        note = "weights " * join(string.(round.(weights; digits = 3)), ";")
        adjacent_fit(model, adjacent_initial_parameters(model, p0),
            adjacent_weighted_set(train_set, weights))
    elseif setting === :sequential
        first_node = sequential_first === nothing ? fx.nodes[1] : sequential_first
        other_node = first(node for node in fx.nodes if node != first_node)
        return adjacent_sequential_run(; seed, noise_σ, fixture, data, verbose, started,
            first_node, other_node)
    end
    return adjacent_rows(; model, params = trained.params, trained, data, fx,
        setting, seed, noise_σ, fixture, note, started)
end

"""
`:sequential`. The upstream term is trained with the downstream mechanism
known at its true parameters; the rate it learns is then frozen as a custom
kinetic term while the downstream term trains alone.
"""
function adjacent_sequential_run(; seed, noise_σ, fixture, data, verbose, started,
        first_node::Symbol, other_node::Symbol)
    fx = data.fx
    train_set = data.train_set
    upstream, downstream = first_node, other_node
    # Step 1: the upstream term alone, the downstream mechanism known.
    first_net = fx.build(; unknown = (upstream,))
    first_model, first_p0 = build_ude_model(MersenneTwister(seed), first_net)
    first_fit = adjacent_fit(first_model,
        adjacent_initial_parameters(first_model, first_p0), train_set)
    first_term = only(neural_destruction_terms(first_model))
    # Step 2: freeze it, train the downstream term.
    frozen_params = first_fit.params
    frozen_nn = first_model.nn
    frozen_st = first_model.st
    frozen_target = first_term.target
    frozen_scale = first_term.scale
    # The evaluator is called inside the right-hand side the optimiser
    # differentiates, so it must not mutate and must not go through
    # `sample_unknown_destruction`, which fills a vector by index and returns
    # no gradient. `_destruction_contribution` is the same rate, read straight
    # off the frozen network, and the custom term reapplies its own scale.
    frozen_rate = function (x, _p, _regulators)
        value = _destruction_contribution(first_term, frozen_target, x,
            frozen_params, frozen_nn, frozen_st)
        return value / frozen_scale
    end
    second_net = adjacent_frozen_network(fx, upstream, downstream, frozen_rate)
    second_model, second_p0 = build_ude_model(MersenneTwister(seed), second_net)
    length(neural_destruction_terms(second_model)) == 1 || throw(ErrorException(
        "the frozen network must leave exactly one neural term"))
    frozen_term = only(term
    for term in second_model.compiled.destruction_terms
    if term isa CustomDestructionTerm)
    frozen_term.scale == frozen_scale || throw(ErrorException(
        "the frozen term's scale ($(frozen_term.scale)) differs from the neural " *
        "term it replaces ($(frozen_scale)); the frozen rate would not be the " *
        "rate that was learned"))
    second_fit = adjacent_fit(second_model,
        adjacent_initial_parameters(second_model, second_p0), train_set)
    rows = NamedTuple[]
    for (model, params, fit, node, role) in (
        (first_model, first_fit.params, first_fit, upstream, "trained first, then frozen"),
        (second_model, second_fit.params, second_fit, downstream, "trained second"))
        term = only(neural_destruction_terms(model))
        scores = adjacent_term_scores(model, params, term, train_set, fx, node)
        push!(rows,
            (; fixture, seed = Int(seed), noise = Float64(noise_σ),
                setting = :sequential, node, role,
                scores.nn_rate_bias, scores.nn_rate_rmse,
                final_loss = fit.final_loss, collinearity = NaN, cross_term_max = NaN,
                data_residual = NaN, holdout_residual = NaN,
                train_time_s = round(time() - started; digits = 1),
                note = "upstream $(upstream) frozen for the downstream fit"))
    end
    return rows
end

"""One row per unknown term of a fitted two-term model."""
function adjacent_rows(; model, params, trained, data, fx, setting, seed, noise_σ,
        fixture, note, started)
    train_set = data.train_set
    terms = neural_destruction_terms(model)
    nodes = adjacent_term_nodes(model)
    roles = adjacent_roles(model)
    first_exp = first(train_set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    cross = try
        cross_term_collinearity(model, params, first_exp.u0, tspan, first_exp.times;
            mask = first_exp.mask)
    catch
        []
    end
    cross_max = isempty(cross) ? NaN : maximum(p.collinearity for p in cross)
    rows = NamedTuple[]
    for (term, node) in zip(terms, nodes)
        scores = adjacent_term_scores(model, params, term, train_set, fx, node)
        identifiability = try
            report_production_destruction_tradeoff(model, params,
                first_exp.observations, first_exp.times, first_exp.u0, tspan;
                term = term, verbose = false, mask = first_exp.mask,
                production_param = fx.production[node])
        catch
            nothing
        end
        push!(rows,
            (; fixture, seed = Int(seed), noise = Float64(noise_σ), setting,
                node, role = roles[node], scores.nn_rate_bias, scores.nn_rate_rmse,
                final_loss = trained.final_loss,
                collinearity = identifiability === nothing ? NaN :
                               Float64(identifiability.collinearity),
                cross_term_max = cross_max,
                data_residual = NaN, holdout_residual = NaN,
                train_time_s = round(time() - started; digits = 1), note))
    end
    return rows
end

# -- The scale question --------------------------------------------------------

"""
How the hybrid residual varies when the two fitted rates are rescaled. A
term's `scale` is a compile-time constant from the reaction stoichiometry, not
a trained parameter, so the two scales cannot be constrained against each
other; what can be measured is whether the data distinguish them at all.
Returns the residual over a grid of `(factor of the first term, factor of the
second)`, which is flat along a trade-off direction when they are not
separately determined.
"""
function adjacent_scale_scan(; seed::Integer, noise_σ::Real = 0.0,
        fixture::Symbol = :coupled, factors = (0.8, 0.9, 1.0, 1.1, 1.25))
    data = adjacent_data(; fixture, seed, noise_σ)
    fx = data.fx
    ude_net = fx.build(; unknown = Tuple(fx.nodes))
    model, p0 = build_ude_model(MersenneTwister(seed), ude_net)
    trained = adjacent_fit(model, adjacent_initial_parameters(model, p0), data.train_set)
    terms = neural_destruction_terms(model)
    nodes = adjacent_term_nodes(model)
    rates = map(zip(terms, nodes)) do (term, node)
        r_range = _regulator_grid(data.train_set, term)
        R, D, _ = sample_unknown_destruction_grid(model, trained.params, term;
            r_range = r_range)
        r = vec(R)
        values = vec(D)
        return function (x)
            position = clamp(searchsortedfirst(r, first(x)), 1, length(values))
            return values[position]
        end
    end
    first_exp = first(data.train_set.experiments)
    rows = NamedTuple[]
    for a in factors, b in factors
        pairs = Pair{Any, Any}[terms[1] => (x -> a * rates[1](x)),
            terms[2] => (x -> b * rates[2](x))]
        residual = try
            hybrid_data_residual(model, trained.params, pairs, first_exp.u0,
                (first(first_exp.times), last(first_exp.times)), first_exp.times,
                first_exp.observations; mask = first_exp.mask)
        catch
            Inf
        end
        push!(rows,
            (; seed = Int(seed), noise = Float64(noise_σ), fixture,
                factor_1 = a, factor_2 = b, node_1 = nodes[1], node_2 = nodes[2],
                residual = Float64(residual)))
    end
    return rows
end

# -- CSV and summary -----------------------------------------------------------

function adjacent_csv_line(row)
    values = map(ADJACENT_COLUMNS) do column
        value = getproperty(row, column)
        # commas would break the row, so text fields carry none
        value isa AbstractString || value isa Symbol ? replace(string(value), "," => ";") :
        value isa Integer ? string(value) : string(Float64(value))
    end
    return join(values, ",")
end

function append_adjacent_row(path::AbstractString, row)
    mkpath(dirname(path))
    fresh = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        fresh && println(io, join(string.(ADJACENT_COLUMNS), ","))
        println(io, adjacent_csv_line(row))
    end
    return path
end

function read_adjacent_csv(path::AbstractString)
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
            column in (:fixture, :setting, :node, :role, :note) ? String(text) :
            column === :seed ? parse(Int, text) : parse(Float64, text)
        end
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end
    return rows
end

"""The median, or `NaN` when the sequential setting recorded no value."""
_adjacent_median_or_nan(values) = isempty(values) ? NaN : median(values)

"""
The label a row is grouped under. The sequential setting is two experiments,
one per ordering, and pooling them would average a term that was fitted first
with the same term fitted second.
"""
function adjacent_setting_label(row)
    setting = string(row.setting)
    setting == "sequential" || return setting
    match = Base.match(r"upstream ([A-Za-z0-9_]+) ", string(row.note))
    return match === nothing ? setting : "sequential, $(match.captures[1]) first"
end

"""
The bias per setting and node. The pre-registered criteria are read against
the joint fit's bias for the same node: returning to within 5 percentage
points of it means the bias is a property of the data; staying under a third
of it means the standard initialisation's search was finding a biased
minimum; a correction confirms when the median bias falls below 5 per cent.
"""
function adjacent_summary(rows)
    out = NamedTuple[]
    for setting in unique(adjacent_setting_label(r) for r in rows),
        node in sort(unique(r.node for r in rows))

        group = [r
                 for r in rows
                 if adjacent_setting_label(r) == setting &&
                    string(r.node) == string(node)]
        isempty(group) && continue
        biases = [r.nn_rate_bias for r in group if isfinite(r.nn_rate_bias)]
        isempty(biases) && continue
        push!(out,
            (; setting = string(setting), node = string(node), n = length(group),
                role = first(group).role,
                median_bias = median(biases), min_bias = minimum(biases),
                max_bias = maximum(biases),
                median_rate_rmse = median([r.nn_rate_rmse
                                           for r in group
                                           if isfinite(r.nn_rate_rmse)]),
                median_cross_term = _adjacent_median_or_nan(
                    [r.cross_term_max for r in group
                     if isfinite(r.cross_term_max)]),
                n_under_5pc = count(b -> abs(b) < 0.05, biases)))
    end
    return out
end

function format_adjacent_summary(rows)
    isempty(rows) && return "no rows"
    io = IOBuffer()
    summary = adjacent_summary(rows)
    println(io,
        "| setting | node | runs | bias median | bias range | rate error median | cross-term | runs under 5% |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for row in summary
        println(io, "| $(row.setting) | $(row.node) | $(row.n) | ",
            round(row.median_bias; digits = 3), " | [",
            round(row.min_bias; digits = 3), ", ", round(row.max_bias; digits = 3), "] | ",
            round(row.median_rate_rmse; digits = 3), " | ",
            isnan(row.median_cross_term) ? "NA" :
            string(round(row.median_cross_term; digits = 3)), " | ",
            row.n_under_5pc, " of ", row.n, " |")
    end
    baselines = Dict(row.node => row.median_bias
    for row in summary if row.setting == "joint")
    isempty(baselines) && return String(take!(io))
    # Which node carries the bias is an empirical question here: the coupled
    # fixture's graph is symmetric, so the joint fit names the biased node
    # rather than the topology.
    biased = argmax(node -> abs(baselines[node]), collect(keys(baselines)))
    println(io, "\nthe node the joint fit leaves biased: $(biased) at ",
        round(baselines[biased]; digits = 3))
    println(io, "against the joint fit, per node:")
    for row in summary
        row.setting == "joint" && continue
        haskey(baselines, row.node) || continue
        baseline = baselines[row.node]
        verdict = if row.setting == "true_init"
            # 2.1 asks what the bias is, not whether it is gone.
            abs(row.median_bias - baseline) ≤ 0.05 ?
            "unchanged: the bias is a property of the data" :
            abs(row.median_bias) ≤ abs(baseline) / 3 ?
            "under a third of it: the bias is a property of the search" :
            "in between: neither branch"
        else
            # 2.2 to 2.4 ask whether the correction removes it.
            abs(row.median_bias) < 0.05 ? "median bias under 5%: the correction holds" :
            abs(row.median_bias) < abs(baseline) ? "smaller but still over 5%" :
            "no smaller than the joint fit"
        end
        println(io, "  $(row.setting), $(row.node): ", round(row.median_bias; digits = 3),
            " against ", round(baseline; digits = 3), "  -> ", verdict)
    end
    return String(take!(io))
end
