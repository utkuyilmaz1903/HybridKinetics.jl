# Training reliability (0.18, hypothesis C).
#
# Three measurements a user should be able to see before trusting a run:
#
#   :initialisation  the same data, several random initialisations of the
#                    neural term: the spread of the final loss, of the learned
#                    rate's error, and the number of distinct discovered
#                    supports;
#   :warmup          the same data, a warm-up of zero, of the current length,
#                    and of twice the current length;
#   :divergence      which runs produce a hybrid model that cannot be
#                    resimulated, and whether the discovered candidate alone
#                    predicts it.
#
# The data of a cell are generated from `seed` exactly as the library
# comparison study generates them, and the network is initialised from
# `init_seed`, which the package's own path ties to `seed`; separating the two
# is what makes the initialisation spread measurable at all.
#
# Not exported.

const RELIABILITY_SEEDS = LIBRARY_STUDY_SEEDS
const RELIABILITY_INITIALISATIONS = 10
const RELIABILITY_COLUMNS = (:fixture, :seed, :init_seed, :noise, :experiment,
    :setting, :final_loss, :initial_loss, :retcode, :nn_rate_rmse,
    :support, :support_f1, :support_recall, :n_terms, :denominator_min,
    :holdout_residual, :diverged, :train_time_s)

# -- One training with the data seed and the initialisation seed separated -----

"""
One two-state training with the data of `seed` and the network initialised
from `init_seed`. `warmup_scale` multiplies the warm-up's Adam iterations
(`0` skips the warm-up); `1` is the package's own warm-up, and
`reliability_matches_package` checks that this path reproduces it.
`mask_state` hides every `mask_stride`-th observation of one state, the
sparsely observed species of the p53 case study.
"""
function reliability_train(; seed::Integer, init_seed::Integer, noise_σ::Real,
        warmup_scale::Real = 1, mask_state::Union{Nothing, Int} = nothing,
        mask_stride::Integer = 2, kind::Symbol = :protocol)
    budget = kind === :smoke ? LIBRARY_STUDY_TWO_STATE_BUDGET.smoke :
             LIBRARY_STUDY_TWO_STATE_BUDGET.protocol
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    started = time()
    set = generate_recovery_experiments(
        MersenneTwister(seed), truth_net, LIBRARY_STUDY_TWO_STATE_PARAMS;
        tspan = budget.tspan, n_points = budget.n_points, noise_σ = Float64(noise_σ))
    if mask_state !== nothing
        # Every `mask_stride`-th observation of that state is hidden, which is
        # the sparsely observed species of the p53 case study. Hiding the
        # state outright would leave the regulator grid nothing to span, so it
        # would measure the absence of a regulator, not a warm-up.
        experiments = map(set.experiments) do experiment
            observations = copy(experiment.observations)
            observations[mask_state, 2:mask_stride:end] .= NaN
            Experiment(experiment.name, experiment.times, observations, experiment.u0)
        end
        set = ExperimentSet(experiments, set.state_names)
    end
    split = reference_protocol_experiment_split(set)
    model, p0 = build_ude_model(MersenneTwister(init_seed), ude_net)
    # The package's own path: a flat physical guess, the reference training
    # config, a warm-up on the first experiment, then the joint fit reusing the
    # warm-up's optimizer state (`fit_unknown_destruction` ->
    # `train_experiments_with_warmup`). `warmup_scale` scales only the warm-up's
    # Adam iterations, so `warmup_scale = 1` is that path exactly.
    names = Tuple(parameter_schema(model).phys_names)
    ude_init = pack_parameters(
        NamedTuple{names}(ntuple(_ -> 0.8, length(names))), p0.nn)
    locked = lock_training_config(model,
        reference_protocol_training_config(;
            adam_iterations = budget.adam_iterations,
            bfgs_iterations = budget.bfgs_iterations, model = model))
    session = training_solve_session(model, split.train, ude_init; solver = locked.solver)
    trained = if warmup_scale > 0 && length(split.train) > 1
        warm_iterations = max(1, round(Int, warmup_scale * locked.adam_iterations))
        warm = warmup_first_experiment(ude_init, split.train, model;
            config = TrainingConfig(locked; adam_iterations = warm_iterations),
            verbose = false, seed = 0)
        train_experiments(warm.params, split.train, model;
            config = locked, verbose = false, seed = 0,
            optimizer_state = warm.optimizer_state, session = session)
    else
        train_experiments(ude_init, split.train, model;
            config = locked, verbose = false, seed = 0, session = session)
    end
    term = only_unknown_destruction(model)
    X = _library_study_two_state_coordinates(
        split.train, term, budget.n_sample_points, budget.x_seed)
    (_, D, _) = sample_unknown_destruction(model, trained.params, X)
    return (; fixture = :two_state, model, params = trained.params, term,
        X = Matrix{Float64}(X), D = Matrix{Float64}(D), training = trained,
        train_set = split.train, holdout_set = split.holdout,
        parent_rows = (; true_parent = [2]),
        truth = LIBRARY_STUDY_TWO_STATE_TRUTH,
        seed = Int(seed), init_seed = Int(init_seed), noise = Float64(noise_σ),
        train_time = time() - started)
end

"""
The reference discovery on one training's samples, with the scores and the
denominator minimum the divergence question needs.
"""
function reliability_discovery(trained)
    rows = trained.parent_rows.true_parent
    R = Matrix(trained.X[rows, :])
    result = regress_unknown_rate(R, dummy_trained_graph_local_times(size(R, 2)),
        Matrix(trained.D); config = rate_discovery_config(), verbose = false)
    truth = _library_study_truth(trained.truth, rows)
    scores = library_study_scores(result, truth; names = [:S, :R])
    candidate = scores.candidate
    support = candidate === nothing ? "" :
              _reliability_support_label(candidate)
    denominator_min = candidate === nothing ? NaN : candidate.denominator_minimum
    holdout = NaN
    diverged = false
    if candidate !== nothing
        rate_fn = try
            equation_to_function(candidate)
        catch
            nothing
        end
        if rate_fn !== nothing
            residuals = Float64[]
            for experiment in trained.holdout_set.experiments
                value = try
                    _library_study_residual(trained.model, trained.params,
                        trained.term, rate_fn, rows, experiment)
                catch
                    Inf
                end
                push!(residuals, value)
            end
            holdout = isempty(residuals) ? NaN : mean(residuals)
            diverged = !isfinite(holdout)
        else
            diverged = true
        end
    end
    r = vec(trained.X[trained.truth.variable, :])
    nn_rate_rmse = rate_rel_rmse(vec(trained.D),
        hill_rate_truth(r; vmax = trained.truth.vmax, K = trained.truth.K,
            n = trained.truth.n))
    n_terms = candidate === nothing ? 0 :
              count(!=(0.0), candidate.numerator_coefficients) +
              count(!=(0.0), candidate.denominator_coefficients)
    return (; scores, support, denominator_min, holdout, diverged, nn_rate_rmse,
        n_terms, candidate, rows)
end

function _reliability_support_label(candidate::ImplicitCandidate)
    spec = candidate.specification
    parts = String[]
    for (coefficient, term) in zip(candidate.numerator_coefficients, spec.numerator)
        coefficient != 0.0 && push!(parts, "n:" * term.label)
    end
    for (coefficient, term) in zip(candidate.denominator_coefficients, spec.denominator)
        coefficient != 0.0 && push!(parts, "d:" * term.label)
    end
    return join(parts, " ")
end

function _reliability_row(trained, discovery, experiment, setting)
    return (; trained.fixture, trained.seed, trained.init_seed, trained.noise,
        experiment, setting,
        final_loss = trained.training.final_loss,
        initial_loss = trained.training.initial_loss,
        retcode = string(trained.training.retcode),
        discovery.nn_rate_rmse, discovery.support,
        support_f1 = discovery.scores.support_f1,
        support_recall = discovery.scores.support_recall,
        discovery.n_terms, discovery.denominator_min,
        holdout_residual = discovery.holdout,
        diverged = discovery.diverged,
        train_time_s = round(trained.train_time; digits = 1))
end

# -- The experiments -----------------------------------------------------------

"""
`:initialisation`. The same data, `n` random initialisations of the neural
term. The spread of the final loss, of the learned rate's error and of the
discovered support is the reliability figure a user should see.
"""
function reliability_initialisation(; seeds = RELIABILITY_SEEDS,
        n_initialisations::Integer = RELIABILITY_INITIALISATIONS,
        noise_σ::Real = 0.0, kind::Symbol = :protocol, on_row = nothing)
    rows = NamedTuple[]
    for seed in seeds, index in 1:n_initialisations
        init_seed = 10_000 * Int(seed) + index
        trained = reliability_train(; seed, init_seed, noise_σ, kind)
        discovery = reliability_discovery(trained)
        row = _reliability_row(trained, discovery, "initialisation",
            "init $(index) of $(n_initialisations)")
        push!(rows, row)
        on_row === nothing || on_row(row)
    end
    return rows
end

"""
`:warmup`. The same data at three warm-up lengths: none, the package's own,
and twice the package's own; on the fully observed fixture and on the same
fixture with one state hidden, which is the case the p53 study exposed.
"""
function reliability_warmup(; seeds = RELIABILITY_SEEDS, noise_σ::Real = 0.0,
        scales = (0, 1, 2), kind::Symbol = :protocol, on_row = nothing)
    rows = NamedTuple[]
    for seed in seeds, scale in scales, mask_state in (nothing, 2)
        trained = reliability_train(;
            seed, init_seed = seed, noise_σ, warmup_scale = scale, mask_state, kind)
        discovery = reliability_discovery(trained)
        label = mask_state === nothing ? "fully observed" : "one state hidden"
        row = _reliability_row(trained, discovery, "warmup",
            "warm-up x$(scale), $(label)")
        push!(rows, row)
        on_row === nothing || on_row(row)
    end
    return rows
end

"""
`:divergence`. Every seed at every noise level, recording whether the hybrid
model with the discovered rate can be resimulated and what the discovered
denominator's smallest value was, so the two can be related.
"""
function reliability_divergence(; seeds = RELIABILITY_SEEDS,
        noise_levels = LIBRARY_STUDY_NOISE_LEVELS, kind::Symbol = :protocol,
        on_row = nothing)
    rows = NamedTuple[]
    for seed in seeds, noise_σ in noise_levels
        trained = reliability_train(; seed, init_seed = seed, noise_σ, kind)
        discovery = reliability_discovery(trained)
        row = _reliability_row(trained, discovery, "divergence", "reference")
        push!(rows, row)
        on_row === nothing || on_row(row)
    end
    return rows
end

"""
Does this path reproduce the package's own training at `warmup_scale = 1`?
The initialisation spread is only meaningful if it does.
"""
function reliability_matches_package(; seed::Integer = 103, noise_σ::Real = 0.0,
        kind::Symbol = :protocol)
    own = reliability_train(; seed, init_seed = seed, noise_σ, kind)
    package = _library_study_train(:two_state, seed, noise_σ, kind,
        fit_unknown_destruction, LIBRARY_STUDY_HOLDOUT_ICS, nothing, :varying)
    rate_difference = maximum(abs, vec(own.D) .- vec(package.D))
    scale = max(maximum(abs, vec(package.D)), eps())
    return (; own_loss = own.training.final_loss,
        rate_difference, relative = rate_difference / scale,
        parameters_equal = vec(own.params) == vec(package.params),
        same = rate_difference / scale < 1.0e-10)
end

# -- Can the candidate alone predict the divergence? ---------------------------

"""Columns of the divergence audit's CSV."""
const RELIABILITY_AUDIT_COLUMNS = (:fixture, :design, :seed, :noise, :library,
    :variant, :success, :support_f1, :extras, :n_terms, :denominator_min_samples,
    :denominator_min_box, :denominator_min_extended, :sign_change_box,
    :sign_change_extended, :negative_rate_box, :visited_min, :data_residual,
    :holdout_residual, :diverged, :train_time_s)

"""How far past the observed range the widened grid of the audit reaches."""
const RELIABILITY_AUDIT_MARGIN = 0.5

"""
A full factorial grid of `n` points per coordinate between `lo` and `hi`,
with `n` reduced until the grid fits in `max_points` columns.
"""
function _reliability_grid(lo, hi, n_grid::Integer, max_points::Integer)
    dimensions = length(lo)
    n = n_grid
    while dimensions > 0 && n^dimensions > max_points && n > 3
        n -= 1
    end
    total = dimensions == 0 ? 0 : n^dimensions
    grid = Matrix{Float64}(undef, dimensions, total)
    for column in 1:total
        rest = column - 1
        for i in 1:dimensions
            index = rest % n
            rest = rest ÷ n
            grid[i, column] = n == 1 || hi[i] == lo[i] ? lo[i] :
                              lo[i] + index * (hi[i] - lo[i]) / (n - 1)
        end
    end
    return grid
end

"""
The discovered denominator away from the samples it was fitted on. `X` holds
the sample coordinates of the state rows the candidate indexes. The report
covers the samples themselves, a dense grid over their bounding box, and the
same box widened by `margin` of its width on each side (never below zero,
which no concentration reaches). Everything here is computable from the
candidate before anything is resimulated, which is what the divergence
question asks of it.
"""
function reliability_denominator_report(candidate::ImplicitCandidate, X;
        margin::Real = RELIABILITY_AUDIT_MARGIN, n_grid::Integer = 21,
        max_points::Integer = 400_000)
    spec = candidate.specification
    numerator = candidate.numerator_coefficients
    denominator = candidate.denominator_coefficients
    _, at_samples = _evaluate_candidate(spec, numerator, denominator, X)
    lo = vec(minimum(X; dims = 2))
    hi = vec(maximum(X; dims = 2))
    width = hi .- lo
    box = _reliability_grid(lo, hi, n_grid, max_points)
    wide = _reliability_grid(max.(0.0, lo .- margin .* width), hi .+ margin .* width,
        n_grid, max_points)
    rate_box, denominator_box = _evaluate_candidate(spec, numerator, denominator, box)
    _, denominator_wide = _evaluate_candidate(spec, numerator, denominator, wide)
    return (;
        min_samples = minimum(at_samples),
        min_box = minimum(denominator_box),
        min_extended = minimum(denominator_wide),
        sign_change_box = minimum(denominator_box) < 0 < maximum(denominator_box),
        sign_change_extended = minimum(denominator_wide) < 0 < maximum(denominator_wide),
        negative_rate_box = any(<(0), rate_box))
end

"""
The smallest denominator value the resimulation actually visits, over the
trajectory the solver produced before it stopped. `NaN` when the solve
produced nothing. This is a post-hoc number, not part of the check.
"""
function _reliability_visited_minimum(model, p, term, candidate, rate_fn, rows,
        experiments)
    spec = candidate.specification
    smallest = Inf
    for experiment in experiments
        rhs = _library_study_hybrid_rhs(model, p, term, rate_fn, rows)
        times = experiment.times
        solution = try
            solve(SciMLBase.ODEProblem(rhs, experiment.u0,
                    (first(times), last(times))), Tsit5(); sensealg = nothing)
        catch
            nothing
        end
        solution === nothing && continue
        states = Array(solution)
        size(states, 2) == 0 && continue
        visited = Matrix(states[rows, :])
        all(isfinite, visited) || (visited = visited[:, vec(all(isfinite, visited; dims = 1))])
        size(visited, 2) == 0 && continue
        _, values = _evaluate_candidate(spec, candidate.numerator_coefficients,
            candidate.denominator_coefficients, visited)
        smallest = min(smallest, minimum(values))
    end
    return isfinite(smallest) ? smallest : NaN
end

"""
`:divergence`, on the runs that actually diverged. Retrains the library
comparison cells of `seeds` and `noise_levels` on `design` and records, for
every library and variant, both what the resimulation does and what the
candidate alone says about its denominator. The stored rows of the 0.11 to
0.17 studies name the cells worth auditing; reproducing their residuals is
how this path is checked.
"""
function reliability_divergence_audit(; seeds = RELIABILITY_SEEDS,
        noise_levels = (0.05,), fixture::Symbol = :four_state,
        design::Symbol = :constant, libraries = LIBRARY_STUDY_LIBRARIES,
        variants = (:study,), kind::Symbol = :protocol,
        margin::Real = RELIABILITY_AUDIT_MARGIN, on_row = nothing)
    rows = NamedTuple[]
    for seed in seeds, noise_σ in noise_levels
        trained = _library_study_train(fixture, seed, noise_σ, kind,
            fit_unknown_destruction, LIBRARY_STUDY_HOLDOUT_ICS, nothing, design)
        for variant in variants, library in libraries
            if variant === :study && trained.study_discoveries !== nothing
                discovery = trained.study_discoveries[library]
                discovery_rows = trained.all_rows
            else
                discovery, discovery_rows = _library_study_discover(
                    variant, library, trained.X, trained.D, trained.times,
                    trained.networks, trained.parent_rows, trained.all_rows,
                    trained.budget)
            end
            truth = _library_study_truth(trained.truth, discovery_rows)
            scores = library_study_scores(discovery, truth;
                names = trained.names[discovery_rows])
            candidate = scores.candidate
            report = candidate isa ImplicitCandidate ?
                     reliability_denominator_report(candidate,
                Matrix(trained.X[discovery_rows, :]); margin) :
                     (; min_samples = NaN, min_box = NaN, min_extended = NaN,
                sign_change_box = false, sign_change_extended = false,
                negative_rate_box = false)
            data_residual = NaN
            holdout_residual = NaN
            visited_min = NaN
            if candidate !== nothing
                rate_fn = equation_to_function(candidate)
                data_residual = _library_study_residual(trained.model, trained.params,
                    trained.term, rate_fn, discovery_rows,
                    trained.train_set.experiments[1])
                holdout_residual = mean(
                    _library_study_residual(trained.model, trained.params,
                        trained.term, rate_fn, discovery_rows, experiment)
                for experiment in trained.holdout_set.experiments)
                candidate isa ImplicitCandidate && (visited_min =
                    _reliability_visited_minimum(trained.model, trained.params,
                        trained.term, candidate, rate_fn, discovery_rows,
                        vcat(collect(trained.train_set.experiments),
                            collect(trained.holdout_set.experiments))))
            end
            n_terms = candidate isa ImplicitCandidate ?
                      count(!=(0.0), candidate.numerator_coefficients) +
                      count(!=(0.0), candidate.denominator_coefficients) : 0
            row = (; fixture = String(fixture), design = String(design),
                seed = Int(seed), noise = Float64(noise_σ),
                library = String(library), variant = String(variant),
                success = scores.success, support_f1 = scores.support_f1,
                extras = join(scores.extras, ";"), n_terms,
                denominator_min_samples = report.min_samples,
                denominator_min_box = report.min_box,
                denominator_min_extended = report.min_extended,
                sign_change_box = report.sign_change_box,
                sign_change_extended = report.sign_change_extended,
                negative_rate_box = report.negative_rate_box,
                visited_min,
                data_residual, holdout_residual,
                diverged = candidate !== nothing &&
                           !(isfinite(data_residual) && isfinite(holdout_residual)),
                train_time_s = round(trained.train_time; digits = 1))
            push!(rows, row)
            on_row === nothing || on_row(row)
        end
    end
    return rows
end

function reliability_audit_csv_line(row)
    values = map(RELIABILITY_AUDIT_COLUMNS) do column
        value = getproperty(row, column)
        value isa AbstractString || value isa Symbol ? replace(string(value), "," => ";") :
        value isa Bool ? string(value) :
        value isa Integer ? string(value) : string(Float64(value))
    end
    return join(values, ",")
end

function append_reliability_audit_row(path::AbstractString, row)
    mkpath(dirname(path))
    fresh = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        fresh && println(io, join(string.(RELIABILITY_AUDIT_COLUMNS), ","))
        println(io, reliability_audit_csv_line(row))
    end
    return path
end

function read_reliability_audit_csv(path::AbstractString)
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
            column in (:fixture, :design, :library, :variant, :extras) ? String(text) :
            column in (:seed, :n_terms) ? parse(Int, text) :
            column in (:success, :sign_change_box, :sign_change_extended,
                :negative_rate_box, :diverged) ? (text == "true") :
            parse(Float64, text)
        end
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end
    return rows
end

"""
Does a candidate-only quantity separate the runs that diverged from the runs
that did not? Reports, for each of the three denominator minima and the two
sign-change flags, how many diverging runs it would flag and how many runs
that were fine it would flag with them.
"""
function reliability_audit_summary(rows)
    isempty(rows) && return NamedTuple[]
    # Three outcomes, not two: a run with no candidate at all never reached
    # the resimulation, so it is neither a divergence nor a run that was fine.
    diverged = [row for row in rows if row.diverged]
    fine = [row for row in rows if row.success && !row.diverged]
    checks = NamedTuple[]
    for (name, flag) in (
        ("denominator below 0.1 on the sample box",
            row -> isfinite(row.denominator_min_box) && row.denominator_min_box < 0.1),
        ("denominator below 0.5 on the sample box",
            row -> isfinite(row.denominator_min_box) && row.denominator_min_box < 0.5),
        ("denominator below 0.1 on the widened box",
            row -> isfinite(row.denominator_min_extended) &&
                   row.denominator_min_extended < 0.1),
        ("sign change on the sample box", row -> row.sign_change_box),
        ("sign change on the widened box", row -> row.sign_change_extended),
        ("negative rate on the sample box", row -> row.negative_rate_box),
        ("no candidate at all", row -> !row.success))
        caught = count(flag, diverged)
        false_alarms = count(flag, fine)
        push!(checks, (; name, caught, of = length(diverged), false_alarms,
            of_fine = length(fine)))
    end
    pushfirst!(checks, (; name = "runs with no candidate (counted in neither)",
        caught = count(row -> !row.success, rows), of = length(rows),
        false_alarms = 0, of_fine = length(fine)))
    return checks
end

# -- CSV and summaries ---------------------------------------------------------

function reliability_csv_line(row)
    values = map(RELIABILITY_COLUMNS) do column
        value = getproperty(row, column)
        # commas would break the row, so text fields carry none
        value isa AbstractString || value isa Symbol ? replace(string(value), "," => ";") :
        value isa Bool ? string(value) :
        value isa Integer ? string(value) : string(Float64(value))
    end
    return join(values, ",")
end

function append_reliability_row(path::AbstractString, row)
    mkpath(dirname(path))
    fresh = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        fresh && println(io, join(string.(RELIABILITY_COLUMNS), ","))
        println(io, reliability_csv_line(row))
    end
    return path
end

function read_reliability_csv(path::AbstractString)
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
            column in (:fixture, :experiment, :setting, :retcode, :support) ? String(text) :
            column in (:seed, :init_seed, :n_terms) ? parse(Int, text) :
            column === :diverged ? (text == "true") :
            parse(Float64, text)
        end
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end
    return rows
end

function _reliability_spread(values)
    isempty(values) ? (; median = NaN, iqr = (NaN, NaN),
        min = NaN, max = NaN) :
    (; median = median(values),
        iqr = (quantile(values, 0.25), quantile(values, 0.75)),
        min = minimum(values), max = maximum(values))
end

"""The initialisation spread per seed, and the number of distinct supports."""
function reliability_initialisation_summary(rows)
    out = NamedTuple[]
    for seed in sort(unique(r.seed for r in rows if r.experiment == "initialisation"))
        group = [r for r in rows if r.experiment == "initialisation" && r.seed == seed]
        isempty(group) && continue
        supports = [r.support for r in group]
        counts = Dict{String, Int}()
        for support in supports
            counts[support] = get(counts, support, 0) + 1
        end
        most = maximum(values(counts))
        push!(out,
            (; seed, n = length(group),
                loss = _reliability_spread([r.final_loss for r in group]),
                rate = _reliability_spread([r.nn_rate_rmse for r in group]),
                f1 = _reliability_spread([r.support_f1 for r in group]),
                distinct_supports = length(counts), most_common = most,
                confirms = most ≥ 9))
    end
    return out
end

"""Warm-up length against the result, per observation setting."""
function reliability_warmup_summary(rows)
    out = NamedTuple[]
    settings = sort(unique(r.setting for r in rows if r.experiment == "warmup"))
    for setting in settings
        group = [r for r in rows if r.experiment == "warmup" && r.setting == setting]
        isempty(group) && continue
        push!(out,
            (; setting, n = length(group),
                loss = _reliability_spread([r.final_loss for r in group]),
                rate = _reliability_spread([r.nn_rate_rmse for r in group]),
                median_f1 = median([r.support_f1 for r in group]),
                n_diverged = count(r -> r.diverged, group),
                n_nonfinite_loss = count(r -> !isfinite(r.final_loss), group)))
    end
    return out
end

"""
Divergence against the denominator minimum: whether the candidate alone
predicts that the hybrid model cannot be resimulated.
"""
function reliability_divergence_summary(rows)
    group = [r for r in rows if isfinite(r.denominator_min)]
    diverged = [r for r in group if r.diverged]
    fine = [r for r in group if !r.diverged]
    best = (; threshold = NaN, caught = 0, false_alarms = 0)
    for threshold in sort(unique([r.denominator_min for r in group]))
        caught = count(r -> r.denominator_min ≤ threshold, diverged)
        alarms = count(r -> r.denominator_min ≤ threshold, fine)
        if alarms == 0 && caught > best.caught
            best = (; threshold, caught, false_alarms = alarms)
        end
    end
    return (; n = length(group), n_diverged = length(diverged),
        diverged_denominator = _reliability_spread([r.denominator_min for r in diverged]),
        fine_denominator = _reliability_spread([r.denominator_min for r in fine]),
        best_threshold = best,
        separates = length(diverged) > 0 && best.caught ≥ 0.8 * length(diverged))
end

function format_reliability_summary(rows)
    isempty(rows) && return "no rows"
    io = IOBuffer()
    initialisation = reliability_initialisation_summary(rows)
    if !isempty(initialisation)
        println(io,
            "--- initialisation spread (same data, different random initialisations) ---")
        println(io,
            "| seed | runs | final loss median [IQR] | learned-rate error median [IQR] | distinct supports | most common |")
        println(io, "|---|---|---|---|---|---|")
        for row in initialisation
            println(io, "| $(row.seed) | $(row.n) | ",
                round(row.loss.median; sigdigits = 3), " [",
                round(row.loss.iqr[1]; sigdigits = 3), ", ",
                round(row.loss.iqr[2]; sigdigits = 3), "] | ",
                round(row.rate.median; digits = 3), " [",
                round(row.rate.iqr[1]; digits = 3), ", ",
                round(row.rate.iqr[2]; digits = 3), "] | ",
                row.distinct_supports, " | ", row.most_common, " of ", row.n, " |")
        end
        println(io,
            "pre-registered confirmation (one support in at least 9 of 10, every seed): ",
            count(r -> r.confirms, initialisation), " of ", length(initialisation), " seeds")
    end
    warmup = reliability_warmup_summary(rows)
    if !isempty(warmup)
        println(io, "\n--- warm-up length ---")
        println(io,
            "| setting | runs | final loss median | learned-rate error median [IQR] | median F1 | non-finite losses | diverged |")
        println(io, "|---|---|---|---|---|---|---|")
        for row in warmup
            println(io, "| $(row.setting) | $(row.n) | ",
                round(row.loss.median; sigdigits = 3), " | ",
                round(row.rate.median; digits = 3), " [",
                round(row.rate.iqr[1]; digits = 3), ", ",
                round(row.rate.iqr[2]; digits = 3), "] | ",
                round(row.median_f1; digits = 3), " | ", row.n_nonfinite_loss,
                " | ", row.n_diverged, " |")
        end
    end
    divergence = [r for r in rows if r.experiment == "divergence"]
    if !isempty(divergence)
        summary = reliability_divergence_summary(divergence)
        println(io, "\n--- divergence and the denominator minimum ---")
        println(io, "runs $(summary.n), diverged $(summary.n_diverged)")
        println(io, "denominator minimum, diverged: median ",
            round(summary.diverged_denominator.median; sigdigits = 3),
            "  range [", round(summary.diverged_denominator.min; sigdigits = 3), ", ",
            round(summary.diverged_denominator.max; sigdigits = 3), "]")
        println(io, "denominator minimum, resimulated: median ",
            round(summary.fine_denominator.median; sigdigits = 3),
            "  range [", round(summary.fine_denominator.min; sigdigits = 3), ", ",
            round(summary.fine_denominator.max; sigdigits = 3), "]")
        println(io, "best threshold with no false alarm: ",
            summary.best_threshold.threshold, " catching ",
            summary.best_threshold.caught, " of ", summary.n_diverged)
    end
    return String(take!(io))
end
