###############################################################################
# Training reuse: compiled model, remake session, locked sensealg (not exported).
#
# train_experiments used to drop the Adam optimiser state after a first-IC
# warmup and, on the Augmented-Lagrangian path, called predict_ude without
# the compiled model (recompile per IC). This file owns the session that
# holds one UDEModel, remakes SciMLBase.ODEProblem across ICs, and locks
# the adjoint already chosen by recommend_sensealg.
#
# Does not drop protocol ICs, points, or seeds. Does not grow exports.
###############################################################################

"""Source strings that prove the training reuse path stays wired."""

const TRAINING_REUSE_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

function training_jl_source_path()
    joinpath(pkgdir(HybridKinetics), "src", "Training.jl")
end

# -- compile_network counter (state lives in MechanismCompiler.jl) ------------

"""
    with_compile_network_counter(f)

Run `f(counter)` while `compile_network` increments `counter`. Nested
calls restore the previous counter. Used to lock the training path:
a compiled `UDEModel` must not compile again per IC.
"""
function with_compile_network_counter(f)
    counter = Ref(0)
    previous = COMPILE_NETWORK_COUNTER[]
    COMPILE_NETWORK_COUNTER[] = counter
    try
        return f(counter)
    finally
        COMPILE_NETWORK_COUNTER[] = previous
    end
end

function compile_network_call_count()
    counter = COMPILE_NETWORK_COUNTER[]
    return counter === nothing ? 0 : counter[]
end

# -- Sensealg lock ------------------------------------------------------------

function training_sensealg_kind(sensealg)
    sensealg === nothing && return :none
    sensealg isa InterpolatingAdjoint && return :interpolating
    sensealg isa BacksolveAdjoint && return :backsolve
    return :other
end

function training_sensealg_kind(solver::SolverConfig)
    return training_sensealg_kind(solver.sensealg)
end

function neural_training_requires_interpolating(model::UDEModel)
    return model.n_neural > 0
end

"""
    lock_training_solver(model, solver) -> SolverConfig

Return a solver config whose adjoint matches `recommend_sensealg` for
`solver.ad_policy`. Neural holes keep `InterpolatingAdjoint`. A
`ProductionAD` forward pass with `sensealg === nothing` is left in place
(in-place RHS; no adjoint).
"""
function lock_training_solver(model::UDEModel, solver::SolverConfig)
    if solver.ad_policy isa ProductionAD && solver.sensealg === nothing
        return solver
    end
    sa = locked_training_sensealg(model, solver.ad_policy, 100)
    return SolverConfig(
        solver.algorithm, sa, solver.abstol, solver.reltol,
        solver.maxiters, solver.ad_policy)
end

function lock_training_solver(model::UDEModel; ad_policy::AbstractADPolicy = ZygoteAD(),
        n_observations::Int = 100)
    return default_solver_config(model; ad_policy = ad_policy,
        n_observations = n_observations)
end

function lock_training_config(model::UDEModel, config::TrainingConfig)
    solver = lock_training_solver(model, config.solver)
    return TrainingConfig(
        config.adam_iterations, config.adam_learning_rate, config.bfgs_iterations,
        config.gradient_clip, config.log_every, config.constraint, solver,
        config.horizon_schedule, config.frozen_phys, config.restarts)
end

"""
    training_sensealg_is_locked(model, solver) -> Bool

Consistency check. Neural models must not use `BacksolveAdjoint`. The
recommended kind from `recommend_sensealg` must match the stored
`sensealg`, except the ProductionAD in-place forward (`sensealg === nothing`).
"""
function training_sensealg_is_locked(model::UDEModel, solver::SolverConfig)
    if solver.ad_policy isa ProductionAD && solver.sensealg === nothing
        return true
    end
    rec = recommend_sensealg(model; policy = solver.ad_policy)
    kind = training_sensealg_kind(solver)
    rec_kind = training_sensealg_kind(rec.sensealg)
    neural_training_requires_interpolating(model) && kind === :backsolve &&
        return false
    return kind === rec_kind
end

function training_sensealg_is_locked(model::UDEModel, config::TrainingConfig)
    return training_sensealg_is_locked(model, config.solver)
end

function assert_training_sensealg(model::UDEModel, solver::SolverConfig)
    training_sensealg_is_locked(model, solver) || throw(ErrorException(
        "training sensealg $(training_sensealg_kind(solver)) is not locked to recommend_sensealg for this model"))
    return solver
end

# -- Solve session ------------------------------------------------------------

"""
    TrainingSolveSession

One compiled `UDEModel`, one template `ODEProblem`, and the locked
solver. `predict_ude_session` remakes `u0` / `tspan` / `p` and does not
call `compile_network`.
"""
mutable struct TrainingSolveSession{M, P, C, S}
    model::M
    template::P
    cache::C
    solver::S
    inplace::Bool
    remake_count::Int
    predict_count::Int
    sensealg_kind::Symbol
end

function training_solve_session(model::UDEModel, u0, tspan, p, solver::SolverConfig)
    assert_training_sensealg(model, solver)
    inplace = _forward_inplace(solver)
    template = SciMLBase.ODEProblem(model, u0, tspan, p; inplace = inplace)
    return TrainingSolveSession(
        model, template, nothing, solver, inplace, 0, 0,
        training_sensealg_kind(solver))
end

function training_solve_session(model::UDEModel, u0, tspan, p;
        solver::SolverConfig = lock_training_solver(model, SolverConfig()),
        cache::Union{Nothing, UDEModelCache} = nothing)
    cache === nothing &&
        return training_solve_session(model, u0, tspan, p, solver)
    assert_training_sensealg(model, solver)
    inplace = _forward_inplace(solver)
    local_cache = cache
    if inplace && local_cache === nothing
        local_cache = allocate_cache(model, Float64)
    end
    template = SciMLBase.ODEProblem(
        model, u0, tspan, p; inplace = inplace, cache = local_cache)
    return TrainingSolveSession(
        model,
        template,
        local_cache,
        solver,
        inplace,
        0,
        0,
        training_sensealg_kind(solver))
end

function training_solve_session(model::UDEModel, experiment::Experiment, p;
        kwargs...)
    tspan = (first(experiment.times), last(experiment.times))
    return training_solve_session(
        model, experiment.u0, tspan, p; kwargs...)
end

function training_solve_session(model::UDEModel, set::ExperimentSet, p; kwargs...)
    return training_solve_session(model, first(set.experiments), p; kwargs...)
end

"""
    predict_ude_session(session, p, u0, tspan, saveat)

Integrate by `SciMLBase.remake` of the session template. Does not compile.
"""
function predict_ude_session(session::TrainingSolveSession, p, u0, tspan, saveat)
    session.remake_count += 1
    session.predict_count += 1
    remade = SciMLBase.remake(session.template; u0 = u0, tspan = tspan, p = p)
    sol = solve(
        remade, session.solver.algorithm;
        saveat = saveat,
        abstol = session.solver.abstol,
        reltol = session.solver.reltol,
        maxiters = session.solver.maxiters,
        sensealg = session.solver.sensealg,
        dense = false,
        save_everystep = false)
    prediction = Array(sol)
    ignore_derivatives() do
        _validate_solution(sol, prediction, saveat, tspan)
    end
    return prediction
end

function session_predicts_without_compile(session::TrainingSolveSession, p, u0,
        tspan, saveat)
    n = with_compile_network_counter() do counter
        predict_ude_session(session, p, u0, tspan, saveat)
        counter[]
    end
    return n == 0
end

function resolve_training_model(model::Union{Nothing, UDEModel}, nn, st,
        network::BiologicalNetwork)
    model === nothing || return model
    return compile_network(network, nn, st)
end

# -- Warmup -------------------------------------------------------------------

"""
    warmup_first_experiment(p_init, set, model; config, ...)

Train on the first IC only (`bfgs_iterations = 0`). Returns
`(result, optimizer_state, session)`. The Optimisers state is the object
`train_experiments` must reuse.
"""
function warmup_first_experiment(p_init, set::ExperimentSet, model::UDEModel;
        config::TrainingConfig = TrainingConfig(),
        verbose::Bool = false,
        seed::Integer = 0)
    isempty(set.experiments) &&
        throw(ArgumentError("warmup_first_experiment needs a non-empty ExperimentSet"))
    locked = lock_training_config(model, config)
    first_exp = first(set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    session = training_solve_session(
        model, first_exp, p_init; solver = locked.solver)
    warm_config = TrainingConfig(locked;
        bfgs_iterations = 0,
        adam_iterations = max(1, locked.adam_iterations))
    result = train_ude(
        p_init, first_exp.observations, first_exp.times, first_exp.u0,
        tspan, model;
        config = warm_config,
        verbose = verbose,
        seed = seed,
        session = session,
        mask = first_exp.mask)
    state = hasproperty(result.diagnostics, :optimizer_state) ?
            result.diagnostics.optimizer_state : nothing
    return (;
        result,
        params = result.params,
        optimizer_state = state,
        session,
        n_ics = 1,
        locked_config = warm_config)
end

function optimizer_state_from_result(result::TrainingResult)
    hasproperty(result.diagnostics, :optimizer_state) || return nothing
    return result.diagnostics.optimizer_state
end

# -- Joint warmup + multi-IC train --------------------------------------------

"""
`restarts` fits from `restarts` different random initialisations of the neural
term, keeping the one that reaches the lowest final loss. The physical guess
and every other setting stay as the caller left them; only the network is drawn
again, from `seed` so that the set of restarts is reproducible. The losses of
every restart are carried in the returned result's metadata under
`restart_losses`, in the order they were run.

0.19 measured this against the alternative of accepting the first fit: a lower
loss does come with a more accurate learned rate, but on the reference protocol
it did not change the discovered support. Default is 1, which is the single fit
the package has always done.
"""
function _train_with_restarts(p_init, set::ExperimentSet, model::UDEModel, locked,
        execution, verbose, seed, warmup)
    single = TrainingConfig(locked; restarts = 1)
    losses = Float64[]
    best = nothing
    for k in 1:(locked.restarts)
        start = k == 1 ? p_init : _reinitialise_network(p_init, model, seed + k)
        attempt = train_experiments_with_warmup(start, set, model;
            config = single, execution = execution, verbose = verbose,
            seed = seed, warmup = warmup)
        push!(losses, Float64(attempt.final_loss))
        if best === nothing || attempt.final_loss < best.final_loss
            best = attempt
        end
    end
    metadata = RunMetadata(; seed = seed,
        config = (; base = best.metadata.config, restart_losses = copy(losses)))
    return TrainingResult(best.params, best.history, best.initial_loss,
        best.final_loss, metadata, best.diagnostics, best.converged, best.retcode)
end

"""`p_init` with the neural block drawn again from `seed`, physical guess kept."""
function _reinitialise_network(p_init, model::UDEModel, seed::Integer)
    parameters, _ = Lux.setup(MersenneTwister(seed), model.nn)
    fresh = copy(p_init)
    data = ComponentArrays.getdata(
        ComponentArrays.ComponentVector(_float64_param_tree(parameters)))
    length(data) == length(ComponentArrays.getdata(fresh.nn)) || throw(ErrorException(
        "a fresh draw of the network has $(length(data)) parameters against " *
        "$(length(ComponentArrays.getdata(fresh.nn))) in the vector being restarted"))
    ComponentArrays.getdata(fresh.nn) .= data
    return fresh
end

"""
    train_experiments_with_warmup(p_init, set, model; config, ...)

Warmup on IC 1, then `train_experiments` with the same compiled model,
locked solver, and Optimisers state. This is the reference-protocol training
entry used by `_train_unknown_edge`.
"""
function train_experiments_with_warmup(p_init, set::ExperimentSet, model::UDEModel;
        config::TrainingConfig = TrainingConfig(),
        execution::ExecutionConfig = ExecutionConfig(),
        verbose::Bool = false,
        seed::Integer = 0,
        warmup::Bool = true)
    locked = lock_training_config(model, config)
    locked.restarts > 1 && return _train_with_restarts(
        p_init, set, model, locked, execution, verbose, seed, warmup)
    session = training_solve_session(model, set, p_init; solver = locked.solver)
    if !warmup || length(set) == 1
        return train_experiments(
            p_init, set, model;
            config = locked, execution = execution, verbose = verbose,
            seed = seed, session = session)
    end
    warm = warmup_first_experiment(
        p_init, set, model; config = locked, verbose = verbose, seed = seed)
    return train_experiments(
        warm.params, set, model;
        config = locked, execution = execution, verbose = verbose,
        seed = seed,
        optimizer_state = warm.optimizer_state,
        session = session)
end

function reference_protocol_training_config(;
        adam_iterations = REFERENCE_PROTOCOL.adam_iterations,
        bfgs_iterations = REFERENCE_PROTOCOL.bfgs_iterations,
        frozen_phys::Vector{Symbol} = Symbol[],
        model::Union{Nothing, UDEModel} = nothing)
    base = TrainingConfig(
        adam_iterations = adam_iterations,
        bfgs_iterations = bfgs_iterations,
        horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0]),
        log_every = 10^6,
        frozen_phys = frozen_phys)
    model === nothing && return base
    return lock_training_config(model, base)
end

# -- Agreement / reports ------------------------------------------------------

function training_session_remake_agreement(model::UDEModel, params, u0;
        tspan = (0.0, 1.0), n_points::Int = 8,
        solver::SolverConfig = lock_training_solver(model, SolverConfig()))
    times = collect(range(first(tspan), last(tspan); length = n_points))
    direct = predict_ude(
        params, u0, tspan, times, model; solver_config = solver)
    session = training_solve_session(
        model, u0, tspan, params; solver = solver)
    remade = predict_ude_session(session, params, u0, tspan, times)
    remade2 = predict_ude_session(session, params, u0, tspan, times)
    compiled = with_compile_network_counter() do counter
        predict_ude_session(session, params, u0, tspan, times)
        counter[]
    end
    return (;
        direct,
        remade,
        remade2,
        remake_count = session.remake_count,
        predict_count = session.predict_count,
        sensealg_kind = session.sensealg_kind,
        matches = direct ≈ remade && remade ≈ remade2,
        no_compile = compiled == 0,
        holds = direct ≈ remade && remade ≈ remade2 && compiled == 0 &&
                session.remake_count ≥ 2)
end

function training_session_multi_ic_agreement(model::UDEModel, params, set::ExperimentSet;
        solver::SolverConfig = lock_training_solver(model, SolverConfig()))
    session = training_solve_session(model, set, params; solver = solver)
    n = with_compile_network_counter() do counter
        for experiment in set.experiments
            tspan = (first(experiment.times), last(experiment.times))
            pred = predict_ude_session(
                session, params, experiment.u0, tspan, experiment.times)
            direct = predict_ude(
                params, experiment.u0, tspan, experiment.times, model;
                solver_config = solver)
            pred ≈ direct || return (;
                counter = counter[], matches = false, holds = false)
        end
        return (;
            counter = counter[],
            remake_count = session.remake_count,
            matches = true,
            holds = counter[] == 0 &&
                    session.remake_count == length(set.experiments))
    end
    return n
end

function train_experiments_compile_report(p_init, set::ExperimentSet, model::UDEModel;
        config::TrainingConfig = TrainingConfig(
            adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
        constraint = nothing)
    cfg = constraint === nothing ? config :
          TrainingConfig(config; constraint = constraint)
    locked = lock_training_config(model, cfg)
    n = with_compile_network_counter() do counter
        train_experiments(
            p_init, set, model;
            config = locked, verbose = false)
        counter[]
    end
    n_nn = with_compile_network_counter() do counter
        train_experiments(
            p_init, set, model.nn, model.st;
            network = model.network,
            config = locked, verbose = false)
        counter[]
    end
    return (;
        with_model = n,
        with_nn_st = n_nn,
        n_ics = length(set),
        holds = n == 0 && n_nn == 1)
end

function warmup_state_reuse_report(p_init, set::ExperimentSet, model::UDEModel;
        adam_iterations::Int = 2)
    config = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations,
            bfgs_iterations = 0,
            log_every = 10^6))
    warm = warmup_first_experiment(
        p_init, set, model; config = config, verbose = false)
    reused = train_experiments(
        warm.params, set, model;
        config = config, verbose = false,
        optimizer_state = warm.optimizer_state)
    fresh = train_experiments(
        warm.params, set, model;
        config = config, verbose = false,
        optimizer_state = nothing)
    return (;
        warmup_has_state = warm.optimizer_state !== nothing,
        reused_history = length(reused.history),
        fresh_history = length(fresh.history),
        sensealg_locked = training_sensealg_is_locked(model, config),
        holds = warm.optimizer_state !== nothing &&
                training_sensealg_is_locked(model, config))
end

function al_constraint_passes_model_source()
    src = read(training_jl_source_path(), String)
    start = findfirst("function train_experiments(p_init, set::ExperimentSet, nn, st", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(
        "function train_experiments(p_init, set::ExperimentSet, model::UDEModel",
        rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("predict_ude(", body) &&
           occursin("model = resolved_model", body) &&
           occursin("session = local_session", body)
end

function train_experiments_accepts_optimizer_state_source()
    src = read(training_jl_source_path(), String)
    return occursin("optimizer_state = nothing", src) &&
           occursin("session = nothing", src) &&
           occursin("optimizer_state = current_optimizer_state", src)
end

function train_unknown_edge_reuses_warmup_source()
    path = joinpath(pkgdir(HybridKinetics), "src", "RecoveryPipeline.jl")
    src = read(path, String)
    start = findfirst("function fit_unknown_destruction", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("train_experiments_with_warmup", body) &&
           occursin("lock_training_config", body) &&
           !occursin("bfgs_iterations = 0,\n            horizon_schedule", body)
end

# -- Linear / neural fixtures for sensealg checks ----------------------------

function train_ude_optimizer_state_is_recorded(p_init, data, times, u0, tspan,
        model; adam_iterations::Int = 2)
    result = train_ude(
        p_init, data, times, u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6),
        verbose = false)
    return (;
        has_state = optimizer_state_from_result(result) !== nothing,
        history = length(result.history),
        holds = optimizer_state_from_result(result) !== nothing &&
                length(result.history) == adam_iterations)
end

function resume_training_compile_report(p_init, data, times, u0, tspan, model;
        checkpoint_dir)
    path = joinpath(checkpoint_dir, "reuse.ckpt")
    result = train_ude(
        p_init, data, times, u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6),
        verbose = false,
        checkpoint_path = path,
        checkpoint_every = 1,
        seed = 3)
    isfile(path) || return (; holds = false, compiles = typemax(Int))
    ckpt = load_checkpoint(path)
    n = with_compile_network_counter() do counter
        resume_training(
            ckpt, data, times, u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        schema = ckpt.schema_version,
        holds = n == 0)
end

# -- Generate / train_ude / mask / n_obs checks ------------------------------

"""Tight solver used only to compare a session remake against generate."""
function _generate_match_solver(model::UDEModel)
    base = lock_training_solver(model, SolverConfig())
    return SolverConfig(
        algorithm = Tsit5(),
        ad_policy = base.ad_policy,
        sensealg = base.sensealg,
        abstol = 1e-9,
        reltol = 1e-9)
end

"""
    training_session_matches_generate(model, params, u0; tspan, n_points)

`predict_ude_session` versus `generate_from_compiled_model` at noise 0.
Forward trajectories must agree; `compile_network` must stay at zero.
"""
function training_session_matches_generate(model::UDEModel, params, u0;
        tspan = (0.0, 0.8), n_points::Int = 8,
        rng::AbstractRNG = MersenneTwister(0))
    solver = _generate_match_solver(model)
    times, clean, _, used = generate_from_compiled_model(
        model, params, rng;
        u0 = Float64.(u0), tspan = tspan, n_points = n_points, noise_σ = 0.0)
    session = training_solve_session(
        model, Float64.(u0), tspan, used; solver = solver)
    remade = predict_ude_session(session, used, Float64.(u0), tspan, times)
    direct = predict_ude(
        used, Float64.(u0), tspan, times, model; solver_config = solver)
    compiled = with_compile_network_counter() do counter
        predict_ude_session(session, used, Float64.(u0), tspan, times)
        counter[]
    end
    return (;
        times,
        clean,
        remade,
        direct,
        remake_count = session.remake_count,
        matches_generate = remade ≈ clean,
        matches_direct = remade ≈ direct,
        no_compile = compiled == 0,
        holds = remade ≈ clean && remade ≈ direct && compiled == 0)
end

function train_ude_compile_report(p_init, data, times, u0, tspan, model;
        adam_iterations::Int = 1)
    locked = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6))
    n = with_compile_network_counter() do counter
        train_ude(
            p_init, data, times, u0, tspan, model;
            config = locked, verbose = false)
        counter[]
    end
    n_nn = with_compile_network_counter() do counter
        train_ude(
            p_init, data, times, u0, tspan, model.nn, model.st;
            network = model.network, config = locked, verbose = false)
        counter[]
    end
    return (;
        with_model = n,
        with_nn_st = n_nn,
        holds = n == 0 && n_nn == 1)
end

function frozen_phys_warmup_report(p_init, set::ExperimentSet, model::UDEModel;
        frozen::Vector{Symbol} = [:k_ba],
        adam_iterations::Int = 2)
    config = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations,
            bfgs_iterations = 0,
            log_every = 10^6,
            frozen_phys = frozen))
    names = Tuple(parameter_schema(model).phys_names)
    before = NamedTuple{names}(ntuple(
        i -> Float64(p_init.phys[i]), length(names)))
    warm = warmup_first_experiment(
        p_init, set, model; config = config, verbose = false)
    reused = train_experiments(
        warm.params, set, model;
        config = config, verbose = false,
        optimizer_state = warm.optimizer_state)
    after = NamedTuple{names}(ntuple(
        i -> Float64(reused.params.phys[i]), length(names)))
    frozen_held = all(name -> getfield(before, name) ≈ getfield(after, name),
        frozen)
    return (;
        frozen,
        before,
        after,
        frozen_held,
        warmup_has_state = warm.optimizer_state !== nothing,
        holds = frozen_held && warm.optimizer_state !== nothing &&
                training_sensealg_is_locked(model, config))
end

function masked_experiment_compile_report(p_init, set::ExperimentSet,
        model::UDEModel; hide_state::Int = 1)
    experiments = map(set.experiments) do experiment
        mask = copy(experiment.mask)
        mask[hide_state, :] .= false
        Experiment(
            experiment.name, experiment.times, experiment.observations,
            experiment.u0; mask = mask, metadata = experiment.metadata)
    end
    masked = ExperimentSet(experiments, set.state_names; units = set.units,
        metadata = set.metadata)
    locked = lock_training_config(
        model, TrainingConfig(
            adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6))
    n = with_compile_network_counter() do counter
        train_experiments(
            p_init, masked, model; config = locked, verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        hidden = hide_state,
        n_ics = length(masked),
        holds = n == 0)
end

function horizon_curriculum_session_report(p_init, data, times, u0, tspan,
        model; adam_iterations::Int = 1)
    curriculum = HorizonCurriculum(
        fractions = [0.5, 1.0], min_points = 3, minimum_fraction = 0.4)
    config = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6, horizon_schedule = curriculum))
    n = with_compile_network_counter() do counter
        train_ude(
            p_init, data, times, u0, tspan, model;
            config = config, verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        fractions = curriculum.fractions,
        holds = n == 0 && training_sensealg_is_locked(model, config))
end

function optimizer_state_roundtrip_report(p_init, set::ExperimentSet,
        model::UDEModel; adam_iterations::Int = 2)
    config = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6))
    warm = warmup_first_experiment(
        p_init, set, model; config = config, verbose = false)
    reused = train_experiments(
        warm.params, set, model;
        config = config, verbose = false,
        optimizer_state = warm.optimizer_state)
    from_warm = optimizer_state_from_result(warm.result)
    from_reused = optimizer_state_from_result(reused)
    return (;
        warmup_has_state = from_warm !== nothing,
        reused_has_state = from_reused !== nothing,
        same_type = typeof(from_warm) === typeof(from_reused),
        holds = from_warm !== nothing && from_reused !== nothing &&
                typeof(from_warm) === typeof(from_reused))
end

function resume_from_diagnostics_report(p_init, data, times, u0, tspan, model;
        adam_iterations::Int = 2)
    first = train_ude(
        p_init, data, times, u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6),
        verbose = false)
    state = optimizer_state_from_result(first)
    n = with_compile_network_counter() do counter
        train_ude(
            first.params, data, times, u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            optimizer_state = state, verbose = false)
        counter[]
    end
    return (;
        has_state = state !== nothing,
        compiles = n,
        holds = state !== nothing && n == 0)
end

function linear_training_fixture()
    rng = MersenneTwister(53)
    net = build_linear_test_network()
    model, p0 = build_ude_model(rng, net)
    truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    set = generate_experiment_set(
        MersenneTwister(53); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
        tspan = (0.0, 0.6), n_points = 6, noise_σ = 0.0,
        truth_params = truth)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    exp = first(set.experiments)
    tspan = (first(exp.times), last(exp.times))
    return (;
        net, model, p0, set, init, exp, tspan,
        data = exp.observations, times = exp.times, u0 = exp.u0)
end

# -- Docs / source locks ------------------------------------------------------
