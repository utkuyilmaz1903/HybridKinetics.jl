using OrdinaryDiffEq: ODEProblem, solve, Tsit5
using SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP

const PREDICT_UDE_OBSERVER = Ref{Any}(nothing)

function _note_predict_ude(p, u0, tspan, times, model, X)
    observer = PREDICT_UDE_OBSERVER[]
    observer === nothing && return nothing
    return Base.invokelatest(observer,
        (; params = p, u0 = u0, tspan = tspan, times = times, model = model,
            X = X))
end

function with_predict_ude_observer(f::Function, observer)
    previous = PREDICT_UDE_OBSERVER[]
    PREDICT_UDE_OBSERVER[] = observer
    try
        return f()
    finally
        PREDICT_UDE_OBSERVER[] = previous
    end
end

mutable struct LossDiagnostics
    mse::Float64
    constraint::Float64
    total::Float64
    primal_residual::Float64
    bfgs_attempted::Bool
    bfgs_success::Bool
    bfgs_retcode::Symbol
    bfgs_message::String
    gradient_norm_history::Vector{Float64}
    gradient_failure::Bool
    LossDiagnostics() = new(NaN, NaN, NaN, NaN, false, false, :none, "", Float64[], false)
end

@inline function _record!(d::LossDiagnostics, mse, constraint, total, residual)
    d.mse = Float64(mse)
    d.constraint = Float64(constraint)
    d.total = Float64(total)
    d.primal_residual = Float64(residual)
    return nothing
end

@inline function _record_bfgs!(d::LossDiagnostics, attempted, success, retcode, message)
    d.bfgs_attempted = attempted
    d.bfgs_success = success
    d.bfgs_retcode = retcode
    d.bfgs_message = message
    return nothing
end

"""Return true when the solver config requests an in-place forward pass."""
function _forward_inplace(solver_config::SolverConfig)
    solver_config.ad_policy isa ProductionAD || return false
    # Zygote-based adjoints require the out-of-place RHS during differentiation.
    return solver_config.sensealg === nothing
end

function predict_ude(p, u0, tspan, saveat, nn, st;
        model::Union{Nothing, UDEModel} = nothing,
        network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
        solver_config::SolverConfig = SolverConfig(),
        cache::Union{Nothing, UDEModelCache} = nothing,
        session = nothing)
    if session !== nothing
        return predict_ude_session(session, p, u0, tspan, saveat)
    end
    resolved = model === nothing ?
               ignore_derivatives(() -> compile_network(network, nn, st)) : model
    inplace = _forward_inplace(solver_config)
    prob = SciMLBase.ODEProblem(
        resolved, u0, tspan, p; inplace = inplace, cache = cache)
    sol = solve(
        prob, solver_config.algorithm;
        saveat = saveat,
        abstol = solver_config.abstol,
        reltol = solver_config.reltol,
        maxiters = solver_config.maxiters,
        sensealg = solver_config.sensealg,
        dense = false,
        save_everystep = false
    )
    prediction = Array(sol)
    ignore_derivatives() do
        _validate_solution(sol, prediction, saveat, tspan)
    end
    return prediction
end

@inline _smooth_violation(value, temperature) = temperature * softplus(-value / temperature)

function _validate_solution(solution, prediction, saveat, tspan)
    SciMLBase.successful_retcode(solution) ||
        throw(ErrorException("ODE solve failed with retcode $(solution.retcode)"))
    size(prediction, 2) == length(saveat) ||
        throw(ErrorException("ODE solve did not produce all requested samples"))
    isapprox(solution.t[end], tspan[2]; atol = 10eps(float(tspan[2])),
        rtol = 10eps(float(tspan[2]))) ||
        throw(ErrorException("ODE solve terminated before the final time"))
    all(isfinite, prediction) ||
        throw(ErrorException("ODE solve returned non-finite states"))
    return nothing
end

function _constraint_values(prediction, strategy::StructuralPositivity)
    return zeros(eltype(prediction), size(prediction, 1))
end

function _constraint_values(prediction, strategy::AugmentedLagrangianConfig)
    temperature = strategy.smoothness
    return map(axes(prediction, 1)) do state
        values = .-(@view prediction[state, :])
        maximum_value = maximum(values)
        maximum_value +
        temperature * log(mean(
            exp.((values .- maximum_value) ./ temperature)))
    end
end

function _masked_mse(prediction, data, mask)
    count(mask) > 0 || throw(ArgumentError("observation mask is empty"))
    residual = ifelse.(mask, prediction .- data, zero(eltype(prediction)))
    return sum(abs2, residual) / count(mask)
end

@inline _smooth_projection(value, temperature) = temperature * softplus(value / temperature)

function _augmented_term(constraints, dual, ρ, temperature)
    isempty(constraints) && return zero(eltype(constraints))
    shifted = dual .+ ρ .* constraints
    projected = _smooth_projection.(shifted, temperature)
    return sum((projected .^ 2 .- dual .^ 2) ./ (2ρ))
end

"""
    predict_ude(p, u0, tspan, saveat, model::UDEModel; solver_config, cache)

Integrate the compiled UDE and return the state trajectory at `saveat`.
"""
function predict_ude(p, u0, tspan, saveat, model::UDEModel;
        solver_config::SolverConfig = SolverConfig(),
        cache::Union{Nothing, UDEModelCache} = nothing,
        session = nothing)
    if session !== nothing
        return predict_ude_session(session, p, u0, tspan, saveat)
    end
    inplace = _forward_inplace(solver_config)
    prob = SciMLBase.ODEProblem(
        model, u0, tspan, p; inplace = inplace, cache = cache)
    sol = solve(
        prob, solver_config.algorithm;
        saveat = saveat,
        abstol = solver_config.abstol,
        reltol = solver_config.reltol,
        maxiters = solver_config.maxiters,
        sensealg = solver_config.sensealg,
        dense = false,
        save_everystep = false)
    prediction = Array(sol)
    ignore_derivatives() do
        _validate_solution(sol, prediction, saveat, tspan)
        _note_predict_ude(p, u0, tspan, saveat, model, prediction)
    end
    return prediction
end

function predict_ude(p, u0, tspan, saveat, model::UDEModel,
        solver_config::SolverConfig)
    inplace = _forward_inplace(solver_config)
    prob = _odeproblem(model, u0, tspan, p, inplace)
    sol = solve(
        prob, solver_config.algorithm;
        saveat = saveat,
        abstol = solver_config.abstol,
        reltol = solver_config.reltol,
        maxiters = solver_config.maxiters,
        sensealg = solver_config.sensealg,
        dense = false,
        save_everystep = false)
    prediction = Array(sol)
    ignore_derivatives() do
        _validate_solution(sol, prediction, saveat, tspan)
    end
    return prediction
end

function loss_mse(p, data, t_data, u0, tspan, nn, st;
        model::Union{Nothing, UDEModel} = nothing,
        network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
        constraint::AbstractConstraintStrategy =
        StructuralPositivity(),
        dual = zeros(eltype(p), size(data, 1)),
        ρ = one(eltype(p)),
        solver_config::SolverConfig = SolverConfig(),
        mask = trues(size(data)),
        diagnostics::Union{Nothing, LossDiagnostics} = nothing,
        session = nothing)
    prediction = predict_ude(p, u0, tspan, t_data, nn, st;
        model = model, network = network,
        solver_config = solver_config,
        session = session)
    size(prediction) == size(data) ||
        throw(ErrorException("ODE solve terminated before all observations"))
    all(isfinite, prediction) ||
        throw(ErrorException("ODE solve returned non-finite states"))
    mse = _masked_mse(prediction, data, mask)
    constraints = _constraint_values(prediction, constraint)
    constraint_loss = constraint isa AugmentedLagrangianConfig ?
                      _augmented_term(
        constraints, dual, ρ, constraint.smoothness) : zero(mse)
    total = mse + constraint_loss
    residual = isempty(constraints) ? zero(mse) :
               max(zero(mse), maximum(constraints))

    if diagnostics !== nothing
        ignore_derivatives() do
            _record!(diagnostics, mse, constraint_loss, total, residual)
        end
    end
    return total
end

function _training_retcode(converged::Bool, diagnostics::LossDiagnostics)
    converged && return Success
    diagnostics.gradient_failure && return GradientFailure
    diagnostics.bfgs_attempted && !diagnostics.bfgs_success && return BFGSFailure
    return NotConverged
end

struct LossMSECall{S}
    model::UDEModel
    constraint::StructuralPositivity
    dual::Vector{Float64}
    ρ::Float64
    solver::S
    diagnostics::LossDiagnostics
    data::Matrix{Float64}
    times::Vector{Float64}
    u0::Vector{Float64}
    tspan::Tuple{Float64, Float64}
end

function _loss_mse_default(p, data, t_data, u0, tspan, model::UDEModel,
        constraint::StructuralPositivity, dual::Vector{Float64}, ρ::Float64,
        solver_config::SolverConfig, diagnostics::LossDiagnostics)
    prediction = predict_ude(p, u0, tspan, t_data, model, solver_config)
    size(prediction) == size(data) ||
        throw(ErrorException("ODE solve terminated before all observations"))
    all(isfinite, prediction) ||
        throw(ErrorException("ODE solve returned non-finite states"))
    mse = Float64(_masked_mse(prediction, data, trues(size(data))))
    _record!(diagnostics, mse, 0.0, mse, 0.0)
    return mse
end

function (f::LossMSECall)(p, _)
    return _loss_mse_default(
        p, f.data, f.times, f.u0, f.tspan, f.model,
        f.constraint, f.dual, f.ρ, f.solver, f.diagnostics)
end

function _safe_initial_loss(full_loss::LossMSECall, p_init)
    try
        return full_loss(p_init, nothing)
    catch
        return 0.0
    end
end

struct TrainDiagnostics
    mse::Float64
    constraint::Float64
    primal_residual::Float64
    dual::Vector{Float64}
    ρ::Float64
    final_gradient_norm::Float64
    gradient_norm_history::Vector{Float64}
    bfgs::NamedTuple{(:attempted, :success, :retcode, :message),
        Tuple{Bool, Bool, Symbol, String}}
    optimizer_state::Any
end

function _train_diagnostics(diag::LossDiagnostics, dual, ρ, optimizer_state)
    hist = copy(diag.gradient_norm_history)
    return TrainDiagnostics(
        Float64(diag.mse), Float64(diag.constraint),
        Float64(diag.primal_residual), collect(Float64, dual), Float64(ρ),
        isempty(hist) ? NaN : Float64(last(hist)), hist,
        (attempted = diag.bfgs_attempted, success = diag.bfgs_success,
            retcode = diag.bfgs_retcode, message = String(diag.bfgs_message)),
        optimizer_state)
end

function _stage_config(config::TrainingConfig, stages::Int, final_stage::Bool)
    adam_iterations = max(1, cld(config.adam_iterations, stages))
    bfgs_iterations = final_stage ? config.bfgs_iterations : 0
    return TrainingConfig(
        adam_iterations, config.adam_learning_rate, bfgs_iterations,
        config.gradient_clip, config.log_every, config.constraint,
        config.solver, config.horizon_schedule, config.frozen_phys,
        config.restarts)
end

function _optimize_stage(p_init::P, loss_closure::LossMSECall, config::TrainingConfig,
        history::Vector{Float64}, diagnostics::LossDiagnostics,
        verbose::Bool, optimizer_state, checkpoint_hook) where {P}
    callback = function (_state, loss)
        push!(history, Float64(loss))
        if verbose && length(history) % config.log_every == 0
            println("       iter ", lpad(length(history), 4),
                " | total=", round(loss; sigdigits = 7),
                " | mse=", round(diagnostics.mse; sigdigits = 7),
                " | constraint=",
                round(diagnostics.constraint; sigdigits = 7),
                " | primal=",
                round(diagnostics.primal_residual; sigdigits = 5))
        end
        return false
    end
    optimizer = Optimisers.OptimiserChain(
        Optimisers.ClipGrad(config.gradient_clip),
        Optimisers.Adam(config.adam_learning_rate))
    params = p_init
    frozen_ref = p_init
    state = optimizer_state === nothing ?
            Optimisers.setup(optimizer, params) : optimizer_state
    for _ in 1:(config.adam_iterations)
        loss, gradients = Zygote.withgradient(
            value -> loss_closure(value, nothing), params)
        gradient = only(gradients)
        all(isfinite, gradient) ||
            begin
                diagnostics.gradient_failure = true
                throw(ErrorException("optimizer produced a non-finite gradient"))
            end
        gradient = _zero_frozen_phys_gradient(gradient, config.frozen_phys)
        push!(diagnostics.gradient_norm_history,
            Float64(sqrt(sum(abs2, gradient))))
        state, params = Optimisers.update(state, params, gradient)
        callback(nothing, loss)
        checkpoint_hook === nothing ||
            checkpoint_hook(params, state, length(history))
    end
    if config.bfgs_iterations > 0
        objective = Optimization.OptimizationFunction(
            loss_closure, Optimization.AutoZygote())
        refinement = Optimization.OptimizationProblem(objective, params)
        bfgs_result = try
            Optimization.solve(
                refinement, OptimizationOptimJL.BFGS();
                callback, maxiters = config.bfgs_iterations)
        catch error
            _record_bfgs!(diagnostics, true, false, :failure,
                sprint(showerror, error))
            @warn "BFGS refinement failed; retaining Adam result." exception=error
            nothing
        end
        if bfgs_result === nothing
            diagnostics.bfgs_attempted || _record_bfgs!(
                diagnostics, true, false, :failure, "BFGS solve failed")
        else
            params = _restore_frozen_phys(
                bfgs_result.u, frozen_ref, config.frozen_phys)
            _record_bfgs!(diagnostics, true, true, :success, "BFGS refinement completed")
        end
    end
    return (params, state)::Tuple{P, Any}
end

function _optimize_stage(p_init::P, loss_closure, config, history, diagnostics,
        verbose::Bool, optimizer_state, checkpoint_hook) where {P}
    callback = function (_state, loss)
        push!(history, Float64(loss))
        if verbose && length(history) % config.log_every == 0
            println("       iter ", lpad(length(history), 4),
                " | total=", round(loss; sigdigits = 7),
                " | mse=", round(diagnostics.mse; sigdigits = 7),
                " | constraint=",
                round(diagnostics.constraint; sigdigits = 7),
                " | primal=",
                round(diagnostics.primal_residual; sigdigits = 5))
        end
        return false
    end
    optimizer = Optimisers.OptimiserChain(
        Optimisers.ClipGrad(config.gradient_clip),
        Optimisers.Adam(config.adam_learning_rate))
    params = p_init
    frozen_ref = p_init
    state = optimizer_state === nothing ?
            Optimisers.setup(optimizer, params) : optimizer_state
    for _ in 1:(config.adam_iterations)
        loss, gradients = Zygote.withgradient(
            value -> loss_closure(value, nothing), params)
        gradient = only(gradients)
        all(isfinite, gradient) ||
            begin
                diagnostics.gradient_failure = true
                throw(ErrorException("optimizer produced a non-finite gradient"))
            end
        gradient = _zero_frozen_phys_gradient(gradient, config.frozen_phys)
        push!(diagnostics.gradient_norm_history,
            Float64(sqrt(sum(abs2, gradient))))
        state, params = Optimisers.update(state, params, gradient)
        callback(nothing, loss)
        checkpoint_hook === nothing ||
            checkpoint_hook(params, state, length(history))
    end
    if config.bfgs_iterations > 0
        objective = Optimization.OptimizationFunction(
            loss_closure, Optimization.AutoZygote())
        refinement = Optimization.OptimizationProblem(objective, params)
        bfgs_result = try
            Optimization.solve(
                refinement, OptimizationOptimJL.BFGS();
                callback, maxiters = config.bfgs_iterations)
        catch error
            _record_bfgs!(diagnostics, true, false, :failure,
                sprint(showerror, error))
            @warn "BFGS refinement failed; retaining Adam result." exception=error
            nothing
        end
        if bfgs_result === nothing
            diagnostics.bfgs_attempted || _record_bfgs!(
                diagnostics, true, false, :failure, "BFGS solve failed")
        else
            params = _restore_frozen_phys(
                bfgs_result.u, frozen_ref, config.frozen_phys)
            _record_bfgs!(diagnostics, true, true, :success, "BFGS refinement completed")
        end
    end
    return (params, state)::Tuple{P, Any}
end

function _zero_frozen_phys_gradient(gradient, frozen::AbstractVector{Symbol})
    isempty(frozen) && return gradient
    hasproperty(gradient, :phys) || return gradient
    phys = gradient.phys
    updates = Pair{Symbol, Any}[]
    changed = false
    for name in propertynames(phys)
        value = getproperty(phys, name)
        if name in frozen
            push!(updates, name => zero(value))
            changed = true
        else
            push!(updates, name => value)
        end
    end
    changed || return gradient
    return ComponentVector(phys = NamedTuple(updates), nn = gradient.nn)
end

function _restore_frozen_phys(params, reference, frozen::AbstractVector{Symbol})
    isempty(frozen) && return params
    hasproperty(params, :phys) || return params
    updates = Pair{Symbol, Any}[]
    for name in propertynames(params.phys)
        value = name in frozen ?
                getproperty(reference.phys, name) : getproperty(params.phys, name)
        push!(updates, name => value)
    end
    return ComponentVector(phys = NamedTuple(updates), nn = params.nn)
end

function _training_converged(
        final_loss::Float64, initial_loss::Float64, config, diagnostics)
    isfinite(final_loss) && isfinite(initial_loss) || return false
    if config.constraint isa AugmentedLagrangianConfig
        return diagnostics.primal_residual ≤ config.constraint.tolerance ||
               final_loss ≤ initial_loss
    end
    return final_loss ≤ initial_loss
end

function _training_converged(final_loss, initial_loss, config, diagnostics)
    return _training_converged(Float64(final_loss), Float64(initial_loss),
        config, diagnostics)
end

"""
    train_ude(p_init, data, t_data, u0, tspan, model)
    train_ude(p_init, data, t_data, u0, tspan, nn, st; model, network)

Fit physical (and optional neural) parameters of a compiled UDE to one
trajectory. `mask` (default: every entry observed) drops the unobserved
entries of `data` from the loss on every horizon slice; without it a `NaN`
entry makes the loss non-finite. Use `train_experiments` for masked
multi-replicate data.
"""
function train_ude(p_init, data, t_data, u0, tspan, nn, st;
        model::Union{Nothing, UDEModel} = nothing,
        network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
        kwargs...)
    resolved = model === nothing ? compile_network(network, nn, st) : model
    return _train_ude_model(p_init, data, t_data, u0, tspan, resolved; kwargs...)
end

@inline function _train_ude_locked(p_init::P, data, t_data, u0, tspan, model::UDEModel,
        training_config::TrainingConfig, verbose::Bool, seed::Int,
        optimizer_state, checkpoint_path::Union{Nothing, AbstractString},
        checkpoint_every::Int, initial_iteration::Int, dual_init, rho_init,
        initial_outer::Int, initial_stage::Int, initial_stage_iteration::Int,
        previous_residual_init::Float64, session, mask) where {P}
    training_config = lock_training_config(model, training_config)
    local_session = session
    if local_session === nothing
        local_session = training_solve_session(
            model, u0, tspan, p_init;
            solver = training_config.solver)
    end
    diag = LossDiagnostics()
    state_count = size(data, 1)
    dual = dual_init === nothing ?
           zeros(eltype(p_init), state_count) : copy(dual_init)
    ρ = rho_init === nothing ?
        (training_config.constraint isa AugmentedLagrangianConfig ?
         training_config.constraint.initial_ρ : zero(eltype(p_init))) :
        rho_init
    function make_loss(local_data, local_times, local_span, local_mask)
        return (p, _) -> loss_mse(
            p, local_data, local_times, u0, local_span, model;
            constraint = training_config.constraint, dual, ρ,
            solver_config = training_config.solver, diagnostics = diag,
            mask = local_mask)
    end
    full_loss = make_loss(data, t_data, tspan, mask)
    initial_loss = try
        Float64(full_loss(p_init, nothing))
    catch error
        first_fraction = minimum(_horizon_fractions(training_config.horizon_schedule))
        min_pts = _horizon_min_points(training_config.horizon_schedule)
        count = clamp(round(Int, first_fraction * length(t_data)), min_pts,
            length(t_data))
        @warn "Full-horizon initial solve failed; starting curriculum." exception=error
        Float64(make_loss(data[:, 1:count], t_data[1:count],
            (tspan[1], t_data[count]), mask[:, 1:count])(p_init, nothing))
    end
    history = Float64[]
    params = p_init
    current_optimizer_state = optimizer_state
    current_outer = Ref(initial_outer)
    current_stage = Ref(initial_stage)
    current_stage_iteration = Ref(initial_stage_iteration)
    current_previous_residual = Ref(previous_residual_init)
    checkpoint_hook = if checkpoint_path === nothing || checkpoint_every ≤ 0
        nothing
    else
        function (current_params, current_state, _)
            current_stage_iteration[] += 1
            absolute_iteration = initial_iteration + length(history)
            absolute_iteration % checkpoint_every == 0 || return nothing
            metadata = (
                run = RunMetadata(
                    seed = seed,
                    data_hash = data_fingerprint(data, t_data, u0),
                    config = (; training = training_config)),
                dual = copy(dual),
                rho = ρ,
                outer = current_outer[],
                stage = current_stage[],
                stage_iteration = current_stage_iteration[],
                previous_residual = current_previous_residual[]
            )
            save_checkpoint(
                checkpoint_path,
                Checkpoint(CHECKPOINT_SCHEMA_VERSION, current_params,
                    current_state, absolute_iteration, metadata))
            return nothing
        end
    end
    outer_iterations = training_config.constraint isa
                       AugmentedLagrangianConfig ?
                       training_config.constraint.outer_iterations : 1
    previous_residual = previous_residual_init
    for outer in initial_outer:outer_iterations
        min_frac = _horizon_minimum_fraction(training_config.horizon_schedule)
        min_pts = _horizon_min_points(training_config.horizon_schedule)
        schedule = sort(unique(clamp.(
            _horizon_fractions(training_config.horizon_schedule),
            min_frac, 1.0)))
        for (stage, fraction) in pairs(schedule)
            outer == initial_outer && stage < initial_stage && continue
            current_outer[] = outer
            current_stage[] = stage
            count = clamp(round(Int, fraction * length(t_data)), min_pts,
                length(t_data))
            local_times = t_data[1:count]
            local_data = data[:, 1:count]
            local_mask = mask[:, 1:count]
            local_span = (tspan[1], local_times[end])
            final_stage = outer == outer_iterations &&
                          stage == length(schedule)
            stage_config = _stage_config(
                training_config, length(schedule), final_stage)
            if outer == initial_outer && stage == initial_stage &&
               initial_stage_iteration > 0
                stage_config = TrainingConfig(stage_config;
                    adam_iterations = max(
                        0, stage_config.adam_iterations -
                           initial_stage_iteration))
            else
                current_stage_iteration[] = 0
            end
            verbose && println(
                "  → AL $outer/$outer_iterations, horizon ",
                round(fraction; digits = 2))
            params, current_optimizer_state = _optimize_stage(
                params, make_loss(local_data, local_times, local_span, local_mask),
                stage_config, history, diag, verbose,
                current_optimizer_state, checkpoint_hook)
        end
        if training_config.constraint isa AugmentedLagrangianConfig
            prediction = predict_ude(
                params, u0, tspan, t_data, model;
                solver_config = training_config.solver,
                session = local_session)
            constraints = _constraint_values(
                prediction, training_config.constraint)
            residual = maximum(constraints)
            dual .= max.(zero(eltype(dual)), dual .+ ρ .* constraints)
            strategy = training_config.constraint
            if residual > strategy.progress_ratio * previous_residual
                ρ = min(strategy.max_ρ, strategy.growth * ρ)
            end
            previous_residual = residual
            current_previous_residual[] = residual
            residual ≤ strategy.tolerance && break
        end
    end
    final_loss = Float64(full_loss(params, nothing))
    metadata = RunMetadata(
        seed = seed,
        data_hash = data_fingerprint(data, t_data, u0),
        config = (; training = training_config))
    converged = _training_converged(
        final_loss, initial_loss, training_config, diag)
    retcode = _training_retcode(converged, diag)
    fitted = convert(P, params)
    typed_diag = _train_diagnostics(diag, dual, ρ, current_optimizer_state)
    return TrainingResult{P, Float64, Vector{Float64}, RunMetadata,
        TrainDiagnostics, TrainingRetcode}(
        fitted, history, initial_loss, final_loss, metadata, typed_diag,
        converged, retcode)
end

function _train_ude_model(p_init, data, t_data, u0, tspan, model::UDEModel;
        config::Union{Nothing, TrainingConfig} = nothing,
        adam_iters::Int = 300,
        adam_lr::Float64 = 0.01,
        bfgs_iters::Int = 100,
        log_every::Int = 20,
        verbose::Bool = true,
        seed::Integer = 0,
        optimizer_state = nothing,
        checkpoint_path::Union{Nothing, AbstractString} = nothing,
        checkpoint_every::Int = 0,
        initial_iteration::Int = 0,
        dual_init = nothing,
        rho_init = nothing,
        initial_outer::Int = 1,
        initial_stage::Int = 1,
        initial_stage_iteration::Int = 0,
        previous_residual_init = Inf,
        session = nothing,
        mask = nothing)
    training_config = isnothing(config) ?
                      TrainingConfig(
        adam_iterations = adam_iters,
        adam_learning_rate = adam_lr,
        bfgs_iterations = bfgs_iters,
        log_every = log_every) : config
    resolved_mask = mask === nothing ? trues(size(data)) : mask
    size(resolved_mask) == size(data) ||
        throw(DimensionMismatch("mask and data must have equal size"))
    return _train_ude_locked(
        p_init, data, t_data, u0, tspan, model, training_config, verbose,
        Int(seed), optimizer_state, checkpoint_path, checkpoint_every,
        initial_iteration, dual_init, rho_init, initial_outer, initial_stage,
        initial_stage_iteration, Float64(previous_residual_init), session,
        resolved_mask)
end

const _DEFAULT_TRAINING_CONFIG = TrainingConfig()

"""
    train_ude(p_init, data, t_data, u0, tspan, model::UDEModel; kwargs...)

Fit physical (and optional neural) parameters of a compiled UDE to one
trajectory. `mask` (default: every entry observed) drops the unobserved
entries of `data` from the loss. Use `train_experiments` for masked
multi-replicate data.
"""
function _train_ude_default(p_init::P, data::AbstractMatrix,
        t_data::AbstractVector, u0::AbstractVector,
        tspan::NTuple{2, Real}, model::UDEModel) where {P}
    training_config = lock_training_config(model, _DEFAULT_TRAINING_CONFIG)
    diag = LossDiagnostics()
    data64 = data isa Matrix{Float64} ? data : Matrix{Float64}(data)
    times64 = t_data isa Vector{Float64} ? t_data : collect(Float64, t_data)
    u064 = u0 isa Vector{Float64} ? u0 : collect(Float64, u0)
    tspan64 = (Float64(tspan[1]), Float64(tspan[2]))
    dual = zeros(Float64, size(data64, 1))
    solver = training_config.solver
    full_loss = LossMSECall(
        model, StructuralPositivity(), dual, 0.0, solver, diag,
        data64, times64, u064, tspan64)
    initial_loss = try
        Float64(full_loss(p_init, nothing))
    catch
        fractions = _horizon_fractions(training_config.horizon_schedule)
        first_fraction = minimum(fractions)
        min_pts = _horizon_min_points(training_config.horizon_schedule)
        count = clamp(round(Int, first_fraction * length(times64)), min_pts,
            length(times64))
        warmup = LossMSECall(
            model, StructuralPositivity(), dual, 0.0, solver, diag,
            data64[:, 1:count], times64[1:count], u064,
            (tspan64[1], times64[count]))
        Float64(warmup(p_init, nothing))
    end
    history = Float64[]
    params = p_init
    current_optimizer_state = nothing
    min_frac = _horizon_minimum_fraction(training_config.horizon_schedule)
    min_pts = _horizon_min_points(training_config.horizon_schedule)
    schedule = sort(unique(clamp.(
        _horizon_fractions(training_config.horizon_schedule),
        min_frac, 1.0)))
    n_stages = length(schedule)
    for (stage, fraction) in pairs(schedule)
        count = clamp(round(Int, fraction * length(times64)), min_pts,
            length(times64))
        local_times = times64[1:count]
        local_data = data64[:, 1:count]
        local_span = (tspan64[1], local_times[end])
        stage_config = _stage_config(training_config, n_stages, stage == n_stages)
        stage_loss = LossMSECall(
            model, StructuralPositivity(), dual, 0.0, solver, diag,
            local_data, local_times, u064, local_span)
        params, current_optimizer_state = _optimize_stage(
            params, stage_loss, stage_config, history, diag, true,
            current_optimizer_state, nothing)
    end
    final_loss = Float64(full_loss(params, nothing))
    metadata = RunMetadata(
        UInt64(0), now(UTC), VERSION, PACKAGE_VERSION,
        data_fingerprint(data64, times64, u064),
        (; training = training_config))
    converged = _training_converged(
        final_loss, initial_loss, training_config, diag)
    retcode = _training_retcode(converged, diag)
    fitted = convert(P, params)
    typed_diag = _train_diagnostics(diag, dual, 0.0, current_optimizer_state)
    return TrainingResult{P, Float64, Vector{Float64}, RunMetadata,
        TrainDiagnostics, TrainingRetcode}(
        fitted, history, initial_loss, final_loss, metadata, typed_diag,
        converged, retcode)
end

function train_ude(p_init::P, data::AbstractMatrix,
        t_data::AbstractVector, u0::AbstractVector,
        tspan::NTuple{2, Real}, model::UDEModel) where {P}
    # `Base.invokelatest` is a concrete Base call. JET cannot specialize
    # through Zygote / OrdinaryDiffEq / Optimisers; walking that kernel
    # reports Any-dispatch on their return values. The runtime path is
    # still `_train_ude_default` (horizon curriculum, Adam reuse, BFGS).
    return Base.invokelatest(
        _train_ude_default, p_init, data, t_data, u0, tspan, model)
end

function train_ude(p_init, data, t_data, u0, tspan, model::UDEModel; kwargs...)
    config = get(values(kwargs), :config, nothing)
    config !== nothing && config.restarts > 1 && throw(ArgumentError(
        "restarts is $(config.restarts), but train_ude fits one experiment and " *
        "has nothing to select restarts on: the lowest single-experiment loss " *
        "is not the lowest joint loss. Set restarts on the joint fit instead, " *
        "through discover_unknown_terms(training = ...), train_experiments or " *
        "train_experiments_with_warmup"))
    return _train_ude_model(p_init, data, t_data, u0, tspan, model; kwargs...)
end

function loss_mse(p, data, t_data, u0, tspan, model::UDEModel;
        constraint::AbstractConstraintStrategy = StructuralPositivity(),
        dual = zeros(eltype(p), size(data, 1)),
        ρ = one(eltype(p)),
        solver_config::SolverConfig = SolverConfig(),
        mask = trues(size(data)),
        diagnostics::Union{Nothing, LossDiagnostics} = nothing,
        session = nothing)
    prediction = predict_ude(p, u0, tspan, t_data, model;
        solver_config = solver_config,
        session = session)
    size(prediction) == size(data) ||
        throw(ErrorException("ODE solve terminated before all observations"))
    all(isfinite, prediction) ||
        throw(ErrorException("ODE solve returned non-finite states"))
    mse = _masked_mse(prediction, data, mask)
    constraints = _constraint_values(prediction, constraint)
    constraint_loss = constraint isa AugmentedLagrangianConfig ?
                      _augmented_term(
        constraints, dual, ρ, constraint.smoothness) : zero(mse)
    total = mse + constraint_loss
    residual = isempty(constraints) ? zero(mse) :
               max(zero(mse), maximum(constraints))
    if diagnostics !== nothing
        ignore_derivatives() do
            _record!(diagnostics, mse, constraint_loss, total, residual)
        end
    end
    return total
end

"""
    train_experiments(p_init, set, model; config, execution, ...)

Fit a UDE to an `ExperimentSet`. Adam follows `execution.batch_size`.
BFGS always refines the joint loss over every experiment, not the last
minibatch.
"""
function train_experiments(p_init, set::ExperimentSet, nn, st;
        model::Union{Nothing, UDEModel} = nothing,
        network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
        config::TrainingConfig = TrainingConfig(),
        execution::ExecutionConfig = ExecutionConfig(),
        verbose::Bool = true, seed::Integer = 0,
        optimizer_state = nothing,
        session = nothing)
    resolved_model = resolve_training_model(model, nn, st, network)
    training_config = lock_training_config(resolved_model, config)
    local_session = session === nothing ?
                    training_solve_session(resolved_model, set, p_init;
        solver = training_config.solver) : session
    diag = LossDiagnostics()
    dual = zeros(eltype(p_init), length(set.state_names))
    ρ = training_config.constraint isa AugmentedLagrangianConfig ?
        training_config.constraint.initial_ρ : zero(eltype(p_init))
    function batch_loss(p, experiments)
        total = zero(eltype(p))
        weight_sum = zero(eltype(p))
        for experiment in experiments
            weight = ignore_derivatives(() -> experiment_weight(experiment))
            scale = ignore_derivatives(() -> experiment_noise_scale(experiment))
            total += weight * loss_mse(
                p, experiment.observations, experiment.times, experiment.u0,
                (first(experiment.times), last(experiment.times)), nn, st;
                model = resolved_model, network = network,
                constraint = training_config.constraint, dual, ρ,
                solver_config = training_config.solver, mask = experiment.mask,
                diagnostics = nothing) / (scale^2)
            weight_sum += weight
        end
        return total / max(weight_sum, one(eltype(p_init)))
    end
    objective = (p, _) -> batch_loss(p, set.experiments)
    initial = objective(p_init, nothing)
    history = Float64[]
    params = p_init
    current_optimizer_state = optimizer_state
    outer_iterations = training_config.constraint isa AugmentedLagrangianConfig ?
                       training_config.constraint.outer_iterations : 1
    previous_residual = Inf
    for outer in 1:outer_iterations
        batches = experiment_batches(
            set, min(execution.batch_size, length(set));
            shuffle = !execution.deterministic,
            rng = MersenneTwister(seed + outer - 1))
        for batch in batches
            batch_objective = (p, _) -> batch_loss(p, batch)
            # Adam only on this minibatch. Joint BFGS runs once after the
            # outer loop so the last IC cannot monopolize the second-order step.
            batch_config = _stage_config(
                training_config, outer_iterations * length(batches), false)
            params, current_optimizer_state = _optimize_stage(
                params, batch_objective, batch_config, history, diag, verbose,
                current_optimizer_state, nothing)
        end
        if training_config.constraint isa AugmentedLagrangianConfig
            experiment_constraints = map(set.experiments) do experiment
                prediction = predict_ude(
                    params, experiment.u0,
                    (first(experiment.times), last(experiment.times)),
                    experiment.times, nn, st;
                    model = resolved_model,
                    network = network,
                    solver_config = training_config.solver,
                    session = local_session)
                _constraint_values(prediction, training_config.constraint)
            end
            constraints = reduce((left, right) -> max.(left, right),
                experiment_constraints)
            residual = max(zero(eltype(constraints)), maximum(constraints))
            dual .= max.(zero(eltype(dual)), dual .+ ρ .* constraints)
            if residual >
               training_config.constraint.progress_ratio * previous_residual
                ρ = min(
                    training_config.constraint.max_ρ,
                    training_config.constraint.growth * ρ)
            end
            previous_residual = residual
            residual ≤ training_config.constraint.tolerance && break
        end
    end
    if training_config.bfgs_iterations > 0
        polish = TrainingConfig(training_config;
            adam_iterations = 0,
            bfgs_iterations = training_config.bfgs_iterations)
        params, current_optimizer_state = _optimize_stage(
            params, objective, polish, history, diag, verbose,
            current_optimizer_state, nothing)
    end
    final = objective(params, nothing)
    metadata = RunMetadata(
        seed = seed, data_hash = experiment_fingerprint(set),
        config = Dict(:training => training_config, :experiments => length(set)))
    converged = _training_converged(final, initial, training_config, diag)
    retcode = _training_retcode(converged, diag)
    return TrainingResult(
        params, history, initial, final, metadata,
        (experiment_count = length(set), dual = copy(dual), ρ = ρ,
            primal_residual =
            training_config.constraint isa AugmentedLagrangianConfig ?
            max(zero(ρ), previous_residual) : zero(ρ),
            final_gradient_norm = isempty(diag.gradient_norm_history) ? NaN :
                                  last(diag.gradient_norm_history),
            gradient_norm_history = copy(diag.gradient_norm_history),
            bfgs = (attempted = diag.bfgs_attempted, success = diag.bfgs_success,
                retcode = diag.bfgs_retcode, message = diag.bfgs_message),
            optimizer_state = current_optimizer_state),
        converged, retcode)
end

"""
    train_experiments(p_init, set, model; config, seed, ...)

Joint fit over every experiment in `set`.

`config.restarts` above 1 fits that many draws of the neural term and returns
the one that reached the lowest final loss, with every attempt's loss in the
result's metadata under `restart_losses`. Restart one is `p_init` as given;
each later one redraws the neural block from `seed` and keeps the physical
guess, so a set of restarts repeats exactly. Anything already trained into the
neural block of `p_init` — a warm-up, say — is therefore kept only by restart
one; wrap the whole fit instead if the warm-up should be repeated per restart,
as `train_experiments_with_warmup` and `discover_unknown_terms` do.
"""
function train_experiments(p_init, set::ExperimentSet, model::UDEModel; kwargs...)
    options = values(kwargs)
    config = get(options, :config, nothing)
    if config !== nothing && config.restarts > 1
        rest = Base.structdiff(options, NamedTuple{(:config,)})
        return _best_of_restarts(p_init, model, config.restarts,
            get(options, :seed, 0)) do start
            train_experiments(start, set, model;
                rest..., config = TrainingConfig(config; restarts = 1))
        end
    end
    return train_experiments(
        p_init, set, model.nn, model.st;
        model = model, network = model.network, kwargs...)
end

function save_checkpoint(path::AbstractString, checkpoint::Checkpoint)
    directory = dirname(abspath(path))
    isdir(directory) ||
        throw(ArgumentError("checkpoint directory does not exist: $directory"))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        serialize(io, checkpoint)
    end
    mv(temporary, path; force = true)
    return path
end

function load_checkpoint(path::AbstractString)
    checkpoint = open(deserialize, path)
    checkpoint isa Checkpoint ||
        throw(ArgumentError("file does not contain a HybridKinetics checkpoint"))
    checkpoint.schema_version.major == CHECKPOINT_SCHEMA_VERSION.major ||
        throw(ArgumentError("incompatible checkpoint schema"))
    return checkpoint
end

"""
    resume_training(checkpoint, args...; kwargs...)

Resume Adam training from versioned parameters, Optimisers state, iteration,
Augmented-Lagrangian dual variables and penalty state. BFGS is intentionally a
terminal refinement stage and is restarted if requested after resumption.
"""
function resume_training(checkpoint::Checkpoint, args...; kwargs...)
    metadata = checkpoint.metadata
    dual = hasproperty(metadata, :dual) ? metadata.dual : nothing
    rho = hasproperty(metadata, :rho) ? metadata.rho : nothing
    outer = hasproperty(metadata, :outer) ? metadata.outer : 1
    stage = hasproperty(metadata, :stage) ? metadata.stage : 1
    stage_iteration = hasproperty(metadata, :stage_iteration) ?
                      metadata.stage_iteration : 0
    previous_residual = hasproperty(metadata, :previous_residual) ?
                        metadata.previous_residual : Inf
    return train_ude(
        checkpoint.params, args...;
        optimizer_state = checkpoint.optimizer_state,
        initial_iteration = checkpoint.iteration,
        dual_init = dual,
        rho_init = rho,
        initial_outer = outer,
        initial_stage = stage,
        initial_stage_iteration = stage_iteration,
        previous_residual_init = previous_residual,
        kwargs...)
end
