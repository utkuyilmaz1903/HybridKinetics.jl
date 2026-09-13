const _TRAINING_SEED = 2024

function _training_fixture(rng::AbstractRNG = MersenneTwister(_TRAINING_SEED))
    nn, nn_parameters, nn_state = build_ude_nn(rng)
    parameters = pack_parameters(
        (α_p53 = 0.9, β_mdm2 = 1.1, γ_mdm2 = 1.5, signal = 1.0),
        nn_parameters)
    u0 = [0.2, 0.1]
    tspan = (0.0, 0.5)
    times, _, noisy, _ = generate_data(
        rng; u0 = u0, tspan = tspan, n_points = 6, noise_σ = 0.02)
    fast_solver = SolverConfig(abstol = 1e-5, reltol = 1e-5)
    return (; parameters, nn, nn_state, u0, tspan, times, noisy, fast_solver)
end

function _apply_al_outer_step!(
        constraints::AbstractVector,
        dual::AbstractVector,
        ρ::Real,
        strategy::AugmentedLagrangianConfig,
        previous_residual::Real)
    residual = maximum(constraints)
    dual .= max.(zero(eltype(dual)), dual .+ ρ .* constraints)
    updated_ρ = ρ
    if residual > strategy.progress_ratio * previous_residual
        updated_ρ = min(strategy.max_ρ, strategy.growth * ρ)
    end
    return residual, updated_ρ
end

@testset "augmented lagrangian mechanics" begin
    strategy = AugmentedLagrangianConfig(
        initial_ρ = 1.0, growth = 2.0, tolerance = 1e-4,
        outer_iterations = 10, progress_ratio = 0.5, smoothness = 1e-2)

    @testset "smooth PHR augmentation" begin
        dual = [0.2, 0.0]
        ρ = strategy.initial_ρ
        feasible = [-0.1, -0.05]
        violated = [0.3, 0.1]
        feasible_term = HybridKinetics._augmented_term(
            feasible, dual, ρ, strategy.smoothness)
        violated_term = HybridKinetics._augmented_term(
            violated, dual, ρ, strategy.smoothness)
        @test violated_term > feasible_term
        @test violated_term > zero(violated_term)
        @test all(isfinite, (feasible_term, violated_term))
    end

    @testset "dual and penalty drive constraints to tolerance" begin
        constraints = [0.8, 0.55]
        dual = zeros(2)
        ρ = strategy.initial_ρ
        previous = Inf
        residuals = Float64[]
        ρ_history = Float64[ρ]
        for _ in 1:(strategy.outer_iterations)
            residual, ρ = _apply_al_outer_step!(
                constraints, dual, ρ, strategy, previous)
            push!(residuals, residual)
            push!(ρ_history, ρ)
            previous = residual
            constraints = max.(0, constraints .* 0.35)
            residual ≤ strategy.tolerance && break
        end
        @test residuals[end] < residuals[1]
        @test residuals[end] ≤ strategy.tolerance
        @test any(>(0), dual)
        @test ρ_history[end] ≥ ρ_history[1]
    end
end

@testset "augmented lagrangian end-to-end training" begin
    fixture = _training_fixture()
    strategy = AugmentedLagrangianConfig(
        initial_ρ = 2.0, growth = 3.0, tolerance = 1e-3,
        outer_iterations = 3, smoothness = 1e-2)
    config = TrainingConfig(
        adam_iterations = 12,
        adam_learning_rate = 0.02,
        bfgs_iterations = 0,
        log_every = 10_000,
        horizon_schedule = [1.0],
        constraint = strategy,
        solver = fixture.fast_solver)

    initial_prediction = predict_ude(
        fixture.parameters, fixture.u0, fixture.tspan, fixture.times,
        fixture.nn, fixture.nn_state; solver_config = config.solver)
    initial_constraints = HybridKinetics._constraint_values(
        initial_prediction, strategy)
    initial_residual = maximum(initial_constraints)

    result = train_ude(
        fixture.parameters, fixture.noisy, fixture.times, fixture.u0,
        fixture.tspan, fixture.nn, fixture.nn_state;
        config = config, verbose = false, seed = _TRAINING_SEED)

    final_prediction = predict_ude(
        result.params, fixture.u0, fixture.tspan, fixture.times,
        fixture.nn, fixture.nn_state; solver_config = config.solver)
    final_constraints = HybridKinetics._constraint_values(
        final_prediction, strategy)
    reported_primal = max(0.0, maximum(final_constraints))

    @test result.diagnostics.primal_residual ≈ reported_primal
    @test result.diagnostics.ρ ≥ strategy.initial_ρ
    @test all(isfinite, result.diagnostics.dual)
    @test reported_primal ≤ max(0.0, initial_residual) + 1e-8
    @test reported_primal ≤ strategy.tolerance ||
          result.diagnostics.ρ > strategy.initial_ρ ||
          any(>(0), result.diagnostics.dual)
    @test all(isfinite, result.params)
    @test isfinite(result.final_loss)
end

@testset "deterministic resume equivalence" begin
    fixture = _training_fixture()
    total_iterations = 20
    split_iteration = 10
    config = TrainingConfig(
        adam_iterations = total_iterations,
        adam_learning_rate = 0.015,
        bfgs_iterations = 0,
        log_every = 10_000,
        horizon_schedule = [1.0],
        constraint = StructuralPositivity(),
        solver = fixture.fast_solver)
    config_first = TrainingConfig(
        adam_iterations = split_iteration,
        adam_learning_rate = config.adam_learning_rate,
        bfgs_iterations = 0,
        log_every = config.log_every,
        horizon_schedule = config.horizon_schedule,
        constraint = config.constraint,
        solver = config.solver)

    continuous = train_ude(
        fixture.parameters, fixture.noisy, fixture.times, fixture.u0,
        fixture.tspan, fixture.nn, fixture.nn_state;
        config = config, verbose = false, seed = _TRAINING_SEED)

    checkpoint_path = tempname()
    try
        partial = train_ude(
            fixture.parameters, fixture.noisy, fixture.times, fixture.u0,
            fixture.tspan, fixture.nn, fixture.nn_state;
            config = config_first, verbose = false, seed = _TRAINING_SEED,
            checkpoint_path = checkpoint_path,
            checkpoint_every = split_iteration)
        checkpoint = load_checkpoint(checkpoint_path)
        resumed = resume_training(
            checkpoint, fixture.noisy, fixture.times, fixture.u0,
            fixture.tspan, fixture.nn, fixture.nn_state;
            config = config, verbose = false, seed = _TRAINING_SEED)

        @test checkpoint.iteration == split_iteration
        @test length(continuous.history) == total_iterations
        @test length(partial.history) == split_iteration
        @test length(resumed.history) == total_iterations - split_iteration

        @test partial.history ≈ continuous.history[1:split_iteration]
        @test continuous.params≈resumed.params atol=0 rtol=0
        @test continuous.final_loss≈resumed.final_loss atol=0 rtol=0
        @test continuous.history ≈ vcat(partial.history, resumed.history)
    finally
        rm(checkpoint_path; force = true)
    end
end

@testset "augmented lagrangian resume equivalence" begin
    fixture = _training_fixture()
    total_iterations = 16
    split_iteration = 8
    strategy = AugmentedLagrangianConfig(
        initial_ρ = 1.5, growth = 2.5, tolerance = 1e-3,
        outer_iterations = 1, smoothness = 1e-2)
    config = TrainingConfig(
        adam_iterations = total_iterations,
        adam_learning_rate = 0.015,
        bfgs_iterations = 0,
        log_every = 10_000,
        horizon_schedule = [1.0],
        constraint = strategy,
        solver = fixture.fast_solver)
    config_first = TrainingConfig(
        adam_iterations = split_iteration,
        adam_learning_rate = config.adam_learning_rate,
        bfgs_iterations = 0,
        log_every = config.log_every,
        horizon_schedule = config.horizon_schedule,
        constraint = config.constraint,
        solver = config.solver)

    continuous = train_ude(
        fixture.parameters, fixture.noisy, fixture.times, fixture.u0,
        fixture.tspan, fixture.nn, fixture.nn_state;
        config = config, verbose = false, seed = _TRAINING_SEED)

    checkpoint_path = tempname()
    try
        partial = train_ude(
            fixture.parameters, fixture.noisy, fixture.times, fixture.u0,
            fixture.tspan, fixture.nn, fixture.nn_state;
            config = config_first, verbose = false, seed = _TRAINING_SEED,
            checkpoint_path = checkpoint_path,
            checkpoint_every = split_iteration)
        checkpoint = load_checkpoint(checkpoint_path)
        resumed = resume_training(
            checkpoint, fixture.noisy, fixture.times, fixture.u0,
            fixture.tspan, fixture.nn, fixture.nn_state;
            config = config, verbose = false, seed = _TRAINING_SEED)

        @test continuous.params≈resumed.params atol=0 rtol=0
        @test continuous.final_loss≈resumed.final_loss atol=0 rtol=0
        @test continuous.history ≈ vcat(partial.history, resumed.history)
        @test getfield(continuous.diagnostics, :dual)≈
        getfield(resumed.diagnostics, :dual) atol=0 rtol=0
        @test getfield(continuous.diagnostics, :ρ)≈
        getfield(resumed.diagnostics, :ρ) atol=0 rtol=0
        @test getfield(continuous.diagnostics,
            :primal_residual)≈
        getfield(resumed.diagnostics, :primal_residual) atol=0 rtol=0
    finally
        rm(checkpoint_path; force = true)
    end
end

@testset "frozen_phys holds k_ba fixed" begin
    rng = MersenneTwister(11)
    net = build_linear_test_network()
    model, p0 = build_ude_model(rng, net)
    truth = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    u0 = [0.35, 0.25]
    tspan = (0.0, 4.0)
    times, data, _, _ = generate_data(
        rng; network = net, u0 = u0, tspan = tspan, n_points = 20,
        noise_σ = 0.0, truth_params = truth)
    init = pack_parameters((k_ba = 1.1, k_a = 0.9, k_b = 0.7), p0.nn)
    raw = init.phys.k_ba
    fit = train_ude(
        init, data, times, u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = 8, bfgs_iterations = 0, log_every = 10^6,
            frozen_phys = [:k_ba]),
        verbose = false)
    @test fit.params.phys.k_ba ≈ raw
    @test fit.params.phys.k_a != init.phys.k_a || isfinite(fit.final_loss)
end

@testset "train_ude applies the observation mask" begin
    # An unobserved state (NaN row, masked) must not poison the loss: before
    # 0.17.2 train_ude ignored the mask, so the warm-up of
    # discover_unknown_terms on a partially observed experiment started from
    # a NaN loss.
    rng = MersenneTwister(11)
    net = build_linear_test_network()
    model, p0 = build_ude_model(rng, net)
    truth = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    u0 = [0.35, 0.25]
    tspan = (0.0, 4.0)
    times, data, _, _ = generate_data(
        rng; network = net, u0 = u0, tspan = tspan, n_points = 20,
        noise_σ = 0.0, truth_params = truth)
    init = pack_parameters((k_ba = 1.1, k_a = 0.9, k_b = 0.7), p0.nn)
    config = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0,
        log_every = 10^6)
    hidden = copy(data)
    hidden[2, :] .= NaN
    mask = trues(size(data))
    mask[2, :] .= false
    fit = train_ude(init, hidden, times, u0, tspan, model;
        config = config, verbose = false, mask = mask)
    @test isfinite(fit.initial_loss)
    @test isfinite(fit.final_loss)
    @test all(isfinite, fit.params)
    # Without the mask the same data give a non-finite loss.
    bare = train_ude(init, hidden, times, u0, tspan, model;
        config = config, verbose = false)
    @test !isfinite(bare.initial_loss)
    # With every entry observed the mask keyword changes nothing.
    plain = train_ude(init, data, times, u0, tspan, model;
        config = config, verbose = false)
    all_true = train_ude(init, data, times, u0, tspan, model;
        config = config, verbose = false, mask = trues(size(data)))
    @test all_true.params == plain.params
    @test all_true.final_loss == plain.final_loss
    @test_throws DimensionMismatch train_ude(init, data, times, u0, tspan, model;
        config = config, verbose = false, mask = trues(2, 3))
end
