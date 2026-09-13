@testset "Fisher identifiability report" begin
    rng = MersenneTwister(51)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    u0 = [0.2, 0.1]
    tspan = (0.0, 2.0)
    times, clean, _, _ = generate_data(
        rng; network = network, u0 = u0, tspan = tspan,
        n_points = 12, noise_σ = 0.0)
    report = assess_identifiability(
        model, params, clean, times, u0, tspan)
    @test length(report.parameter_names) == 3
    @test size(report.fisher_information) == (3, 3)
    @test report.condition_number > 0
    @test all(report.identifiable)
    uncertainty = estimate_parameter_uncertainty(
        model, params, clean, times, u0, tspan; level = 0.95)
    @test uncertainty.method == :fisher
    @test length(uncertainty.estimates) == 3
    @test all(uncertainty.lower .<= uncertainty.estimates)
    # The Fisher information is formed from the observed entries only: with
    # state 2 unobserved it equals J'J/σ² over the rows of state 1, and with
    # every entry observed the mask changes nothing.
    hidden_mask = trues(size(clean))
    hidden_mask[2, :] .= false
    hidden = assess_identifiability(
        model, params, clean, times, u0, tspan; mask = hidden_mask)
    jacobian, _ = HybridKinetics.trajectory_jacobian(model, params, u0, tspan, times)
    rows = vec(hidden_mask)
    expected = (jacobian[rows, :]' * jacobian[rows, :]) ./ hidden.residual_variance
    @test hidden.fisher_information≈expected rtol=1e-12
    @test hidden.fisher_information != report.fisher_information
    all_true = assess_identifiability(
        model, params, clean, times, u0, tspan; mask = trues(size(clean)))
    @test all_true.fisher_information == report.fisher_information
    @test all_true.condition_number == report.condition_number
end

@testset "production/destruction cosine uses observed entries only" begin
    rng = MersenneTwister(52)
    network = build_hill_recovery_network(; known = false, hill_order = 2)
    model, params = build_ude_model(rng, network)
    u0 = [0.4, 0.3]
    tspan = (0.0, 3.0)
    times = collect(range(0.0, 3.0; length = 10))
    data = predict_ude(params, u0, tspan, times, model)
    full = HybridKinetics.production_destruction_tradeoff(
        model, params, data, times, u0, tspan)
    same = HybridKinetics.production_destruction_tradeoff(
        model, params, data, times, u0, tspan; mask = trues(size(data)))
    @test same.collinearity == full.collinearity
    @test same.condition_number == full.condition_number
    hidden_mask = trues(size(data))
    hidden_mask[2, :] .= false
    hidden = HybridKinetics.production_destruction_tradeoff(
        model, params, data, times, u0, tspan; mask = hidden_mask)
    @test 0.0 <= hidden.collinearity <= 1.0
    @test hidden.collinearity != full.collinearity
    # Over a single observed entry the two sensitivities are scalars, so the
    # cosine is exactly one: the unobserved entries contribute nothing.
    one_mask = falses(size(data))
    one_mask[1, end] = true
    single = HybridKinetics.production_destruction_tradeoff(
        model, params, data, times, u0, tspan; mask = one_mask)
    @test single.collinearity≈1.0 atol=1e-12
end

@testset "training retcode and gradient diagnostics" begin
    rng = MersenneTwister(52)
    model, params = build_ude_model(rng, build_linear_test_network())
    times, _, noisy, _ = generate_data(
        rng; network = build_linear_test_network(),
        u0 = [0.2, 0.1], tspan = (0.0, 1.0), n_points = 8, noise_σ = 0.01)
    result = train_ude(
        params, noisy, times, [0.2, 0.1], (0.0, 1.0), model;
        adam_iters = 3, bfgs_iters = 0, verbose = false)
    @test result.retcode isa HybridKinetics.TrainingRetcode
    @test result.retcode in (HybridKinetics.Success, HybridKinetics.NotConverged)
    @test hasproperty(result.diagnostics, :final_gradient_norm)
    @test !isempty(result.diagnostics.gradient_norm_history)
end

@testset "discovery uncertainty reports" begin
    network = BiologicalNetwork([NodeSpec(name = :substrate)], EdgeSpec[])
    x = collect(range(0.05, 3.0; length = 240))
    X = reshape(x, 1, :)
    derivative = 2.4 .* x ./ (0.7 .+ x)
    spec = local_basis(
        network, 1; degree = 4, include_interactions = false,
        X, derivative, max_variables = 1)
    numerator, denominator = HybridKinetics._fit_implicit(
        spec, X, derivative, collect(eachindex(x)), 1e-7)
    candidate = ImplicitCandidate(
        1, spec, numerator, denominator,
        fill(0.9, length(numerator) + length(denominator)),
        1e-8, 1.0)
    result = DiscoveryResult(
        true, "ok", format_equation(candidate), [spec], nothing,
        [candidate], RunMetadata(seed = 1))
    reports = uncertainty_reports(result)
    @test length(reports) == 1
    @test !isempty(reports[1].stable_terms)
end

@testset "multi-trajectory discovery" begin
    rng = MersenneTwister(53)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    set = generate_experiment_set(
        rng; network = network,
        initial_conditions = [[0.2, 0.1], [0.3, 0.15]], n_points = 20, noise_σ = 0.01)
    trained = train_ude(
        params, set.experiments[1].observations,
        set.experiments[1].times, set.experiments[1].u0,
        (first(set.experiments[1].times), last(set.experiments[1].times)),
        model; adam_iters = 4, bfgs_iters = 0, verbose = false)
    result = discover_equations(
        trained.params, model, set;
        n_samples = 40, verbose = false,
        config = DiscoveryConfig(backend = ExplicitSTLSQ()))
    @test result isa DiscoveryResult
    @test result.success
    @test result.retcode === DiscoverySuccess
    @test !isempty(result.equations)
end

@testset "scientific benchmark suite" begin
    outcomes = run_benchmark_suite(MersenneTwister(54);
        adam_iters = 10, discovery = true)
    @test length(outcomes) == 3
    @test all(outcome -> isfinite(outcome.final_loss), outcomes)
    @test all(outcome -> outcome.retcode isa HybridKinetics.TrainingRetcode, outcomes)
end

@testset "ground truth generator separation" begin
    rng = MersenneTwister(55)
    p53_compiled = GroundTruthModel(rng, DEFAULT_EXAMPLE_NETWORK)
    @test p53_compiled.generator == :compiled_mechanism
    p53_hill = GroundTruthModel(
        rng, DEFAULT_EXAMPLE_NETWORK; generator = :hill_p53_fixture)
    @test p53_hill.generator == :hill_p53_fixture
    linear_truth = GroundTruthModel(rng, build_linear_test_network())
    @test linear_truth.generator == :compiled_mechanism
    tspan = (0.0, 1.0)
    times = collect(range(tspan...; length = 6))
    u0 = [0.2, 0.1]
    compiled = predict_ude(
        linear_truth.parameters, u0, tspan, times, linear_truth.model)
    hill_params = default_truth_params()
    hill_prob = SciMLBase.ODEProblem(ground_truth!, u0, tspan, hill_params)
    hill_sol = Array(solve(hill_prob, Tsit5(); saveat = times, sensealg = nothing))
    @test size(compiled) == size(hill_sol)
    @test !isapprox(compiled, hill_sol; rtol = 0.05)
end
