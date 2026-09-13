@testset "the freeze of the adjacent-bias study is a real freeze" begin
    # The sequential setting trains one term, then holds it fixed while the
    # other trains. The hold must not be a parameter the optimiser can still
    # move, or the experiment would measure something else.
    fx = HybridKinetics.multi_term_fixture(:coupled)
    calls = Ref(0)
    rate = (x, p, regulators) -> (calls[] += 1; 0.37)
    network = HybridKinetics.adjacent_frozen_network(fx, :A, :B, rate)
    model, params = build_ude_model(MersenneTwister(1), network)
    # Exactly one term is still neural; the frozen one is a custom term.
    @test length(HybridKinetics.neural_destruction_terms(model)) == 1
    @test count(t -> t isa HybridKinetics.CustomDestructionTerm,
        model.compiled.destruction_terms) == 1
    state = [0.5, 0.6, 0.4]
    first_du = ude_rhs(state, params, 0.0, model)
    @test calls[] > 0
    # `CustomKineticMetadata` declares a rate parameter, so the schema carries
    # one; the evaluator never reads it, so it cannot change the frozen rate.
    moved = copy(params)
    if :vmax in parameter_schema(model).phys_names
        moved.phys[:vmax] = 5.0
        @test ude_rhs(state, moved, 0.0, model) == first_du
    end
end

@testset "pre-training one head leaves the other parameters alone" begin
    fx = HybridKinetics.multi_term_fixture(:coupled)
    network = fx.build(; unknown = Tuple(fx.nodes))
    model, p0 = build_ude_model(MersenneTwister(103), network)
    start = HybridKinetics.adjacent_initial_parameters(model, p0)
    term = first(HybridKinetics.neural_destruction_terms(model))
    r = collect(range(0.1, 1.6; length = 12))
    target = hill_rate_truth(r; vmax = fx.truth.vmax_a, K = fx.truth.K_a, n = 2)
    fitted = HybridKinetics.adjacent_pretrain_head!(model, start, term, r, target;
        iterations = 120)
    other = Symbol("head_", term.nn_index == 1 ? 2 : 1)
    @test fitted.phys == start.phys
    @test getproperty(fitted.nn, other) == getproperty(start.nn, other)
    @test getproperty(fitted.nn, Symbol("head_", term.nn_index)) !=
          getproperty(start.nn, Symbol("head_", term.nn_index))
    # The fit must actually move towards the target: a loss built on
    # `sample_unknown_destruction` would silently return no gradient.
    before = HybridKinetics.rate_rel_rmse(
        vec(sample_unknown_destruction_grid(model, start, term; r_range = r)[2]), target)
    after = HybridKinetics.rate_rel_rmse(
        vec(sample_unknown_destruction_grid(model, fitted, term; r_range = r)[2]), target)
    @test after < before
end

@testset "per-state weights reach the trainer" begin
    # The trainer multiplies each experiment's loss by `metadata[:weight]`, so
    # one single-state replica per state expresses a per-state weight without
    # changing the trainer.
    fx = HybridKinetics.multi_term_fixture(:coupled)
    data = HybridKinetics.adjacent_data(;
        fixture = :coupled, seed = 103, noise_σ = 0.0, n_points = 8, n_ics = 3)
    weights = HybridKinetics.adjacent_inverse_variance_weights(data.train_set)
    @test length(weights) == length(data.train_set.state_names)
    @test all(0 .< weights .<= 1)
    replicas = HybridKinetics.adjacent_weighted_set(data.train_set, weights)
    @test length(replicas) == length(data.train_set) * length(weights)
    @test sort(unique(HybridKinetics.experiment_weight(e) for e in replicas.experiments)) ≈
          sort(unique(weights))
    # Every replica observes exactly one state, and together they cover the
    # same entries as the experiment they came from.
    for experiment in replicas.experiments
        @test count(any(experiment.mask[state, :])
        for state in 1:size(experiment.mask, 1)) == 1
    end
    covered = sum(count(e.mask) for e in replicas.experiments)
    @test covered == sum(count(e.mask) for e in data.train_set.experiments)
end

@testset "the coupled fixture's ordering is not readable from the graph" begin
    # Both unknown nodes regulate the other's term, so a user cannot tell from
    # the interaction graph which term is upstream. The sequential experiment
    # therefore runs both orderings rather than assuming one.
    fx = HybridKinetics.multi_term_fixture(:coupled)
    network = fx.build(; unknown = Tuple(fx.nodes))
    model, _ = build_ude_model(MersenneTwister(1), network)
    roles = HybridKinetics.adjacent_roles(model)
    @test length(roles) == 2
    @test all(role == "regulated by the other unknown" for role in values(roles))
    @test !HybridKinetics.adjacent_ordering_is_graph_readable(model)
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"AdjacentBiasStudy.jl\")", src)
    @test occursin("include(\"ReliabilityStudy.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "AdjacentBiasStudy.jl"))
    @test isfile(joinpath(@__DIR__, "..", "src", "ReliabilityStudy.jl"))
    for name in (:adjacent_run, :reliability_train, :adjacent_frozen_network)
        @test !(name in names(HybridKinetics))
    end
end

@testset "the denominator report reaches past the samples" begin
    # The divergence question asks whether the candidate alone says the
    # resimulation will fail. The report must therefore look where the
    # resimulation can go, not only where the fit was made.
    candidate = HybridKinetics.synthetic_near_zero_implicit_candidate()
    X = reshape(collect(range(0.1, 0.5; length = 25)), 1, :)
    near = HybridKinetics.reliability_denominator_report(candidate, X; margin = 0.5)
    # D = 1 - r on [0.1, 0.5]: smallest value 0.5, nothing to warn about.
    @test near.min_samples≈0.5 atol=1e-12
    @test near.min_box≈0.5 atol=1e-12
    @test !near.sign_change_box
    @test !near.sign_change_extended
    far = HybridKinetics.reliability_denominator_report(candidate, X; margin = 1.5)
    # Widened to [0, 1.1] the root at r = 1 is inside, and only the widened
    # grid sees it.
    @test far.min_box≈0.5 atol=1e-12
    @test far.min_extended < 0
    @test !far.sign_change_box
    @test far.sign_change_extended
    # The lower edge is never pushed below zero, which no concentration reaches.
    safe = HybridKinetics.reliability_denominator_report(
        HybridKinetics.synthetic_safe_implicit_candidate(), X; margin = 5.0)
    @test safe.min_extended == 1.0
end

@testset "a run with no candidate is neither a divergence nor a run that was fine" begin
    rows = [(; success = true, diverged = true, denominator_min_box = 0.05,
            denominator_min_extended = -0.2, sign_change_box = false,
            sign_change_extended = true, negative_rate_box = false),
        (; success = true, diverged = false, denominator_min_box = 0.9,
            denominator_min_extended = 0.8, sign_change_box = false,
            sign_change_extended = false, negative_rate_box = false),
        (; success = false, diverged = false, denominator_min_box = NaN,
            denominator_min_extended = NaN, sign_change_box = false,
            sign_change_extended = false, negative_rate_box = false)]
    checks = HybridKinetics.reliability_audit_summary(rows)
    @test first(checks).caught == 1
    for check in checks[2:end]
        @test check.of == 1
        @test check.of_fine == 1
    end
    widened = only(c for c in checks if c.name == "sign change on the widened box")
    @test widened.caught == 1
    @test widened.false_alarms == 0
end

@testset "the grid of the report fits the point budget" begin
    grid = HybridKinetics._reliability_grid([0.1, 0.2], [0.5, 0.6], 21, 400_000)
    @test size(grid) == (2, 441)
    @test minimum(grid[1, :])≈0.1 atol=1e-12
    @test maximum(grid[2, :])≈0.6 atol=1e-12
    @test size(HybridKinetics._reliability_grid(zeros(5), ones(5), 21, 1000), 2) ≤ 1000
    # A coordinate that never moves, as in the constant sample design, gives a
    # grid and not an error.
    flat = HybridKinetics._reliability_grid([0.4], [0.4], 5, 100)
    @test all(flat .== 0.4)
end

@testset "hiding observations leaves the regulator a grid to span" begin
    # Blanking a whole state leaves `_regulator_grid` nothing finite to work
    # with, so the masked warm-up would measure the absence of a regulator.
    # The study hides every second observation instead.
    trained = HybridKinetics.reliability_train(;
        seed = 103, init_seed = 103, noise_σ = 0.0, mask_state = 2,
        warmup_scale = 1, kind = :smoke)
    @test isfinite(trained.training.final_loss)
    @test all(isfinite, trained.X)
    @test all(isfinite, trained.D)
    hidden = sum(count(isnan, e.observations[2, :]) for e in trained.train_set.experiments)
    total = sum(size(e.observations, 2) for e in trained.train_set.experiments)
    @test 0 < hidden < total
    @test all(any(isfinite, e.observations[2, :]) for e in trained.train_set.experiments)
end

@testset "the frozen rate is the rate that was learned, and it carries a gradient" begin
    # The sequential setting freezes the upstream term inside the right-hand
    # side the optimiser differentiates. A rate read through
    # `sample_unknown_destruction` fills a vector by index, so Zygote refuses
    # it; one built on `_destruction_contribution` does not, and the custom
    # term's own scale must put back exactly what was divided out.
    fx = HybridKinetics.multi_term_fixture(:coupled)
    first_net = fx.build(; unknown = (:A,))
    first_model, first_p0 = build_ude_model(MersenneTwister(103), first_net)
    first_term = only(HybridKinetics.neural_destruction_terms(first_model))
    frozen = HybridKinetics.adjacent_initial_parameters(first_model, first_p0)
    rate = (x, _p, _regulators) -> HybridKinetics._destruction_contribution(
        first_term, first_term.target, x, frozen, first_model.nn,
        first_model.st) / first_term.scale
    second_net = HybridKinetics.adjacent_frozen_network(fx, :A, :B, rate)
    second_model, second_p0 = build_ude_model(MersenneTwister(103), second_net)
    custom = only(term for term in second_model.compiled.destruction_terms
    if term isa HybridKinetics.CustomDestructionTerm)
    @test custom.scale == first_term.scale
    params = HybridKinetics.adjacent_initial_parameters(second_model, second_p0)
    state = [0.5, 0.6, 0.4]
    learned = HybridKinetics._destruction_contribution(first_term, first_term.target,
        state, frozen, first_model.nn, first_model.st)
    held = HybridKinetics._destruction_contribution(custom, custom.target, state,
        params, second_model.nn, second_model.st)
    @test held == learned
    gradient = only(Zygote.gradient(
        q -> sum(abs2, ude_rhs(state, q, 0.0, second_model)), params))
    @test all(isfinite, ComponentArrays.getdata(gradient))
    @test sqrt(sum(abs2, ComponentArrays.getdata(gradient))) > 0
end
