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
