@testset "the study's loss is the loss training minimises" begin
    # Everything in this phase is read off the loss between two solutions, so
    # the loss the study evaluates has to be the one the optimiser descended.
    # At the trained parameters it must reproduce the training's own final
    # loss, and at the starting parameters its initial loss.
    trained = HybridKinetics.gap_train(; seed = 103, noise_σ = 0.0, kind = :smoke)
    own = HybridKinetics.gap_training_loss(
        trained.model, trained.train_set, trained.params; config = trained.config)
    @test own≈trained.training.final_loss rtol=1e-10
    interpolated = HybridKinetics.gap_interpolate(
        trained.model, trained.train_set, trained.params, trained.params;
        n = 3, config = trained.config)
    @test length(interpolated) == 3
    @test all(row -> row.loss ≈ trained.training.final_loss, interpolated)
    @test first(interpolated).t == 0.0
    @test last(interpolated).t == 1.0
end

@testset "the ceiling is a real ceiling" begin
    # The true rate, put through the reference discovery unchanged, must give
    # the true support. Without that the milestone would be chasing a target
    # the discovery step cannot reach whatever training does.
    trained = HybridKinetics.gap_train(; seed = 103, noise_σ = 0.0, kind = :smoke)
    exact = HybridKinetics.gap_discover(trained, HybridKinetics.gap_true_rate(trained))
    @test exact.support_f1 == 1.0
    @test exact.support_recall == 1.0
    @test exact.support == "n:x[1]^2 d:x[1]^2"
    # ... and the sweep's first point is that rate, at zero error.
    sweep = HybridKinetics.gap_alpha_sweep(trained; alphas = [0.0, 1.0])
    @test first(sweep).alpha == 0.0
    @test first(sweep).rate_error == 0.0
    @test first(sweep).support_f1 == 1.0
    @test last(sweep).rate_error > 0
end

@testset "pre-training to the true rate moves the network and nothing else" begin
    # One unknown term gives a single network rather than a multi-head one, so
    # the mask that 0.18 used for heads does not apply; the physical guess has
    # to stay where the package's own path puts it either way.
    ude_net = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    model, p0 = build_ude_model(MersenneTwister(103), ude_net)
    names = Tuple(parameter_schema(model).phys_names)
    start = pack_parameters(
        NamedTuple{names}(ntuple(_ -> 0.8, length(names))), p0.nn)
    term = HybridKinetics.only_unknown_destruction(model)
    @test !hasproperty(start.nn, :head_1)
    truth = HybridKinetics.LIBRARY_STUDY_TWO_STATE_TRUTH
    r = collect(range(0.1, 1.6; length = 12))
    target = HybridKinetics.gap_true_rate_on(r, truth)
    fitted = HybridKinetics.gap_pretrain_network(model, start, term, r, target;
        iterations = 150)
    @test fitted.phys == start.phys
    @test fitted.nn != start.nn
    before = HybridKinetics.rate_rel_rmse(
        vec(sample_unknown_destruction_grid(model, start, term; r_range = r)[2]), target)
    after = HybridKinetics.rate_rel_rmse(
        vec(sample_unknown_destruction_grid(model, fitted, term; r_range = r)[2]), target)
    @test after < before
end

@testset "the standard start is the package's own start" begin
    # The ceiling is only readable against a floor produced the same way.
    same = HybridKinetics.gap_matches_reliability(; seed = 103, noise_σ = 0.0,
        kind = :smoke)
    @test same.same
    @test same.rate_difference == 0.0
    @test same.parameters_equal
end

@testset "convergence is read off the training's own history" begin
    trained = HybridKinetics.gap_train(; seed = 103, noise_σ = 0.0, kind = :smoke)
    convergence = HybridKinetics.gap_convergence(trained)
    @test convergence.iterations == length(trained.training.history)
    @test convergence.final_gradient_norm ==
          trained.training.diagnostics.final_gradient_norm
    @test convergence.converged == trained.training.converged
end

@testset "module include and the study stays internal" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"OptimisationGapStudy.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "OptimisationGapStudy.jl"))
    for name in (:gap_train, :gap_discover, :gap_interpolate, :gap_alpha_sweep)
        @test !(name in names(HybridKinetics))
    end
end

@testset "restarts default to the single fit the package has always done" begin
    # The fingerprint suite records outputs of the current training defaults,
    # so the new option has to be inert until it is asked for.
    @test TrainingConfig().restarts == 1
    @test TrainingConfig(; restarts = 4).restarts == 4
    @test TrainingConfig(TrainingConfig(; restarts = 4)).restarts == 4
    @test TrainingConfig(TrainingConfig(); restarts = 7).restarts == 7
    @test_throws ArgumentError TrainingConfig(; restarts = 0)
    # the copy that lock_training_config makes must carry it through
    net = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    model, _ = build_ude_model(MersenneTwister(103), net)
    locked = HybridKinetics.lock_training_config(model, TrainingConfig(; restarts = 5))
    @test locked.restarts == 5
end

@testset "a restart redraws the network and keeps the physical guess" begin
    net = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    model, p0 = build_ude_model(MersenneTwister(103), net)
    names = Tuple(parameter_schema(model).phys_names)
    start = pack_parameters(
        NamedTuple{names}(ntuple(_ -> 0.8, length(names))), p0.nn)
    fresh = HybridKinetics._reinitialise_network(start, model, 7)
    @test fresh.phys == start.phys
    @test fresh.nn != start.nn
    @test length(fresh) == length(start)
    # the same seed gives the same draw, so a set of restarts is reproducible
    @test vec(HybridKinetics._reinitialise_network(start, model, 7)) == vec(fresh)
    @test vec(HybridKinetics._reinitialise_network(start, model, 8)) != vec(fresh)
end

@testset "restarts keep the lowest loss and report every attempt" begin
    net = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    truth_net = HybridKinetics.build_hill_recovery_network(; known = true, hill_order = 2)
    model, p0 = build_ude_model(MersenneTwister(103), net)
    names = Tuple(parameter_schema(model).phys_names)
    start = pack_parameters(
        NamedTuple{names}(ntuple(_ -> 0.8, length(names))), p0.nn)
    budget = HybridKinetics.LIBRARY_STUDY_TWO_STATE_BUDGET.smoke
    set = HybridKinetics.generate_recovery_experiments(
        MersenneTwister(103), truth_net, HybridKinetics.LIBRARY_STUDY_TWO_STATE_PARAMS;
        tspan = budget.tspan, n_points = budget.n_points, noise_σ = 0.0)
    split = HybridKinetics.reference_protocol_experiment_split(set)
    config = HybridKinetics.lock_training_config(model,
        HybridKinetics.reference_protocol_training_config(;
            adam_iterations = budget.adam_iterations,
            bfgs_iterations = budget.bfgs_iterations, model = model))
    once = HybridKinetics.train_experiments_with_warmup(start, split.train, model;
        config = config, verbose = false)
    thrice = HybridKinetics.train_experiments_with_warmup(start, split.train, model;
        config = TrainingConfig(config; restarts = 3), verbose = false)
    losses = thrice.metadata.config.restart_losses
    @test length(losses) == 3
    @test thrice.final_loss == minimum(losses)
    # restart one is the fit the package would have done on its own, so the
    # option can only ever improve on it
    @test losses[1] == once.final_loss
    @test thrice.final_loss ≤ once.final_loss
end

@testset "restarts reach the entry points that accept the configuration" begin
    # An option a caller can set and the code then ignores is worse than one
    # that is not there. Every entry point taking a TrainingConfig either acts
    # on restarts or says it cannot.
    net = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    truth_net = HybridKinetics.build_hill_recovery_network(; known = true, hill_order = 2)
    model, p0 = build_ude_model(MersenneTwister(103), net)
    names = Tuple(parameter_schema(model).phys_names)
    start = pack_parameters(
        NamedTuple{names}(ntuple(_ -> 0.8, length(names))), p0.nn)
    budget = HybridKinetics.LIBRARY_STUDY_TWO_STATE_BUDGET.smoke
    set = HybridKinetics.generate_recovery_experiments(
        MersenneTwister(103), truth_net, HybridKinetics.LIBRARY_STUDY_TWO_STATE_PARAMS;
        tspan = budget.tspan, n_points = budget.n_points, noise_σ = 0.0)
    split = HybridKinetics.reference_protocol_experiment_split(set)
    config = HybridKinetics.lock_training_config(model,
        HybridKinetics.reference_protocol_training_config(;
            adam_iterations = budget.adam_iterations,
            bfgs_iterations = budget.bfgs_iterations, model = model))

    once = train_experiments(start, split.train, model; config = config, verbose = false)
    twice = train_experiments(start, split.train, model;
        config = TrainingConfig(config; restarts = 2), verbose = false)
    losses = twice.metadata.config.restart_losses
    @test length(losses) == 2
    @test losses[1] == once.final_loss
    @test twice.final_loss == minimum(losses)

    # train_ude fits one experiment and has nothing to select on, so it says so
    # rather than accepting the option and dropping it
    first_exp = first(split.train.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    @test_throws ArgumentError train_ude(start, first_exp.observations,
        first_exp.times, first_exp.u0, tspan, model;
        config = TrainingConfig(config; restarts = 2), verbose = false)
end

@testset "discover_unknown_terms acts on the restarts it is given" begin
    # The headline entry point takes a TrainingConfig, so restarts has to reach
    # the fit it runs; and one restart has to leave that fit exactly as it was.
    net = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    truth_net = HybridKinetics.build_hill_recovery_network(; known = true, hill_order = 2)
    budget = HybridKinetics.LIBRARY_STUDY_TWO_STATE_BUDGET.smoke
    set = HybridKinetics.generate_recovery_experiments(
        MersenneTwister(103), truth_net, HybridKinetics.LIBRARY_STUDY_TWO_STATE_PARAMS;
        tspan = budget.tspan, n_points = budget.n_points, noise_σ = 0.0)
    training = TrainingConfig(adam_iterations = budget.adam_iterations,
        bfgs_iterations = budget.bfgs_iterations, log_every = 10^6)
    once = discover_unknown_terms(net, set; training = training, seed = 103,
        verbose = false)
    again = discover_unknown_terms(net, set;
        training = TrainingConfig(training; restarts = 1), seed = 103, verbose = false)
    @test again.training.final_loss == once.training.final_loss

    twice = discover_unknown_terms(net, set;
        training = TrainingConfig(training; restarts = 2), seed = 103, verbose = false)
    losses = twice.training.metadata.config.restart_losses
    @test length(losses) == 2
    @test losses[1] == once.training.final_loss
    @test twice.training.final_loss == minimum(losses)
    @test twice.training.final_loss ≤ once.training.final_loss
end
