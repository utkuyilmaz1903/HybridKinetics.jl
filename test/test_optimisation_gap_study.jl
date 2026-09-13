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
