@testset "the true support is labelled correctly" begin
    # The 0.18 audit of the support F1 metric, pinned as a test: the headline
    # number 0.571 is a statement about the learned rate, not an artefact of a
    # mislabelled truth or a ceiling the library cannot reach.
    network = HybridKinetics._rate_network_from_samples(reshape([0.5, 1.0, 1.5], 1, 3))
    config = rate_discovery_config()
    backend = config.backend
    spec = HybridKinetics.local_basis(
        network, 1;
        degree = max(backend.max_degree, backend.max_hill_degree),
        max_variables = backend.max_parents,
        include_interactions = config.include_interactions,
        scope = config.basis_scope)
    # The library of the reference protocol: a constant only in the numerator.
    @test [term.label for term in spec.numerator] == ["1", "x[1]", "x[1]^2"]
    @test [term.label for term in spec.denominator] == ["x[1]", "x[1]^2"]
    # `D(0) = 1` is the implicit normalisation, so the denominator carries no
    # constant to be fitted and none to be scored.
    @test !any(isempty(term.variables) for term in spec.denominator)
    @test any(isempty(term.variables) for term in spec.numerator)

    truth = hill_rate_support(2)
    @test truth.numerator == Set([((1,), (2,))])
    @test truth.denominator == Set([((1,), (2,))])

    # A candidate carrying exactly the true support, with the coefficients the
    # Hill law has after the normalisation is applied, IS the true law.
    vmax, K = 1.8, 0.55
    numerator = [0.0, 0.0, vmax / K^2]
    denominator = [0.0, 1 / K^2]
    r = collect(range(0.05, 2.0; length = 41))
    R = reshape(r, 1, length(r))
    predicted, _ = HybridKinetics._evaluate_candidate(spec, numerator, denominator, R)
    @test predicted≈hill_rate_truth(r; vmax = vmax, K = K, n = 2) atol=1e-12

    # ... and it scores a perfect support against the recorded truth, so the
    # metric's ceiling is attainable.
    exact = ImplicitCandidate(1, spec, numerator, denominator,
        zeros(Float64, 5), 0.0, 1.0)
    scores = support_f1(exact, truth.numerator, truth.denominator)
    @test scores.combined.f1 == 1.0
    @test scores.combined.recall == 1.0
    @test scores.combined.precision == 1.0

    # Keeping the whole library instead gives the published 0.571 exactly:
    # two true positives and three false positives, the constant and the
    # linear term of the numerator and the linear term of the denominator.
    full = ImplicitCandidate(1, spec, [0.3, -1.0, vmax / K^2], [-0.5, 1 / K^2],
        zeros(Float64, 5), 0.0, 1.0)
    full_scores = support_f1(full, truth.numerator, truth.denominator)
    @test full_scores.combined.recall == 1.0
    @test full_scores.combined.precision == 0.4
    @test full_scores.combined.f1 ≈ 4 / 7
end

@testset "the extra-terms study reproduces the package fit" begin
    # The study fits the dense design itself so that it can use per-block
    # thresholds and extra rows. It must agree with the package's own fit,
    # or the hypotheses would be measured against the wrong baseline.
    vmax, K = 1.8, 0.55
    r = collect(range(0.1, 2.0; length = 60))
    R = reshape(r, 1, length(r))
    D = reshape(hill_rate_truth(r; vmax = vmax, K = K, n = 2), 1, length(r))
    network = HybridKinetics._rate_network_from_samples(R)
    config = rate_discovery_config()
    backend = config.backend
    spec = HybridKinetics.local_basis(
        network, 1;
        degree = max(backend.max_degree, backend.max_hill_degree),
        max_variables = backend.max_parents,
        include_interactions = config.include_interactions,
        scope = config.basis_scope)
    indices = collect(1:48)
    train_R = Matrix(R[:, indices])
    train_y = collect(vec(D)[indices])
    design = HybridKinetics.extra_terms_design(spec, train_R, train_y)
    own_num, own_den = HybridKinetics.extra_terms_stlsq(
        design, train_y, length(spec.numerator), backend.threshold, backend.threshold;
        spec = spec, R = train_R)
    package_num, package_den = HybridKinetics._fit_implicit(
        spec, R, vec(D), indices, backend.threshold; chunk_size = backend.chunk_size)
    @test [c != 0.0 for c in own_num] == [c != 0.0 for c in package_num]
    @test [c != 0.0 for c in own_den] == [c != 0.0 for c in package_den]
    @test own_num≈package_num rtol=1e-6
    @test own_den≈package_den rtol=1e-6

    # On the analytic rate the fit recovers the true support, which is the
    # control the study's runs are compared against: the extras of a real run
    # come from the learned rate, not from the library or the fitting rule.
    truth = hill_rate_support(2)
    candidate = ImplicitCandidate(1, spec, own_num, own_den, zeros(Float64, 5), 0.0, 1.0)
    scores = support_f1(candidate, truth.numerator, truth.denominator)
    @test scores.combined.recall == 1.0
    @test scores.combined.f1 == 1.0
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"ExtraTermsStudy.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "ExtraTermsStudy.jl"))
    @test isfile(joinpath(@__DIR__, "..", "benchmark", "extra_terms_study.jl"))
    study = read(joinpath(@__DIR__, "..", "src", "ExtraTermsStudy.jl"), String)
    for name in ("extra_terms_ablation", "extra_terms_selection",
        "extra_terms_split", "extra_terms_derivative")
        @test occursin("function $(name)(", study)
    end
    # None of the study's names reach the public surface.
    for name in (:extra_terms_run, :extra_terms_study, :extra_terms_design)
        @test !(name in names(HybridKinetics))
    end
end
