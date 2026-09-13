# One-call entry point: discover_unknown_terms must run the same functions as
# the chained calls of examples/unknown_inhibition.jl, in the same order and
# with the same defaults, so its result matches the chain field by field.

const _DUT_TRUTH = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
const _DUT_CONFIG = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0,
    log_every = 10^6)

function _dut_fixture()
    truth_net = HybridKinetics.build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    set = HybridKinetics.reference_protocol_experiment_set(
        MersenneTwister(103), truth_net; smoke = true, truth_params = _DUT_TRUTH,
        initial_conditions = [[0.25, 0.20], [0.80, 0.35], [0.40, 1.10]])
    return ude_net, set
end

function _dut_chain(ude_net, set; rng_seed = 7)
    rng = MersenneTwister(rng_seed)
    model, params = build_ude_model(rng, ude_net)
    phys_names = Tuple(parameter_schema(model).phys_names)
    guess = NamedTuple{phys_names}(ntuple(_ -> 0.8, length(phys_names)))
    ude_init = pack_parameters(guess, params.nn)
    first_exp = first(set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    warm = train_ude(ude_init, first_exp.observations, first_exp.times, first_exp.u0,
        tspan, model;
        config = TrainingConfig(adam_iterations = _DUT_CONFIG.adam_iterations,
            bfgs_iterations = 0,
            horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0]),
            log_every = 10^6),
        verbose = false)
    trained = train_experiments(warm.params, set, model; config = _DUT_CONFIG,
        verbose = false)
    term = only(HybridKinetics.neural_destruction_terms(model))
    r_range = HybridKinetics._regulator_grid(set, term)
    R, D, term = HybridKinetics.sample_unknown_destruction_grid(
        model, trained.params, term;
        r_range = r_range)
    times_grid = collect(range(0.0, 1.0; length = size(R, 2)))
    discovery = regress_unknown_rate(R, times_grid, D;
        config = HybridKinetics.reference_protocol_discovery_config(), verbose = false,
        strict = false)
    ident = HybridKinetics.report_production_destruction_tradeoff(
        model, trained.params, first_exp.observations, first_exp.times,
        first_exp.u0, tspan; term = term, verbose = false)
    residual = Inf
    if discovery.success && !isempty(discovery.candidates)
        rate_fn = equation_to_function(discovery.candidates[1])
        residual = hybrid_data_residual(model, trained.params, term, rate_fn,
            first_exp.u0, tspan, first_exp.times, first_exp.observations)
    end
    return (; model, trained, term, R, D, discovery, ident, residual)
end

@testset "discover_unknown_terms" begin
    ude_net, set = _dut_fixture()

    @testset "matches the chained calls field by field" begin
        chain = _dut_chain(ude_net, set)
        result = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), verbose = false,
            known_support = HybridKinetics.hill_rate_support(2))
        @test result isa DiscoveryRun
        @test length(result) == 1 && keys(result) == [:S] && result[:S] === result[1]
        @test result[1] isa UnknownTermResult
        @test HybridKinetics.nn_parameter_fingerprint(result.params.nn) ==
              HybridKinetics.nn_parameter_fingerprint(chain.trained.params.nn)
        @test collect(result.params.phys) == collect(chain.trained.params.phys)
        @test result.training.final_loss == chain.trained.final_loss
        @test result[1].term === only(HybridKinetics.neural_destruction_terms(result.model))
        @test result[1].samples.R == chain.R
        @test result[1].samples.D == chain.D
        @test result[1].discovery.success == chain.discovery.success
        @test result[1].discovery.equations == chain.discovery.equations
        @test result[1].discovery.retcode == chain.discovery.retcode
        @test result[1].identifiability.unidentifiable_edge ==
              chain.ident.unidentifiable_edge
        @test result[1].identifiability.collinearity == chain.ident.collinearity
        @test isequal(result.residuals.data_residual, chain.residual)
        @test isnan(result.residuals.data_residual_holdout)
        @test result.training_indices == 1:3
        @test isempty(result.holdout_indices)
        if chain.discovery.success
            @test result[1].extras ==
                  HybridKinetics.reference_protocol_discovery_extras(chain.discovery.candidates[1])
            @test isfinite(result.residuals.data_residual_train)
        end
        @test result.settings.n_ics == 3
        @test result.settings.n_points == size(first(set.experiments).observations, 2)
        @test result.settings.adam_iters == 2
        @test result.settings.bfgs_iters == 0
        @test result.settings.bootstrap == HybridKinetics.REFERENCE_PROTOCOL.bootstrap
        @test result.settings.discovery_seed ==
              HybridKinetics.REFERENCE_PROTOCOL.discovery_seed
        @test result.settings.unknown_holes == 1
    end

    @testset "report and show" begin
        result = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 1, rng = MersenneTwister(7), verbose = false, seed = 103)
        text = report_unknown_terms(result)
        for section in ("IDENTIFIABILITY", "FIT", "DISCOVERY", "REPRODUCTION")
            @test occursin("\n" * section * "\n", "\n" * text)
        end
        @test occursin("hybrid_data_residual_train:", text)
        @test occursin("hybrid_data_residual_holdout:", text)
        @test occursin("seed: 103", text)
        @test occursin("n_ics: 3", text)
        @test occursin("extras: NA", text)
        @test sprint(show, MIME("text/plain"), result) == text
        summary = sprint(show, result)
        @test startswith(summary, "DiscoveryRun(")
        @test occursin("held out = 1", summary)
        @test result.training_indices == 1:2
        @test result.holdout_indices == [3]
        # The default report carries no held-out lines, so existing output is unchanged.
        plain = HybridKinetics.format_protocol_result(result[1].identifiability)
        @test !occursin("hybrid_data_residual_train", plain)
        @test !occursin("hybrid_data_residual_holdout", plain)
        captured_path = joinpath(mktempdir(), "verbose.txt")
        open(captured_path, "w") do io
            redirect_stdout(io) do
                discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
                    holdout = 1, rng = MersenneTwister(7), verbose = true)
            end
        end
        @test occursin("REPRODUCTION", read(captured_path, String))
    end

    @testset "regulator grid and observation masks" begin
        given = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), verbose = false,
            regulator_grid = range(0.2, 1.4; length = 24))
        @test vec(given[1].samples.R) == collect(range(0.2, 1.4; length = 24))
        @test given.settings.regulator_grid === :given
        called = Ref(0)
        by_function = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), verbose = false,
            regulator_grid = (model, params, train_set, term) -> begin
                called[] += 1
                @test model isa UDEModel && train_set isa ExperimentSet
                @test term === only(HybridKinetics.neural_destruction_terms(model))
                range(0.2, 1.4; length = 24)
            end)
        @test called[] == 1
        @test by_function.settings.regulator_grid === :function
        @test by_function[1].samples.R == given[1].samples.R
        @test by_function[1].samples.D == given[1].samples.D
        @test by_function[1].discovery.equations == given[1].discovery.equations
        default = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), verbose = false)
        @test default.settings.regulator_grid === :observed
        @test vec(default[1].samples.R) ==
              collect(HybridKinetics._regulator_grid(set, default[1].term))
        # An unobserved state: NaN observations are masked out of the loss, the
        # residuals, and the identifiability diagnostic.
        masked_experiments = map(set.experiments) do e
            observations = copy(e.observations)
            observations[2, :] .= NaN
            Experiment(e.name, e.times, observations, e.u0)
        end
        masked_set = ExperimentSet(masked_experiments, set.state_names)
        @test all(!any(e.mask[2, :]) && all(e.mask[1, :]) for e in masked_set.experiments)
        # The default regulator grid is set by the observed, finite regulator
        # values only: one NaN used to make its extrema, and the grid, NaN.
        hole = copy(first(set.experiments).observations)
        hole[2, 1] = NaN
        hole_exp = Experiment(first(set.experiments).name,
            first(set.experiments).times, hole, first(set.experiments).u0)
        hole_set = ExperimentSet([hole_exp], set.state_names)
        hole_grid = collect(HybridKinetics._regulator_grid(hole_set, default[1].term))
        @test all(isfinite, hole_grid)
        finite_values = filter(isfinite, hole[2, :])
        @test minimum(hole_grid) <= minimum(finite_values)
        @test maximum(hole_grid) >= maximum(finite_values)
        # A masked entry is ignored even when it is finite.
        cover_exp = Experiment(first(set.experiments).name, first(set.experiments).times,
            first(set.experiments).observations, first(set.experiments).u0;
            mask = hole_exp.mask)
        @test collect(HybridKinetics._regulator_grid(
            ExperimentSet([cover_exp],
                set.state_names), default[1].term)) == hole_grid
        # No observed regulator value at all is an error, not a NaN grid.
        @test_throws ArgumentError HybridKinetics._regulator_grid(
            masked_set, default[1].term)
        masked = discover_unknown_terms(ude_net, masked_set; training = _DUT_CONFIG,
            holdout = 1, rng = MersenneTwister(7), verbose = false,
            regulator_grid = range(0.2, 1.4; length = 24))
        # The warm-up honoured the mask, so the joint fit started from finite
        # parameters and a finite loss.
        @test isfinite(masked.training.initial_loss)
        @test isfinite(masked.training.final_loss)
        @test all(isfinite, masked.params)
        if masked[1].discovery.success
            @test isfinite(masked.residuals.data_residual)
            @test isfinite(masked.residuals.data_residual_train)
            @test isfinite(masked.residuals.data_residual_holdout)
        end
        @test !isnan(masked[1].identifiability.condition_number)
        # With every entry observed the mask changes nothing.
        full = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 1, rng = MersenneTwister(7), verbose = false)
        chain_residual = full[1].discovery.success ?
                         hybrid_data_residual(full.model, full.params, full[1].term,
            equation_to_function(full[1].discovery.candidates[1]),
            set.experiments[1].u0,
            (first(set.experiments[1].times), last(set.experiments[1].times)),
            set.experiments[1].times, set.experiments[1].observations) : Inf
        @test isequal(full.residuals.data_residual, chain_residual)
    end

    @testset "arguments" begin
        @test_throws ArgumentError discover_unknown_terms(ude_net, set;
            training = _DUT_CONFIG, holdout = 3, verbose = false)
        @test_throws ArgumentError discover_unknown_terms(ude_net, set;
            training = _DUT_CONFIG, holdout = -1, verbose = false)
        known = HybridKinetics.build_hill_recovery_network(; known = true, hill_order = 2)
        # zero unknown terms: an ArgumentError that says how to mark one (0.15 raised
        # the reference protocol's ErrorException from assert_single_unknown_destruction)
        @test_throws ArgumentError discover_unknown_terms(known, set;
            training = _DUT_CONFIG, verbose = false)
        @test occursin("no unknown destruction term",
            sprint(showerror,
                try
                    discover_unknown_terms(known, set; training = _DUT_CONFIG, verbose = false)
                catch e
                    e
                end))
        no_warm = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), warmup = false, verbose = false)
        warm = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), verbose = false)
        @test no_warm.settings.warmup == false
        @test HybridKinetics.nn_parameter_fingerprint(no_warm.params.nn) !=
              HybridKinetics.nn_parameter_fingerprint(warm.params.nn)
        # `terms` names the unknown terms (0.15's `term` index is gone); a spec on a
        # node that has no unknown term is refused
        indexed = discover_unknown_terms(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), terms = [UnknownTerm(:S)],
            verbose = false)
        @test indexed[1].discovery.equations == warm[1].discovery.equations
        @test_throws ArgumentError discover_unknown_terms(ude_net, set;
            training = _DUT_CONFIG, holdout = 0, terms = [UnknownTerm(:R)], verbose = false)
        @test_throws MethodError discover_unknown_terms(ude_net, set;
            training = _DUT_CONFIG, holdout = 0, term = 1, verbose = false)
    end
end
