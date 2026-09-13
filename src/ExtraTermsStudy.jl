# Where the extra terms of the reference protocol come from (0.18, hypothesis A).
#
# The reference protocol recovers every true monomial of the Hill term
# (recall 1.0) but keeps a constant and a linear term, so the combined
# support F1 sits at about 0.57. This study measures four explanations on
# the same trained models, so that what varies is the discovery step and not
# the training:
#
#   :ablation    remove one accepted term at a time, refit the rest, and
#                record what the removal costs against the learned rate,
#                against the true rate, and on the held-out experiments;
#   :selection   sweep the sparsity threshold, record the whole front of
#                (term count, residual), and select by AIC, by BIC and by the
#                knee of the front;
#   :split       separate sparsity thresholds for the numerator and the
#                denominator;
#   :derivative  add rows to the regression that constrain the derivative of
#                the rate with respect to the regulator.
#
# All four run on the samples of one training, on the graph-local library and
# the reference discovery configuration, so the library is the numerator
# `1, R, R^2` and the denominator `R, R^2` (the denominator has no constant:
# `D(0) = 1` is the implicit normalisation of the implicit form). The true
# support is `R^2` in the numerator and `R^2` in the denominator, so three of
# the five fittable terms are false positives when all five survive.
#
# Not exported.

const EXTRA_TERMS_SEEDS = LIBRARY_STUDY_SEEDS
const EXTRA_TERMS_NOISE_LEVELS = LIBRARY_STUDY_NOISE_LEVELS
const EXTRA_TERMS_FIXTURES = (:two_state, :four_state)
const EXTRA_TERMS_HYPOTHESES = (:ablation, :selection, :split, :derivative)

"""Sparsity thresholds swept by the `:selection` and `:split` hypotheses."""
const EXTRA_TERMS_THRESHOLDS = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2,
    0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

const EXTRA_TERMS_COLUMNS = (:fixture, :seed, :noise, :hypothesis, :setting,
    :n_terms, :support_recall, :support_precision, :support_f1, :extras,
    :rss_train, :err_val, :rate_rmse_true, :holdout_residual,
    :denominator_min, :aic, :bic, :note, :run_time_s)

# -- Inputs --------------------------------------------------------------------

"""
The discovery inputs of the reference (`:reference`) variant for one trained
fixture: the permuted regulator samples, the learned rate, the graph-local
library specification, and the training / validation split that
`_discover_implicit` uses. Reproduces `regress_unknown_rate` exactly, so that
the hypotheses below vary only the fitting rule.
"""
function extra_terms_inputs(trained)
    rows = trained.parent_rows.true_parent
    R = Matrix{Float64}(trained.X[rows, :])
    D = Matrix{Float64}(trained.D)
    config = rate_discovery_config()
    nreg = size(R, 1)
    D_full = if size(D, 1) == nreg
        D
    else
        padded = zeros(Float64, nreg, size(R, 2))
        padded[1, :] .= vec(D)
        padded
    end
    R_perm, D_perm = _permute_rate_samples(R, D_full, config.seed)
    derivative = vec(D_perm[1, :])
    network = _rate_network_from_samples(R_perm)
    backend = config.backend
    spec = local_basis(
        network, 1;
        degree = max(backend.max_degree, backend.max_hill_degree),
        max_variables = backend.max_parents,
        include_interactions = config.include_interactions,
        X = R_perm, derivative = derivative,
        extra_candidates = backend.extra_candidates,
        scope = config.basis_scope)
    n = size(R_perm, 2)
    validation_count = clamp(round(Int, backend.validation_fraction * n), 1, n - 2)
    training_indices = collect(1:(n - validation_count))
    validation_indices = collect((n - validation_count + 1):n)
    return (; R = R_perm, derivative, spec, training_indices, validation_indices,
        backend, config, network, rows,
        truth_support = _library_study_truth(trained.truth, rows),
        truth = trained.truth)
end

# -- A self-contained implicit fit ---------------------------------------------
# The package's own streaming fit takes one threshold for the whole design.
# The hypotheses need per-block thresholds and extra rows, so the study fits
# the (tiny: 80 samples, 5 terms) dense design itself. `extra_terms_fit_matches_package`
# checks that this fit reproduces the package's fit at equal thresholds.

"""Dense implicit design `[Phi_num | -Phi_den .* y]` for samples `R` and rate `y`."""
function extra_terms_design(spec::LocalBasisSpec, R::AbstractMatrix, y::AbstractVector)
    n = size(R, 2)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    design = Matrix{Float64}(undef, n, n_num + n_den)
    for (j, term) in pairs(spec.numerator)
        design[:, j] .= evaluate_term(term, R)
    end
    for (j, term) in pairs(spec.denominator)
        design[:, n_num + j] .= .-(evaluate_term(term, R) .* y)
    end
    return design
end

"""Ridge-regularised least squares on the kept columns, as the package's STLSQ."""
function _extra_terms_solve(design::AbstractMatrix, y::AbstractVector,
        indices::AbstractVector{Int}; ridge::Real = 1.0e-10)
    coefficients = zeros(Float64, size(design, 2))
    isempty(indices) && return coefficients
    A = @view design[:, indices]
    gram = A' * A + ridge * I
    coefficients[indices] .= gram \ (A' * y)
    return coefficients
end

function _extra_terms_sweep(design, y, n_num, threshold_numerator,
        threshold_denominator, iterations, ridge)
    k = size(design, 2)
    keep = trues(k)
    coefficients = zeros(Float64, k)
    for _ in 1:iterations
        indices = findall(keep)
        isempty(indices) && break
        coefficients = _extra_terms_solve(design, y, indices; ridge)
        next = BitVector(undef, k)
        for j in 1:k
            threshold = j ≤ n_num ? threshold_numerator : threshold_denominator
            next[j] = abs(coefficients[j]) ≥ threshold
        end
        next == keep && break
        keep = next
    end
    return _extra_terms_solve(design, y, findall(keep); ridge)
end

"""
Sequentially thresholded least squares with one threshold per block, followed
by the prediction-reweighted second pass the package's own fit performs: the
design is rebuilt with the fitted rate in place of the measured one and
solved against the measured rate again, which removes most of the bias the
implicit form's error-in-the-denominator introduces.
"""
function extra_terms_stlsq(design::AbstractMatrix, y::AbstractVector, n_num::Int,
        threshold_numerator::Real, threshold_denominator::Real;
        iterations::Int = 20, ridge::Real = 1.0e-10, spec = nothing,
        R = nothing, reweight::Bool = true)
    coefficients = _extra_terms_sweep(design, y, n_num, threshold_numerator,
        threshold_denominator, iterations, ridge)
    numerator = coefficients[1:n_num]
    denominator = coefficients[(n_num + 1):end]
    (reweight && spec !== nothing && R !== nothing) || return numerator, denominator
    predicted, _ = _evaluate_candidate(spec, numerator, denominator, R)
    all(isfinite, predicted) || return numerator, denominator
    reweighted = extra_terms_design(spec, R, collect(predicted))
    coefficients = _extra_terms_sweep(reweighted, y, n_num, threshold_numerator,
        threshold_denominator, iterations, ridge)
    return coefficients[1:n_num], coefficients[(n_num + 1):end]
end

"""
Refit the coefficients of a fixed support, with the same prediction-reweighted
second pass. No thresholding: the support is given.
"""
function extra_terms_refit(design::AbstractMatrix, y::AbstractVector, n_num::Int,
        keep::AbstractVector{Bool}; ridge::Real = 1.0e-10, spec = nothing,
        R = nothing, reweight::Bool = true)
    indices = findall(keep)
    coefficients = _extra_terms_solve(design, y, indices; ridge)
    numerator = coefficients[1:n_num]
    denominator = coefficients[(n_num + 1):end]
    (reweight && spec !== nothing && R !== nothing) || return numerator, denominator
    predicted, _ = _evaluate_candidate(spec, numerator, denominator, R)
    all(isfinite, predicted) || return numerator, denominator
    reweighted = extra_terms_design(spec, R, collect(predicted))
    coefficients = _extra_terms_solve(reweighted, y, indices; ridge)
    return coefficients[1:n_num], coefficients[(n_num + 1):end]
end

# -- Scoring -------------------------------------------------------------------

"""
Every score of one fitted candidate: its support against the truth, its
residual against the learned rate on the training rows and on the held-out
validation rows, the relative error of the rational rate against the true
Hill rate on all samples, and the smallest value its denominator takes.
"""
function extra_terms_scores(inputs, numerator, denominator; trained = nothing)
    spec = inputs.spec
    candidate = ImplicitCandidate(1, spec, numerator, denominator,
        zeros(Float64, length(numerator) + length(denominator)), 0.0, 0.0)
    scores = support_f1(candidate, inputs.truth_support.numerator,
        inputs.truth_support.denominator)
    recovered = active_support(candidate)
    truth_keys = union(inputs.truth_support.numerator, inputs.truth_support.denominator)
    extras = String[]
    for key in sort!(
        collect(union(recovered.numerator, recovered.denominator)); by = string)
        key in truth_keys && continue
        push!(extras, _extra_terms_key_label(key))
    end
    train_R = @view inputs.R[:, inputs.training_indices]
    val_R = @view inputs.R[:, inputs.validation_indices]
    train_y = @view inputs.derivative[inputs.training_indices]
    val_y = @view inputs.derivative[inputs.validation_indices]
    rss_train, _, den_train = _implicit_rss(spec, numerator, denominator,
        Matrix(train_R), collect(train_y))
    err_val, _, den_val = _implicit_rss(spec, numerator, denominator,
        Matrix(val_R), collect(val_y))
    predicted, den_all = _evaluate_candidate(spec, numerator, denominator, inputs.R)
    r = vec(inputs.R[1, :])
    true_rate = hill_rate_truth(r; vmax = inputs.truth.vmax, K = inputs.truth.K,
        n = inputs.truth.n)
    rate_rmse_true = all(isfinite, predicted) ?
                     rate_rel_rmse(predicted, true_rate) : Inf
    denominator_min = isempty(den_all) ? 1.0 :
                      minimum(vcat(collect(den_all), collect(den_train), collect(den_val)))
    n_terms = count(!=(0.0), numerator) + count(!=(0.0), denominator)
    n_train = length(inputs.training_indices)
    aic, bic = if isfinite(rss_train) && rss_train > 0
        (n_train * log(rss_train) + 2 * n_terms,
            n_train * log(rss_train) + n_terms * log(n_train))
    else
        (Inf, Inf)
    end
    holdout = if trained === nothing || !all(isfinite, predicted)
        NaN
    else
        extra_terms_holdout_residual(trained, inputs, numerator, denominator)
    end
    return (; n_terms, candidate,
        support_recall = scores.combined.recall,
        support_precision = scores.combined.precision,
        support_f1 = scores.combined.f1,
        extras = join(extras, ";"),
        rss_train, err_val, rate_rmse_true, denominator_min, aic, bic,
        holdout_residual = holdout)
end

function _extra_terms_key_label(key)
    variables, powers = key
    isempty(variables) && return "1"
    parts = String[]
    for (variable, power) in zip(variables, powers)
        push!(parts, power == 1 ? "R" : "R^$(power)")
    end
    return join(parts, "*")
end

"""Mean hybrid residual on the held-out experiments with this rational rate."""
function extra_terms_holdout_residual(trained, inputs, numerator, denominator)
    candidate = ImplicitCandidate(1, inputs.spec, numerator, denominator,
        zeros(Float64, length(numerator) + length(denominator)), 0.0, 0.0)
    rate_fn = try
        equation_to_function(candidate)
    catch
        return NaN
    end
    residuals = Float64[]
    for experiment in trained.holdout_set.experiments
        value = try
            _library_study_residual(trained.model, trained.params, trained.term,
                rate_fn, inputs.rows, experiment)
        catch
            Inf
        end
        push!(residuals, value)
    end
    return isempty(residuals) ? NaN : mean(residuals)
end

# -- Hypotheses ----------------------------------------------------------------

"""
`:ablation`. Fit the reference support, then remove one accepted term at a
time and refit the remaining coefficients on the same rows. A term whose
removal costs nothing against the learned rate is doing no work; a term whose
removal improves the error against the true rate was describing something the
network got wrong rather than the mechanism.
"""
function extra_terms_ablation(trained, inputs; push_row!)
    spec = inputs.spec
    n_num = length(spec.numerator)
    train_R = Matrix(inputs.R[:, inputs.training_indices])
    train_y = collect(inputs.derivative[inputs.training_indices])
    design = extra_terms_design(spec, train_R, train_y)
    threshold = inputs.backend.threshold
    numerator, denominator = extra_terms_stlsq(design, train_y, n_num, threshold,
        threshold; spec, R = train_R)
    base = extra_terms_scores(inputs, numerator, denominator; trained)
    push_row!(_extra_terms_row(trained, inputs, :ablation, "full", base,
        "the fitted support, nothing removed"))
    keep = vcat([c != 0.0 for c in numerator], [c != 0.0 for c in denominator])
    labels = vcat(["numerator $(term.label)" for term in spec.numerator],
        ["denominator $(term.label)" for term in spec.denominator])
    for j in eachindex(keep)
        keep[j] || continue
        dropped = copy(keep)
        dropped[j] = false
        num_j, den_j = extra_terms_refit(design, train_y, n_num, dropped;
            spec, R = train_R)
        scores = extra_terms_scores(inputs, num_j, den_j; trained)
        is_true = _extra_terms_is_true_term(inputs, spec, j, n_num)
        push_row!(_extra_terms_row(trained, inputs, :ablation,
            "drop $(labels[j])", scores,
            is_true ? "a true term" : "an extra term"))
    end
    return base
end

function _extra_terms_is_true_term(inputs, spec, j, n_num)
    if j ≤ n_num
        return term_key(spec.numerator[j]) in inputs.truth_support.numerator
    end
    return term_key(spec.denominator[j - n_num]) in inputs.truth_support.denominator
end

"""
`:selection`. Sweep the sparsity threshold, record the whole front of
(term count, training residual), and select by AIC, by BIC and by the knee of
the front. The front is recorded so the docs can show it, not only the winner.
"""
function extra_terms_selection(trained, inputs; push_row!)
    spec = inputs.spec
    n_num = length(spec.numerator)
    train_R = Matrix(inputs.R[:, inputs.training_indices])
    train_y = collect(inputs.derivative[inputs.training_indices])
    design = extra_terms_design(spec, train_R, train_y)
    front = NamedTuple[]
    for threshold in EXTRA_TERMS_THRESHOLDS
        numerator, denominator = extra_terms_stlsq(
            design, train_y, n_num, threshold, threshold; spec, R = train_R)
        scores = extra_terms_scores(inputs, numerator, denominator; trained)
        push_row!(_extra_terms_row(trained, inputs, :selection,
            "threshold $(threshold)", scores, "sweep point"))
        push!(front, (; threshold, numerator, denominator, scores))
    end
    # One point per distinct term count: the lowest training residual reached.
    best = Dict{Int, NamedTuple}()
    for point in front
        k = point.scores.n_terms
        if !haskey(best, k) || point.scores.rss_train < best[k].scores.rss_train
            best[k] = point
        end
    end
    finite = [p
              for p in values(best)
              if isfinite(p.scores.rss_train) && p.scores.rss_train > 0]
    isempty(finite) && return front
    by_aic = argmin([p.scores.aic for p in finite])
    by_bic = argmin([p.scores.bic for p in finite])
    knee = _extra_terms_knee(finite)
    for (rule, point) in (("AIC", finite[by_aic]), ("BIC", finite[by_bic]),
        ("knee", finite[knee]))
        push_row!(_extra_terms_row(trained, inputs, :selection,
            "select $(rule)", point.scores,
            "selected at threshold $(point.threshold)"))
    end
    return front
end

"""Knee of the (term count, log residual) front: farthest point from the chord."""
function _extra_terms_knee(points)
    length(points) == 1 && return 1
    ks = Float64[p.scores.n_terms for p in points]
    rs = Float64[log(p.scores.rss_train) for p in points]
    order = sortperm(ks)
    ks, rs = ks[order], rs[order]
    k1, r1, k2, r2 = ks[1], rs[1], ks[end], rs[end]
    span = hypot(k2 - k1, r2 - r1)
    span == 0 && return order[1]
    distances = [abs((k2 - k1) * (r1 - rs[i]) - (k1 - ks[i]) * (r2 - r1)) / span
                 for i in eachindex(ks)]
    return order[argmax(distances)]
end

"""
`:split`. Separate sparsity thresholds for the numerator and the denominator.
The denominator carries no constant (it is pinned by the normalisation), so
the sub-case "do not sparsify the denominator's normalisation term" does not
arise; that is recorded rather than tested.
"""
function extra_terms_split(trained, inputs; push_row!)
    spec = inputs.spec
    n_num = length(spec.numerator)
    train_R = Matrix(inputs.R[:, inputs.training_indices])
    train_y = collect(inputs.derivative[inputs.training_indices])
    design = extra_terms_design(spec, train_R, train_y)
    grid = [1.0e-3, 1.0e-2, 0.05, 0.1, 0.3, 0.5, 1.0]
    for threshold_numerator in grid, threshold_denominator in grid
        numerator, denominator = extra_terms_stlsq(
            design, train_y, n_num, threshold_numerator, threshold_denominator;
            spec, R = train_R)
        scores = extra_terms_scores(inputs, numerator, denominator; trained)
        push_row!(_extra_terms_row(trained, inputs, :split,
            "num $(threshold_numerator) den $(threshold_denominator)", scores,
            "grid point"))
    end
    return nothing
end

"""
`:derivative`. Add rows constraining the derivative of the rate with respect
to the regulator. Differentiating the implicit identity
`N(r) - D(r) * (1 + Den(r)) = 0` gives another set of rows that is linear in
the same coefficients:
`Phi_num'(r) * a - (D'(r) * Phi_den(r) + D(r) * Phi_den'(r)) * b = D'(r)`.
The derivative of the learned rate is estimated by central differences on the
samples sorted by the regulator, so the study costs no extra samples and no
extra training; the derivative rows are scaled so that the two right-hand
sides have equal norm, and the scale is recorded.
"""
function extra_terms_derivative(trained, inputs; push_row!)
    spec = inputs.spec
    size(inputs.R, 1) == 1 || return nothing
    n_num = length(spec.numerator)
    train_R = Matrix(inputs.R[:, inputs.training_indices])
    train_y = collect(inputs.derivative[inputs.training_indices])
    design = extra_terms_design(spec, train_R, train_y)
    r = vec(train_R[1, :])
    order = sortperm(r)
    r_sorted = r[order]
    y_sorted = train_y[order]
    dy = _extra_terms_central_difference(r_sorted, y_sorted)
    n = length(r_sorted)
    sorted_R = reshape(r_sorted, 1, n)
    derivative_design = Matrix{Float64}(undef, n, size(design, 2))
    for (j, term) in pairs(spec.numerator)
        derivative_design[:, j] .= _extra_terms_monomial_derivative(term, r_sorted)
    end
    for (j, term) in pairs(spec.denominator)
        value = evaluate_term(term, sorted_R)
        slope = _extra_terms_monomial_derivative(term, r_sorted)
        derivative_design[:, n_num + j] .= .-(dy .* value .+ y_sorted .* slope)
    end
    scale = norm(dy) == 0 ? 1.0 : norm(train_y) / norm(dy)
    augmented = vcat(design, scale .* derivative_design)
    augmented_y = vcat(train_y, scale .* dy)
    for threshold in (inputs.backend.threshold, 1.0e-2, 0.1)
        numerator, denominator = extra_terms_stlsq(
            augmented, augmented_y, n_num, threshold, threshold; reweight = false)
        scores = extra_terms_scores(inputs, numerator, denominator; trained)
        push_row!(_extra_terms_row(trained, inputs, :derivative,
            "threshold $(threshold)", scores,
            "derivative rows, scale $(round(scale; digits = 4))"))
    end
    return nothing
end

function _extra_terms_central_difference(x::AbstractVector, y::AbstractVector)
    n = length(x)
    d = similar(y)
    n == 1 && return fill!(d, 0.0)
    d[1] = (y[2] - y[1]) / (x[2] - x[1])
    d[n] = (y[n] - y[n - 1]) / (x[n] - x[n - 1])
    for i in 2:(n - 1)
        d[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    end
    return d
end

function _extra_terms_monomial_derivative(term::MonomialTerm, r::AbstractVector)
    isempty(term.variables) && return zeros(Float64, length(r))
    power = only(term.powers)
    power == 1 && return ones(Float64, length(r))
    return power .* r .^ (power - 1)
end

# -- Rows and the study driver -------------------------------------------------

function _extra_terms_row(trained, inputs, hypothesis, setting, scores, note)
    return (; fixture = trained.fixture, seed = trained.seed, noise = trained.noise,
        hypothesis, setting, scores.n_terms,
        scores.support_recall, scores.support_precision, scores.support_f1,
        scores.extras, scores.rss_train, scores.err_val, scores.rate_rmse_true,
        scores.holdout_residual, scores.denominator_min, scores.aic, scores.bic,
        note, run_time_s = 0.0)
end

"""
The four hypotheses on one trained fixture at one seed and noise level. The
training happens once; every hypothesis runs on its samples.
"""
function extra_terms_run(; fixture::Symbol, seed::Integer, noise_σ::Real,
        hypotheses = EXTRA_TERMS_HYPOTHESES, kind::Symbol = :protocol, on_row = nothing)
    started = time()
    trained_raw = _library_study_train(
        fixture, seed, noise_σ, kind, fit_unknown_destruction,
        LIBRARY_STUDY_HOLDOUT_ICS, nothing, library_study_default_design(fixture))
    trained = merge(trained_raw, (; seed = Int(seed), noise = Float64(noise_σ)))
    inputs = extra_terms_inputs(trained)
    rows = NamedTuple[]
    push_row! = row -> begin
        stamped = merge(row, (; run_time_s = round(time() - started; digits = 1)))
        push!(rows, stamped)
        on_row === nothing || on_row(stamped)
        stamped
    end
    :ablation in hypotheses && extra_terms_ablation(trained, inputs; push_row!)
    :selection in hypotheses && extra_terms_selection(trained, inputs; push_row!)
    :split in hypotheses && extra_terms_split(trained, inputs; push_row!)
    :derivative in hypotheses && extra_terms_derivative(trained, inputs; push_row!)
    return rows
end

"""
Does the study's own dense fit reproduce the package's streaming fit at the
same threshold? Guards the comparison: every hypothesis below is measured
against a baseline produced by this fit.
"""
function extra_terms_fit_matches_package(inputs; atol::Real = 1.0e-8)
    spec = inputs.spec
    n_num = length(spec.numerator)
    train_R = Matrix(inputs.R[:, inputs.training_indices])
    train_y = collect(inputs.derivative[inputs.training_indices])
    design = extra_terms_design(spec, train_R, train_y)
    threshold = inputs.backend.threshold
    own_num, own_den = extra_terms_stlsq(design, train_y, n_num, threshold, threshold;
        spec, R = train_R)
    package_num, package_den = _fit_implicit(
        spec, inputs.R, inputs.derivative, inputs.training_indices, threshold;
        chunk_size = inputs.backend.chunk_size)
    same_support = ([c != 0.0 for c in own_num] == [c != 0.0 for c in package_num]) &&
                   ([c != 0.0 for c in own_den] == [c != 0.0 for c in package_den])
    close = same_support &&
            all(isapprox(a, b; atol = atol, rtol = 1.0e-6)
    for (a, b) in zip(vcat(own_num, own_den), vcat(package_num, package_den)))
    return (; same_support, close, own = (own_num, own_den),
        package = (package_num, package_den))
end

# -- CSV -----------------------------------------------------------------------

function extra_terms_csv_line(row)
    values = map(EXTRA_TERMS_COLUMNS) do column
        value = getproperty(row, column)
        # commas would break the row, so text fields carry none
        value isa AbstractString || value isa Symbol ? replace(string(value), "," => ";") :
        value isa Bool ? string(value) :
        value isa Integer ? string(value) : string(Float64(value))
    end
    return join(values, ",")
end

function append_extra_terms_row(path::AbstractString, row)
    mkpath(dirname(path))
    fresh = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        fresh && println(io, join(string.(EXTRA_TERMS_COLUMNS), ","))
        println(io, extra_terms_csv_line(row))
    end
    return path
end

function append_extra_terms_rows(path::AbstractString, rows)
    for row in rows
        append_extra_terms_row(path, row)
    end
    return path
end

function read_extra_terms_csv(path::AbstractString)
    isfile(path) || return NamedTuple[]
    lines = readlines(path)
    length(lines) ≤ 1 && return NamedTuple[]
    header = Symbol.(split(lines[1], ","))
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(line, ","; limit = length(header))
        length(fields) == length(header) || continue
        values = map(zip(header, fields)) do (column, text)
            column in (:fixture, :hypothesis, :setting, :extras, :note) ? String(text) :
            column === :seed ? parse(Int, text) :
            column === :n_terms ? parse(Int, text) :
            parse(Float64, text)
        end
        push!(rows, NamedTuple{Tuple(header)}(Tuple(values)))
    end
    return rows
end

"""Cells already in the CSV, so a rerun resumes where it stopped."""
function extra_terms_done_keys(path::AbstractString)
    keys = Set{Tuple{String, Int, Float64}}()
    for row in read_extra_terms_csv(path)
        push!(keys, (row.fixture, row.seed, row.noise))
    end
    return keys
end

"""
The whole study: every fixture, seed and noise level, appending rows as they
finish and skipping the cells already in the CSV.
"""
function extra_terms_study(; fixtures = EXTRA_TERMS_FIXTURES,
        seeds = EXTRA_TERMS_SEEDS, noise_levels = EXTRA_TERMS_NOISE_LEVELS,
        hypotheses = EXTRA_TERMS_HYPOTHESES, kind::Symbol = :protocol,
        out::AbstractString = joinpath(@__DIR__, "..", "benchmark", "results",
            "extra_terms_study.csv"),
        verbose::Bool = true)
    done = extra_terms_done_keys(out)
    for fixture in fixtures, seed in seeds, noise_σ in noise_levels
        key = (String(fixture), Int(seed), Float64(noise_σ))
        if key in done
            verbose &&
                println("skip $(fixture) seed $(seed) noise $(noise_σ) (already in the CSV)")
            continue
        end
        verbose && println("run  $(fixture) seed $(seed) noise $(noise_σ)")
        flush(stdout)
        rows = extra_terms_run(; fixture, seed, noise_σ, hypotheses, kind,
            on_row = row -> append_extra_terms_row(out, row))
        verbose && println("  $(length(rows)) rows")
        flush(stdout)
    end
    return out
end

# -- Summaries -----------------------------------------------------------------
# Each helper answers one of the milestone's pre-registered criteria directly,
# so that the verdict of a hypothesis is read off the rows rather than argued.

_extra_terms_cells(rows) = unique((r.fixture, r.seed, r.noise) for r in rows)

"""
`:ablation` verdict. For every run and every removed extra term: the factor by
which the training residual against the learned rate grows, and the change in
the error against the true rate. The pre-registered confirmation is a residual
growth of at least ten times with the true-rate error unchanged or improved
(within 5 per cent).
"""
function extra_terms_ablation_summary(rows)
    out = NamedTuple[]
    for cell in _extra_terms_cells(rows)
        cell_rows = [r
                     for r in rows
                     if (r.fixture, r.seed, r.noise) == cell &&
                        r.hypothesis == "ablation"]
        base = findfirst(r -> r.setting == "full", cell_rows)
        base === nothing && continue
        full = cell_rows[base]
        for row in cell_rows
            row.setting == "full" && continue
            is_extra = row.note == "an extra term"
            residual_factor = full.rss_train > 0 ? row.rss_train / full.rss_train : Inf
            true_ratio = full.rate_rmse_true > 0 ?
                         row.rate_rmse_true / full.rate_rmse_true : Inf
            push!(out,
                (; fixture = cell[1], seed = cell[2], noise = cell[3],
                    term = row.setting, is_extra, residual_factor, true_ratio,
                    holdout_ratio = full.holdout_residual > 0 ?
                                    row.holdout_residual / full.holdout_residual : NaN,
                    confirms = is_extra && residual_factor ≥ 10 && true_ratio ≤ 1.05))
        end
    end
    return out
end

"""
`:selection` verdict. For each run and each selection rule, whether the rule
recovered the true support exactly (F1 = 1.0) while keeping recall 1.0.
"""
function extra_terms_selection_summary(rows)
    out = NamedTuple[]
    for cell in _extra_terms_cells(rows)
        for rule in ("AIC", "BIC", "knee")
            row = findfirst(
                r -> (r.fixture, r.seed, r.noise) == cell &&
                         r.hypothesis == "selection" &&
                         r.setting == "select $(rule)",
                rows)
            row === nothing && continue
            selected = rows[row]
            push!(out,
                (; fixture = cell[1], seed = cell[2], noise = cell[3], rule,
                    selected.n_terms, selected.support_f1, selected.support_recall,
                    selected.extras, selected.rate_rmse_true,
                    confirms = selected.support_f1 ≈ 1.0 && selected.support_recall ≈ 1.0))
        end
    end
    return out
end

"""
`:split` verdict. The (numerator, denominator) threshold pairs that recover the
true support, counted over runs; a pair only counts as a rule if it is the same
pair across fixtures and noise levels.
"""
function extra_terms_split_summary(rows)
    hits = Dict{String, Vector{Tuple{String, Int, Float64}}}()
    cells = _extra_terms_cells(rows)
    for row in rows
        row.hypothesis == "split" || continue
        (row.support_f1 ≈ 1.0 && row.support_recall ≈ 1.0) || continue
        push!(get!(hits, row.setting, Tuple{String, Int, Float64}[]),
            (row.fixture, row.seed, row.noise))
    end
    out = NamedTuple[]
    for (setting, cells_hit) in hits
        fixtures = unique(first.(cells_hit))
        push!(out,
            (; setting, n_hits = length(cells_hit), n_cells = length(cells),
                fixtures = join(sort(fixtures), "+"),
                confirms = length(cells_hit) ≥ 14 && length(fixtures) ≥ 2))
    end
    sort!(out; by = r -> -r.n_hits)
    return out
end

"""`:derivative` verdict: recovery of the true support per threshold."""
function extra_terms_derivative_summary(rows)
    out = NamedTuple[]
    settings = unique(r.setting for r in rows if r.hypothesis == "derivative")
    cells = _extra_terms_cells(rows)
    for setting in settings
        matching = [r for r in rows if r.hypothesis == "derivative" && r.setting == setting]
        recovered = count(r -> r.support_f1 ≈ 1.0 && r.support_recall ≈ 1.0, matching)
        full_recall = count(r -> r.support_recall ≈ 1.0, matching)
        push!(out,
            (; setting, n = length(matching), recovered, full_recall,
                median_f1 = isempty(matching) ? NaN :
                            median([r.support_f1 for r in matching]),
                confirms = recovered ≥ 12))
    end
    return out
end

"""One printable block per hypothesis, read straight off the rows."""
function format_extra_terms_summary(rows)
    isempty(rows) && return "no rows"
    io = IOBuffer()
    cells = _extra_terms_cells(rows)
    println(io, "cells: $(length(cells)) (fixture, seed, noise)")
    ablation = extra_terms_ablation_summary(rows)
    extras_only = [a for a in ablation if a.is_extra]
    true_only = [a for a in ablation if !a.is_extra]
    println(io, "\n--- ablation: what removing one term costs ---")
    println(io,
        "| term kind | n | median residual growth | median true-rate error ratio | median held-out ratio |")
    println(io, "|---|---|---|---|---|")
    for (label, group) in (("extra", extras_only), ("true", true_only))
        isempty(group) && continue
        finite = [g for g in group if isfinite(g.residual_factor)]
        println(io, "| $(label) | $(length(group)) | ",
            isempty(finite) ? "NA" :
            string(round(median([g.residual_factor for g in finite]); sigdigits = 3)), " | ",
            string(round(median([g.true_ratio for g in group if isfinite(g.true_ratio)]);
                digits = 3)), " | ",
            string(round(
                median([g.holdout_ratio for g in group if isfinite(g.holdout_ratio)]);
                digits = 3)), " |")
    end
    println(io,
        "pre-registered confirmation (extra term, residual x10 or more, true-rate error within 5%): ",
        count(a -> a.confirms, extras_only), " of ", length(extras_only), " removals")
    println(io, "\n--- selection: AIC, BIC, knee ---")
    println(io, "| rule | recovers the true support | median terms kept |")
    println(io, "|---|---|---|")
    selection = extra_terms_selection_summary(rows)
    for rule in ("AIC", "BIC", "knee")
        group = [s for s in selection if s.rule == rule]
        isempty(group) && continue
        println(io, "| $(rule) | $(count(s -> s.confirms, group)) of $(length(group)) | ",
            median([s.n_terms for s in group]), " |")
    end
    println(io, "\n--- split thresholds: pairs that recover the true support ---")
    split = extra_terms_split_summary(rows)
    if isempty(split)
        println(
            io, "no (numerator, denominator) pair recovered the true support in any run")
    else
        println(io, "| pair | runs recovered | of | fixtures |")
        println(io, "|---|---|---|---|")
        for row in first(split, 5)
            println(
                io, "| $(row.setting) | $(row.n_hits) | $(row.n_cells) | $(row.fixtures) |")
        end
    end
    println(io, "\n--- derivative rows ---")
    derivative = extra_terms_derivative_summary(rows)
    if isempty(derivative)
        println(io, "not run")
    else
        println(io,
            "| setting | runs | recover the true support | keep recall 1.0 | median F1 |")
        println(io, "|---|---|---|---|---|")
        for row in derivative
            println(io,
                "| $(row.setting) | $(row.n) | $(row.recovered) | $(row.full_recall) | ",
                round(row.median_f1; digits = 3), " |")
        end
    end
    return String(take!(io))
end
