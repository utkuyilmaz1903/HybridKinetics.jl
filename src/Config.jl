"""
    AbstractConstraintStrategy

Training constraint policy. `StructuralPositivity` relies on the compiled
`P,D ≥ 0` form; `AugmentedLagrangianConfig` adds a smooth penalty.
"""
abstract type AbstractConstraintStrategy end
abstract type AbstractDiscoveryBackend end

"""
    AbstractADPolicy

Adjoint / forward-pass policy for `SolverConfig`. Public subtypes are
`ZygoteAD` and `ProductionAD`.
"""
abstract type AbstractADPolicy end

"""
    StructuralPositivity

Default constraint: trust the compiled production–destruction form. No extra
penalty term is added to the training loss.
"""
struct StructuralPositivity <: AbstractConstraintStrategy end

"""
    AugmentedLagrangianConfig

Smooth inequality-constraint policy. `ρ` is updated automatically from
primal progress; users do not select a fixed penalty multiplier.
"""
Base.@kwdef struct AugmentedLagrangianConfig <: AbstractConstraintStrategy
    initial_ρ::Float64 = 1.0
    growth::Float64 = 5.0
    max_ρ::Float64 = 1e6
    tolerance::Float64 = 1e-6
    smoothness::Float64 = 1e-3
    outer_iterations::Int = 4
    progress_ratio::Float64 = 0.5
end

"""Compatibility AD backend using Zygote out-of-place sensitivities."""
struct ZygoteAD <: AbstractADPolicy end

"""
    ProductionAD

Production forward pass using the in-place compiled RHS (`ude_rhs!`) with a
preallocated cache when `sensealg = nothing`. Adjoint training continues to
use the out-of-place RHS because Zygote cannot differentiate through the
mutating cache path today.
"""
struct ProductionAD <: AbstractADPolicy end

function sensealg(policy::ZygoteAD)
    return InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true)
end

function sensealg(::ProductionAD)
    return InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true)
end

"""
    SolverConfig(; algorithm=Tsit5(), ad_policy=ZygoteAD(), sensealg, ...)

ODE integrator and adjoint settings used by `train_ude`, `predict_ude`, and
`SciMLBase.solve(::UDEModel, ...)`.
"""
struct SolverConfig{A, S, T <: AbstractFloat, P <: AbstractADPolicy}
    algorithm::A
    sensealg::S
    abstol::T
    reltol::T
    maxiters::Int
    ad_policy::P
end

function SolverConfig(; algorithm = Tsit5(),
        ad_policy::AbstractADPolicy = ZygoteAD(),
        sensealg = sensealg(ad_policy),
        abstol::Real = 1e-6, reltol::Real = 1e-6,
        maxiters::Int = 1_000_000)
    tolerance_type = promote_type(typeof(float(abstol)), typeof(float(reltol)))
    return SolverConfig(
        algorithm, sensealg, tolerance_type(abstol), tolerance_type(reltol),
        maxiters, ad_policy)
end

"""
    TrainingConfig

Adam then optional BFGS settings for `train_ude` / `train_experiments`.
`frozen_phys` names physical parameters whose raw values are held fixed
(gradient zeroed; restored after BFGS). Use this to pin a known production
rate; it does not remove `k_prod`↔`D(z)` Jacobian collinearity.
"""
struct TrainingConfig{T <: AbstractFloat, C <: AbstractConstraintStrategy, S, H}
    adam_iterations::Int
    adam_learning_rate::T
    bfgs_iterations::Int
    gradient_clip::T
    log_every::Int
    constraint::C
    solver::S
    horizon_schedule::H
    frozen_phys::Vector{Symbol}
    restarts::Int
end

"""
    HorizonCurriculum

Typed horizon curriculum for stable long-horizon UDE training. Pass as
`horizon_schedule` to `TrainingConfig` instead of a raw fraction vector.
"""
struct HorizonCurriculum{T <: AbstractFloat}
    fractions::Vector{T}
    min_points::Int
    minimum_fraction::T
end

function HorizonCurriculum(;
        fractions = [0.25, 0.5, 1.0],
        min_points::Int = 2,
        minimum_fraction = 0.05)
    T = promote_type(eltype(fractions), typeof(float(minimum_fraction)))
    return HorizonCurriculum{T}(T.(fractions), min_points, T(minimum_fraction))
end

function _horizon_fractions(schedule)
    schedule isa HorizonCurriculum && return schedule.fractions
    return schedule
end

function _horizon_min_points(schedule)
    schedule isa HorizonCurriculum && return schedule.min_points
    return 2
end

function _horizon_minimum_fraction(schedule)
    schedule isa HorizonCurriculum && return schedule.minimum_fraction
    return 0.05
end

"""
    SensealgRecommendation

Human-readable adjoint policy chosen by `recommend_sensealg`.

The wrapper type is intentionally unparameterized so
`recommend_sensealg` infers a single concrete return type even when
the payload adjoint switches between `BacksolveAdjoint` and
`InterpolatingAdjoint`.
"""
struct SensealgRecommendation
    sensealg::Any
    name::Symbol
    rationale::String
end

function TrainingConfig(; adam_iterations::Int = 300,
        adam_learning_rate::Real = 1e-2,
        bfgs_iterations::Int = 100,
        gradient_clip::Real = 10,
        log_every::Int = 20,
        constraint::AbstractConstraintStrategy =
        StructuralPositivity(),
        solver::SolverConfig = SolverConfig(),
        horizon_schedule = [0.25, 0.5, 1.0],
        frozen_phys::Vector{Symbol} = Symbol[],
        restarts::Int = 1)
    T = promote_type(typeof(float(adam_learning_rate)),
        typeof(float(gradient_clip)))
    resolved_schedule = if horizon_schedule isa HorizonCurriculum
        horizon_schedule
    else
        T.(collect(horizon_schedule))
    end
    restarts ≥ 1 || throw(ArgumentError("restarts must be at least 1; got $(restarts)"))
    return TrainingConfig(
        adam_iterations, T(adam_learning_rate), bfgs_iterations,
        T(gradient_clip), log_every, constraint, solver,
        resolved_schedule, copy(frozen_phys), restarts)
end

function TrainingConfig(base::TrainingConfig;
        adam_iterations = base.adam_iterations,
        adam_learning_rate = base.adam_learning_rate,
        bfgs_iterations = base.bfgs_iterations,
        gradient_clip = base.gradient_clip,
        log_every = base.log_every,
        constraint = base.constraint,
        solver = base.solver,
        horizon_schedule = base.horizon_schedule,
        frozen_phys = base.frozen_phys,
        restarts = base.restarts)
    return TrainingConfig(;
        adam_iterations, adam_learning_rate, bfgs_iterations, gradient_clip,
        log_every, constraint, solver, horizon_schedule,
        frozen_phys = copy(frozen_phys), restarts)
end

"""Explicit polynomial STLSQ backend (`dx/dt = Φ(x)ξ`)."""
Base.@kwdef struct ExplicitSTLSQ <: AbstractDiscoveryBackend
    threshold::Float64 = 1e-2
end

"""
    DataDrivenSparseSTLSQ

Optional explicit discovery backend that fits coefficients with
DataDrivenSparse.jl `STLSQ`. Requires `using DataDrivenSparse`. Graph-local
libraries and denominator safety remain HybridKinetics-owned.
"""
Base.@kwdef struct DataDrivenSparseSTLSQ <: AbstractDiscoveryBackend
    threshold::Float64 = 1e-2
    ridge::Float64 = 1e-10
end

"""
    ImplicitSINDyPI

Configuration for graph-local rational discovery through the implicit identity
`D(z)ẋ - N(z) = 0`.
"""
Base.@kwdef struct ImplicitSINDyPI <: AbstractDiscoveryBackend
    threshold::Float64 = 1e-2
    max_degree::Int = 3
    max_hill_degree::Int = 4
    max_parents::Int = 8
    extra_candidates::Int = 0
    bootstrap_samples::Int = 32
    validation_fraction::Float64 = 0.2
    denominator_floor::Float64 = 1e-8
    """Extra orthant stress-test samples for denominator safety (`0` disables)."""
    domain_samples::Int = 256
    """Row-block size for streaming library / blocked STLSQ."""
    chunk_size::Int = 256
end

"""
    StabilitySelection(; n_boot=100, τ=0.8, seed=7)

Optional pruning stage for implicit discovery, passed as
`stability_selection = StabilitySelection()` to `regress_unknown_rate` or
`discover_equations`. After the candidate has been fitted as usual, the
training rows of the regression are resampled with replacement `n_boot`
times, the thresholded fit is repeated on every resample, and a term of the
candidate is kept only if it is selected in at least a fraction `τ` of the
resamples; the coefficients of the kept terms are then refitted on the
training rows. Terms are never added, only removed, and the stage is skipped
when it would remove every numerator term or make the denominator unsafe.
The frequency of every library term is reported through
`stability_selection_report`. Off by default (`stability_selection =
nothing`), in which case discovery output is unchanged. `tau` is accepted as
an alias of `τ`. Cost: `n_boot` thresholded fits on the training rows.
"""
struct StabilitySelection
    n_boot::Int
    τ::Float64
    seed::UInt64
end

function StabilitySelection(; n_boot::Integer = 100, τ::Real = 0.8, tau::Real = τ,
        seed::Integer = 7)
    n_boot ≥ 1 || throw(ArgumentError("n_boot must be at least 1, got $n_boot"))
    0 < tau ≤ 1 || throw(ArgumentError("τ must be in (0, 1], got $tau"))
    return StabilitySelection(Int(n_boot), Float64(tau), UInt64(seed))
end

"""
    DiscoveryConfig(; backend=ImplicitSINDyPI(), basis_scope=:graph, ...)

Discovery options. `basis_scope=:graph` is the product prior; `:global` is
the internal ablation that adds every other dynamic node to the library.
"""
struct DiscoveryConfig{B <: AbstractDiscoveryBackend}
    backend::B
    include_constant::Bool
    include_interactions::Bool
    max_interaction_order::Int
    seed::UInt64
    basis_scope::Symbol
end

function DiscoveryConfig(; backend::AbstractDiscoveryBackend = ImplicitSINDyPI(),
        include_constant::Bool = true,
        include_interactions::Bool = true,
        max_interaction_order::Int = 2,
        seed::Integer = 42,
        basis_scope::Symbol = :graph)
    basis_scope === :graph || basis_scope === :global ||
        throw(ArgumentError("basis_scope must be :graph or :global, got $basis_scope"))
    return DiscoveryConfig(
        backend, include_constant, include_interactions,
        max_interaction_order, UInt64(seed), basis_scope)
end

"""
    ExecutionConfig

Experiment executor. `:gpu` is **experimental array transfer only** (CUDA `cu`
on experiment arrays). It is not a batched GPU ODE/training stack.
"""
Base.@kwdef struct ExecutionConfig
    backend::Symbol = :serial
    # Adam minibatch size. BFGS always refines the joint loss over `set`.
    batch_size::Int = 1
    deterministic::Bool = true
    checkpoint_every::Int = 50
end
