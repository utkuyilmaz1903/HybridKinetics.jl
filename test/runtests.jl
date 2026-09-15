# Default test suite. Set HYBRIDKINETICS_TEST_HEAVY=1 to also run the multi-minute
# training-loop testsets; the trained-model recovery protocol lives in
# run_recovery_hard.jl and run_trained_library_comparison.jl and is not part of Pkg.test().
using HybridKinetics
include("internals.jl")
include(joinpath("support", "support.jl"))
using ComponentArrays
using LinearAlgebra
using Lux
using Optimization
using OrdinaryDiffEq: Tsit5, solve
using SciMLBase
using SciMLSensitivity: InterpolatingAdjoint, BacksolveAdjoint
using Random
using Statistics
using StaticArrays
using Test
using Zygote

LinearAlgebra.BLAS.set_num_threads(1)

include("test_baseline.jl")
include("test_network.jl")
include("test_ude.jl")
include("test_training.jl")
include("test_discovery.jl")
include("test_experiments.jl")
include("test_compiler.jl")
include("test_quality_checks.jl")
include("test_release.jl")
include("test_sciml_interface.jl")
include("test_metadata.jl")
include("test_explicit_discovery.jl")
include("test_phase1.jl")
include("test_phase2.jl")
include("test_phase3.jl")
include("test_phase4.jl")
include("test_recovery.jl")
include("test_recovery_pipeline.jl")
include("test_holdout.jl")
include("test_functional_identifiability.jl")
include("test_trajectory_occupancy.jl")
include("test_occupancy_separation.jl")
include("test_trained_library_comparison.jl")
include("test_library_comparison_study.jl")
include("test_extra_terms_study.jl")
include("test_bias_and_reliability_studies.jl")
include("test_optimisation_gap_study.jl")
include("test_stability_selection.jl")
include("test_discover_unknown_term.jl")
include("support/fingerprints_015.jl")
include("test_unknown_terms.jl")
include("test_catalyst_extension.jl")
include("test_reference_path.jl")
include("test_invariants.jl")
include("test_example_smoke.jl")
include("test_reference_protocol_product.jl")
include("test_protocol_fingerprint.jl")
include("test_compiler_spec.jl")
include("test_protocol_surface.jl")
include("test_datagen_spec.jl")
include("test_recovery_admission.jl")
include("test_discovery_workspace.jl")
include("test_training_reuse.jl")
include("test_recovery_suite_skip.jl")
include("test_experiment_checkpoint.jl")
include("test_failure_modes.jl")
include("test_hybrid_compose.jl")
include("test_hybrid_residual.jl")
include("test_identifiability_product.jl")
include("test_graph_local_library.jl")
include("test_denominator_domain.jl")
include("test_parameter_schema_pack.jl")
