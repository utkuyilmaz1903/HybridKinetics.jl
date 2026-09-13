###############################################################################
# HybridKinetics.jl — graph-guided biological UDEs with local rational discovery.
#
# Top-level module: brings in all submodule files (single shared namespace,
# the standard Julia package pattern) and curates the public API.
###############################################################################
module HybridKinetics

const PACKAGE_VERSION = v"0.17.2"

# -- External dependencies ----------------------------------------------------
using Dates
using DelimitedFiles
using Distributed
using Graphs
using LinearAlgebra
using Lux
using NNlib: sigmoid, softplus
using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity: InterpolatingAdjoint, BacksolveAdjoint, ZygoteVJP
using SciMLBase
using Optimization
using OptimizationOptimJL
using Optimisers
using PrecompileTools
using Zygote
import ChainRulesCore
using ChainRulesCore: ignore_derivatives
using Random
using Serialization
using SHA
using Statistics
using StaticArrays

# -- Source files --------------------------------------------------------------
include("Types.jl")
include("ScientificCore.jl")
include("Config.jl")
include("Metadata.jl")
include("Network.jl")
include("Experiments.jl")
include("UDE.jl")
include("ModelCache.jl")
include("ParameterSchema.jl")
include("MechanismCompiler.jl")
include("SciMLInterface.jl")
include("DataGen.jl")
include("Training.jl")
include("OptimizationInterface.jl")
include("BasisFactory.jl")
include("DiscoveryWorkspace.jl")
include("Discovery.jl")
include("Identifiability.jl")
include("BenchmarkNetworks.jl")
include("Recovery.jl")
include("ReferenceProtocol.jl")
include("CompilerSpec.jl")
include("DataGenSpec.jl")
include("RecoveryAdmission.jl")
include("TrainingReuse.jl")
include("RecoveryPipeline.jl")
include("FunctionalIdentifiability.jl")
include("TrajectoryOccupancy.jl")
include("RecoverySuiteSkip.jl")
include("ExperimentCheckpoint.jl")
include("FailureModes.jl")
include("HybridCompose.jl")
include("HybridResidual.jl")
include("IdentifiabilityProduct.jl")
include("GraphLocalLibrary.jl")
include("TrainedGraphLocal.jl")
include("LibraryComparisonStudy.jl")
include("DiscoverUnknownTerm.jl")
include("ExtraTermsStudy.jl")
include("ReliabilityStudy.jl")
include("MultiTermStudy.jl")
include("AdjacentBiasStudy.jl")
include("DenominatorDomain.jl")
include("ParameterSchemaPack.jl")
include("Bridge.jl")
include("Execution.jl")
include("Visualization.jl")
include("Precompile.jl")

# -- Public API ---------------------------------------------------------------
# Freeze list + reference-example verbs. Fixtures, Fisher, GPU/SBML/MTK, and
# library internals are `HybridKinetics.foo` (not exported).
export BiologicalNetwork, NodeSpec, EdgeSpec, ReactionSpec,
       EdgeKind, NodeKind, KineticFamily,
       ACTIVATION, INHIBITION, UNKNOWN_NN,
       STATE, INPUT, LATENT,
       MASS_ACTION, SATURATION, HILL, COMPETITIVE, CUSTOM_KINETIC
export KineticMetadata, InputDriveMetadata, MassActionMetadata,
       HillMetadata, CompetitiveMetadata, LinearDecayMetadata,
       SaturationMetadata, CustomKineticMetadata
export UDEModel, build_ude_model, compile_mechanism, ude_rhs, ude_rhs!,
       parameter_schema, ParameterSchema
export TrainingResult, TrainingRetcode, TrainingConfig, HorizonCurriculum,
       SolverConfig, StructuralPositivity, AugmentedLagrangianConfig,
       AbstractConstraintStrategy, AbstractADPolicy, ZygoteAD, ProductionAD,
       train_ude, train_experiments, predict_ude
export Experiment, ExperimentSet, experiment_from_csv, write_experiment_csv,
       generate_experiment_set
export discover_equations, regress_unknown_rate, DiscoveryResult, DiscoveryRetcode,
       DiscoverySuccess, InsufficientSamples, DenominatorUnsafe, EmptySupport,
       SingularLibrary, DiscoveryFailed,
       DiscoveryConfig, ImplicitSINDyPI, ExplicitSTLSQ,
       StabilitySelection, stability_selection_report, format_stability_selection,
       discover_unknown_term, UnknownTermResult, report_unknown_term,
       discover_unknown_terms, DiscoveryRun, report_unknown_terms,
       UnknownTerm, unknown_terms, cross_term_collinearity,
       network_from_reactionsystem, symbolic,
       ImplicitCandidate, ExplicitCandidate,
       local_basis, export_rhs, equation_to_latex, equation_to_function,
       estimate_derivatives, compose_hybrid_rhs, sample_unknown_destruction,
       hybrid_data_residual, NeuralDestructionTerm
export validate_network, state_nodes, candidate_parents
export build_ude_function, auto_sensealg, default_solver_config

end # module HybridKinetics
