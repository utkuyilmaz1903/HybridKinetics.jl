#!/usr/bin/env julia
# How much one run varies (0.18, hypothesis C). Three measurements on the
# two-state reference protocol, with the data seed and the initialisation seed
# separated so that the spread over initialisations is measurable at all:
#   initialisation  ten random initialisations of the neural term per seed
#   warmup          warm-up of zero, of the package's length and of twice it,
#                   with all observations and with every second observation of
#                   the regulator hidden
#   divergence      whether the hybrid model with the discovered rate can be
#                   resimulated, and what the candidate alone says about it
#   audit           the same question on the library comparison study's cells,
#                   where the stored runs that do not resimulate are
# Rows are appended as they finish and a rerun skips what is already in the
# file. Runtime: about 3 minutes per run on 4 cores.
# Run:  julia --project=. benchmark/reliability_study.jl [options]
#   --experiment initialisation|warmup|divergence|audit
#   --seeds 103,107        comma-separated seeds
#   --noise 0.0,0.05       comma-separated noise levels (divergence and audit)
#   --design constant      sample design of the audit's four-state fixture
#   --out PATH             CSV path
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using HybridKinetics
const HK = HybridKinetics

function option(args, flag, default)
    index = findfirst(==(flag), args)
    index === nothing ? default : args[index + 1]
end

function main(args)
    experiment = Symbol(option(args, "--experiment", "initialisation"))
    seeds = Tuple(parse(Int, s)
    for s in split(option(args, "--seeds", join(HK.RELIABILITY_SEEDS, ",")), ","))
    noise = Tuple(parse(Float64, s)
    for s in split(option(args, "--noise", "0.0,0.02,0.05"), ","))
    results = joinpath(@__DIR__, "results")
    if experiment === :audit
        design = Symbol(option(args, "--design", "constant"))
        out = option(args, "--out",
            joinpath(results, "reliability_divergence_audit.csv"))
        rows = HK.reliability_divergence_audit(; seeds, noise_levels = noise, design,
            variants = HK.LIBRARY_STUDY_VARIANTS,
            on_row = row -> (HK.append_reliability_audit_row(out, row);
            println("  ", row.variant, "/", row.library, " diverged ", row.diverged);
            flush(stdout)))
        for check in HK.reliability_audit_summary(rows)
            println(rpad(check.name, 46), " ", check.caught, " of ", check.of,
                ", false alarms ", check.false_alarms, " of ", check.of_fine)
        end
        return nothing
    end
    out = option(args, "--out", joinpath(results, "reliability_study.csv"))
    record = row -> (HK.append_reliability_row(out, row);
    println("  ", row.setting, " loss ", row.final_loss, " rate ", row.nn_rate_rmse);
    flush(stdout))
    if experiment === :initialisation
        HK.reliability_initialisation(; seeds, on_row = record)
    elseif experiment === :warmup
        HK.reliability_warmup(; seeds, on_row = record)
    elseif experiment === :divergence
        HK.reliability_divergence(; seeds, noise_levels = noise, on_row = record)
    else
        error("--experiment must be initialisation, warmup, divergence or audit")
    end
    println("\n", HK.format_reliability_summary(HK.read_reliability_csv(out)))
    return nothing
end

main(ARGS)
