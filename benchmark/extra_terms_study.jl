#!/usr/bin/env julia
# Where the extra terms of the reference protocol come from (0.18, hypothesis A).
# Trains each fixture once per seed and observation-noise level, then measures
# four explanations on that training's samples, so that what varies is the
# discovery step and not the training:
#   ablation    one accepted term removed at a time, the rest refitted
#   selection   a sparsity sweep with AIC, BIC and knee selection
#   split       separate numerator and denominator sparsity thresholds
#   derivative  derivative rows added to the regression
# Default study: fixtures two_state and four_state; seeds 103, 107, 111, 113,
# 127; noise 0.0, 0.02, 0.05 (30 trainings). Rows are appended to
# benchmark/results/extra_terms_study.csv as they finish and a rerun skips the
# cells already in the file, so an interrupted study resumes where it stopped.
# Runtime: about four minutes of training per cell on 4 cores, and a few
# seconds for all four hypotheses, so about two hours for the default study.
# Run:  julia --project=. benchmark/extra_terms_study.jl [options]
#   --fixtures two_state,four_state   comma-separated fixtures
#   --seeds 103,107                   comma-separated seeds
#   --noise 0.0,0.02                  comma-separated noise levels
#   --hypotheses ablation,selection    comma-separated hypotheses
#   --smoke                           the smoke training budget (a quick check)
#   --out PATH                        CSV path
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using HybridKinetics
const HK = HybridKinetics

function parse_list(args, flag, default, convert)
    index = findfirst(==(flag), args)
    index === nothing && return default
    index < length(args) || error("$(flag) needs a value")
    return Tuple(convert(strip(item)) for item in split(args[index + 1], ","))
end

function main(args)
    fixtures = parse_list(args, "--fixtures", HK.EXTRA_TERMS_FIXTURES, Symbol)
    seeds = parse_list(args, "--seeds", HK.EXTRA_TERMS_SEEDS, x -> parse(Int, x))
    noise = parse_list(args, "--noise", HK.EXTRA_TERMS_NOISE_LEVELS,
        x -> parse(Float64, x))
    hypotheses = parse_list(args, "--hypotheses", HK.EXTRA_TERMS_HYPOTHESES, Symbol)
    kind = "--smoke" in args ? :smoke : :protocol
    out_index = findfirst(==("--out"), args)
    out = out_index === nothing ?
          joinpath(@__DIR__, "results", "extra_terms_study.csv") : args[out_index + 1]
    println("fixtures $(fixtures) seeds $(seeds) noise $(noise) hypotheses $(hypotheses) kind $(kind)")
    println("out $(out)")
    flush(stdout)
    HK.extra_terms_study(; fixtures, seeds, noise_levels = noise, hypotheses,
        kind, out)
    rows = HK.read_extra_terms_csv(out)
    println("\n$(length(rows)) rows in $(out)")
    println(HK.format_extra_terms_summary(rows))
    return nothing
end

main(ARGS)
