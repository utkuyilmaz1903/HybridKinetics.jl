#!/usr/bin/env julia
# Where the adjacent-node bias comes from (0.18, hypothesis B). Two unknown
# terms on adjacent nodes leave one term's learned rate biased low; this runs
# the four settings the 0.18 milestone measured, on the coupled fixture:
#   joint       the package's own path, the baseline the others are read against
#   true_init   both terms pre-trained to their true rates, then trained
#               normally: whether the bias is the data or the search
#   sequential  one term trained with the other mechanism known, frozen, then
#               the other; run in both orderings, because the coupled
#               fixture's graph does not say which term is downstream
#   weighted    each node's residual weighted by that state's observed variance
# Rows are appended as they finish and a rerun skips the cells already in the
# file, so an interrupted study resumes where it stopped.
# Runtime: about 5 minutes per joint or weighted run and about 13 minutes per
# sequential run (two trainings) on 4 cores.
# Run:  julia --project=. benchmark/adjacent_bias_study.jl [options]
#   --settings joint,true_init      comma-separated settings
#   --seeds 103,107                 comma-separated seeds
#   --noise 0.0                     observation-noise standard deviation
#   --scan                          the scale scan instead of the settings
#   --out PATH                      CSV path
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using HybridKinetics
const HK = HybridKinetics

function option(args, flag, default)
    index = findfirst(==(flag), args)
    index === nothing ? default : args[index + 1]
end

function main(args)
    seeds = Tuple(parse(Int, s)
    for s in split(option(args, "--seeds", join(HK.LIBRARY_STUDY_SEEDS, ",")), ","))
    noise = parse(Float64, option(args, "--noise", "0.0"))
    results = joinpath(@__DIR__, "results")
    if "--scan" in args
        out = option(args, "--out", joinpath(results, "adjacent_scale_scan.csv"))
        open(out, "w") do io
            println(io, "seed,noise,fixture,factor_1,factor_2,node_1,node_2,residual")
            for seed in seeds
                println("scan seed $(seed)")
                flush(stdout)
                for row in HK.adjacent_scale_scan(; seed, noise_σ = noise)
                    println(io,
                        join(
                            (row.seed, row.noise, row.fixture, row.factor_1,
                                row.factor_2, row.node_1, row.node_2, row.residual),
                            ","))
                end
                flush(io)
            end
        end
        println("\nwrote $(out)")
        return nothing
    end
    settings = Tuple(Symbol(s)
    for s in split(option(args, "--settings", join(HK.ADJACENT_SETTINGS, ",")), ","))
    out = option(args, "--out", joinpath(results, "adjacent_bias_study.csv"))
    done = Set{Tuple{String, Int, String}}()
    if isfile(out)
        for row in HK.read_adjacent_csv(out)
            push!(done, (row.setting, row.seed, row.note))
        end
    end
    for setting in settings, seed in seeds
        orders = setting === :sequential ? (:A, :B) : (nothing,)
        for first_node in orders
            seen = any(done) do key
                key[1] == String(setting) && key[2] == seed &&
                    (first_node === nothing || occursin("upstream $(first_node) ", key[3]))
            end
            seen && (println("skip $(setting) seed $(seed) $(first_node)");
            flush(stdout);
            continue)
            println("run $(setting) seed $(seed) first=$(first_node)")
            flush(stdout)
            for row in HK.adjacent_run(;
                setting, seed, noise_σ = noise, sequential_first = first_node)
                HK.append_adjacent_row(out, row)
                println("  ", row.node, " bias ", round(row.nn_rate_bias; digits = 4))
                flush(stdout)
            end
        end
    end
    println("\n", HK.format_adjacent_summary(HK.read_adjacent_csv(out)))
    return nothing
end

main(ARGS)
