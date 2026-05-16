using Printf, Serialization, Statistics, TOML
using Plots

include("simulate_data.jl")

gr()

function usage()
    """
    Usage:
      julia --project=. --startup-file=no plot_run_summary.jl PARAMS.toml RUN_OR_CHAIN.jls... [--out-prefix OUT] [--burn-in N]

    Inputs:
      - one TOML parameter file
      - either a *_run.jls metadata file, or individual simulation/allocation/regression .jls files

    Outputs:
      - OUT.pdf with allocation and regression log-posterior traces
      - OUT.txt with trace validation and gamma feature-identification summaries

    Burn-in:
      --burn-in N applies to both allocation and regression traces.
      --allocation-burn-in N and --regression-burn-in N override it separately.
      Equivalent TOML defaults can be set under [plotting].
    """
end

function parse_nonnegative_int(value, name::AbstractString)
    n = tryparse(Int, String(value))
    n === nothing && error("$name must be a non-negative integer, got $(repr(value)).")
    n >= 0 || error("$name must be non-negative, got $n.")
    n
end

function parse_args(args)
    toml_files = String[]
    jls_files = String[]
    out_prefix = nothing
    burn_in = nothing
    allocation_burn_in = nothing
    regression_burn_in = nothing
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--out-prefix"
            i < length(args) || error("--out-prefix requires a value.")
            out_prefix = args[i + 1]
            i += 2
        elseif arg in ("--burn-in", "--burn")
            i < length(args) || error("$arg requires a value.")
            burn_in = parse_nonnegative_int(args[i + 1], arg)
            i += 2
        elseif arg in ("--allocation-burn-in", "--alloc-burn-in")
            i < length(args) || error("$arg requires a value.")
            allocation_burn_in = parse_nonnegative_int(args[i + 1], arg)
            i += 2
        elseif arg in ("--regression-burn-in", "--reg-burn-in")
            i < length(args) || error("$arg requires a value.")
            regression_burn_in = parse_nonnegative_int(args[i + 1], arg)
            i += 2
        elseif endswith(lowercase(arg), ".toml")
            push!(toml_files, arg)
            i += 1
        elseif endswith(lowercase(arg), ".jls")
            push!(jls_files, arg)
            i += 1
        elseif arg in ("-h", "--help")
            println(usage())
            exit(0)
        else
            error("Unrecognized argument: $arg\n$(usage())")
        end
    end
    !isempty(toml_files) || error("Pass one TOML parameter file.\n$(usage())")
    !isempty(jls_files) || error("Pass at least one .jls file.\n$(usage())")
    length(toml_files) == 1 || error("Pass exactly one TOML parameter file.")
    (; toml_file = toml_files[1], jls_files, out_prefix,
       burn_in, allocation_burn_in, regression_burn_in)
end

has_field(x, key::Symbol) = hasproperty(x, key)

function load_jls(path::AbstractString)
    isfile(path) || error("File does not exist: $path")
    deserialize(path)
end

function classify_payload(payload)
    if has_field(payload, :paths) && has_field(payload, :metadata)
        return :run
    elseif has_field(payload, :regression_truth) && has_field(payload, :supervised_data)
        return :simulation
    elseif has_field(payload, :fit) && has_field(payload, :keep_iterations)
        return :allocation
    elseif has_field(payload, :fit) && has_field(payload, :model)
        return :regression
    else
        return :unknown
    end
end

function maybe_load_path(path)
    path === nothing && return nothing
    p = String(path)
    isfile(p) || error("Run metadata points to missing file: $p")
    load_jls(p)
end

function load_inputs(jls_files::Vector{String})
    run_payload = nothing
    simulation = nothing
    allocation = nothing
    regression = nothing

    for path in jls_files
        payload = load_jls(path)
        kind = classify_payload(payload)
        if kind == :run
            run_payload = payload
        elseif kind == :simulation
            simulation = payload
        elseif kind == :allocation
            allocation = payload
        elseif kind == :regression
            regression = payload
        else
            error("Could not classify .jls payload: $path")
        end
    end

    if run_payload !== nothing
        paths = run_payload.paths
        simulation === nothing && has_field(paths, :simulation) &&
            (simulation = maybe_load_path(paths.simulation.jls_path))
        allocation === nothing && has_field(paths, :allocation_chain) &&
            (allocation = maybe_load_path(paths.allocation_chain.jls_path))
        regression === nothing && has_field(paths, :regression_chain) &&
            (regression = maybe_load_path(paths.regression_chain.jls_path))
    end

    allocation !== nothing || error("No allocation chain payload found.")
    regression !== nothing || error("No regression chain payload found.")
    (; run_payload, simulation, allocation, regression)
end

function logpost_trace(payload, label::AbstractString)
    trace = if has_field(payload, :logpost_trace)
        payload.logpost_trace
    elseif has_field(payload, :fit) && has_field(payload.fit, :logpost)
        payload.fit.logpost
    else
        error("$label .jls payload does not contain log posterior trace data. Expected payload.logpost_trace or payload.fit.logpost.")
    end
    !isempty(trace) || error("$label log posterior trace is empty.")
    collect(Float64, trace)
end

function toml_int(table, key::AbstractString, default)
    haskey(table, key) || return default
    value = table[key]
    value isa Integer || error("plotting.$key must be an integer, got $(repr(value)).")
    value >= 0 || error("plotting.$key must be non-negative, got $value.")
    Int(value)
end

function burn_settings(args, params)
    plotting = get(params, "plotting", Dict{String,Any}())
    default_burn = args.burn_in === nothing ? toml_int(plotting, "burn_in", 0) : args.burn_in
    alloc_burn = args.allocation_burn_in === nothing ? toml_int(plotting, "allocation_burn_in", default_burn) : args.allocation_burn_in
    reg_burn = args.regression_burn_in === nothing ? toml_int(plotting, "regression_burn_in", default_burn) : args.regression_burn_in
    (; allocation = alloc_burn, regression = reg_burn)
end

function discard_burn(trace::AbstractVector, burn::Int, label::AbstractString)
    burn < length(trace) || error("$label burn-in $burn discards all $(length(trace)) samples.")
    collect(trace[(burn + 1):end])
end

function discard_burn_rows(chain::AbstractMatrix, burn::Int, label::AbstractString)
    burn < size(chain, 1) || error("$label burn-in $burn discards all $(size(chain, 1)) samples.")
    chain[(burn + 1):end, :]
end

function gamma_truth(inputs)
    if inputs.simulation !== nothing && has_field(inputs.simulation, :regression_truth) &&
       has_field(inputs.simulation.regression_truth, :gamma)
        return inputs.simulation.regression_truth.gamma
    end
    if inputs.run_payload !== nothing && has_field(inputs.run_payload, :plotting_metadata) &&
       has_field(inputs.run_payload.plotting_metadata, :regression_truth) &&
       has_field(inputs.run_payload.plotting_metadata.regression_truth, :gamma)
        return inputs.run_payload.plotting_metadata.regression_truth.gamma
    end
    nothing
end

function gamma_chain(regression)
    if has_field(regression, :fit) && has_field(regression.fit, :gamma_chain)
        return regression.fit.gamma_chain
    end
    nothing
end

function gamma_summary(gchain::AbstractMatrix{Bool}, gtrue::AbstractVector{Bool})
    size(gchain, 2) == length(gtrue) ||
        error("gamma_chain has $(size(gchain, 2)) features but gamma_true has $(length(gtrue)).")
    true1 = BitVector(gtrue)
    true0 = .!true1
    sampled1 = gchain
    sampled0 = .!gchain
    pip = vec(mean(Float64.(gchain), dims = 1))
    call = pip .>= 0.5

    (; nsamples = size(gchain, 1),
       nfeatures = size(gchain, 2),
       ntrue_active = count(true1),
       ntrue_inactive = count(true0),
       p_sampled1_true1 = any(true1) ? mean(sampled1[:, true1]) : NaN,
       p_sampled0_true1 = any(true1) ? mean(sampled0[:, true1]) : NaN,
       p_sampled1_true0 = any(true0) ? mean(sampled1[:, true0]) : NaN,
       p_sampled0_true0 = any(true0) ? mean(sampled0[:, true0]) : NaN,
       threshold = 0.5,
       tp = count(call .& true1),
       tn = count((.!call) .& true0),
       fp = count(call .& true0),
       fn = count((.!call) .& true1),
       posterior_inclusion_prob = pip,
       gamma_true = true1)
end

function default_out_prefix(args, inputs, params)
    args.out_prefix !== nothing && return args.out_prefix
    if inputs.run_payload !== nothing && has_field(inputs.run_payload, :metadata)
        return joinpath(inputs.run_payload.metadata.output_dir,
                        "$(inputs.run_payload.metadata.prefix)_plot_summary")
    end
    run = get(params, "run", Dict{String,Any}())
    prefix = get(run, "prefix", splitext(basename(args.toml_file))[1])
    outdir = get(run, "output_dir", dirname(args.jls_files[1]))
    joinpath(String(outdir), "$(String(prefix))_plot_summary")
end

function trace_axes!(sp, trace; color, title, ylabel, burn_in)
    plot!(sp, eachindex(trace), trace; color, lw = 1.3, xlabel = "iteration",
          ylabel, title = "$title\nburn-in discarded=$burn_in", legend = false)
    hline!(sp, [maximum(trace)]; color = :gray45, ls = :dash, lw = 1.0)
end

function gamma_heatmap_axes!(sp, gs)
    G = [gs.p_sampled0_true0 gs.p_sampled1_true0;
         gs.p_sampled0_true1 gs.p_sampled1_true1]
    heatmap!(sp, ["sampled=0", "sampled=1"], ["true=0", "true=1"], G;
             c = :viridis, clims = (0, 1), aspect_ratio = :equal,
             title = "Gamma recovery probabilities", legend = false)
    for i in 1:2, j in 1:2
        annotate!(sp, j, i, text(@sprintf("%.3f", G[i, j]), 10, :white))
    end
end

function gamma_pip_axes!(sp, gs)
    ord = sortperm(gs.posterior_inclusion_prob)
    active = gs.gamma_true[ord]
    x = collect(1:gs.nfeatures)
    scatter!(sp, x[.!active], gs.posterior_inclusion_prob[ord][.!active];
             color = :gray55, ms = 3, alpha = 0.75, label = "true inactive",
             xlabel = "feature sorted by PIP", ylabel = "P(gamma=1)",
             title = "Posterior inclusion probabilities")
    scatter!(sp, x[active], gs.posterior_inclusion_prob[ord][active];
             color = :firebrick3, ms = 4, alpha = 0.9, label = "true active")
    hline!(sp, [0.5]; color = :black, ls = :dash, lw = 1.0)
end

function make_pdf(path, alloc_trace, reg_trace, gs, burns)
    if gs === nothing
        plt = plot(layout = (2, 1), size = (950, 700))
        trace_axes!(plt[1], alloc_trace; color = :purple4,
                    title = "Allocation log posterior", ylabel = "allocation logpost",
                    burn_in = burns.allocation)
        trace_axes!(plt[2], reg_trace; color = :firebrick3,
                    title = "Regression log posterior", ylabel = "regression logpost",
                    burn_in = burns.regression)
    else
        plt = plot(layout = (2, 2), size = (1150, 850))
        trace_axes!(plt[1], alloc_trace; color = :purple4,
                    title = "Allocation log posterior", ylabel = "allocation logpost",
                    burn_in = burns.allocation)
        trace_axes!(plt[2], reg_trace; color = :firebrick3,
                    title = "Regression log posterior", ylabel = "regression logpost",
                    burn_in = burns.regression)
        gamma_heatmap_axes!(plt[3], gs)
        gamma_pip_axes!(plt[4], gs)
    end
    savefig(plt, path)
    path
end

function write_text_summary(path, args, inputs, alloc_trace_raw, reg_trace_raw, alloc_trace, reg_trace, gs, burns)
    open(path, "w") do io
        println(io, "Run Plot Summary")
        println(io, "================")
        println(io, "parameter_file: ", abspath(args.toml_file))
        println(io, "jls_files:")
        for f in args.jls_files
            println(io, "  - ", abspath(f))
        end
        if inputs.run_payload !== nothing
            println(io)
            println(io, "run_metadata:")
            println(io, "  prefix: ", inputs.run_payload.metadata.prefix)
            println(io, "  seed: ", inputs.run_payload.metadata.seed)
            println(io, "  output_dir: ", inputs.run_payload.metadata.output_dir)
            println(io, "  elapsed_seconds: ", get(inputs.run_payload.metadata, :elapsed_seconds, "NA"))
        end

        println(io)
        println(io, "Burn-in:")
        println(io, "  allocation_burn_in: ", burns.allocation)
        println(io, "  regression_burn_in: ", burns.regression)

        println(io)
        println(io, "Log posterior trace validation:")
        println(io, "  allocation: present, raw_length=$(length(alloc_trace_raw)), kept_length=$(length(alloc_trace)), final=$(@sprintf("%.3f", alloc_trace[end])), max=$(@sprintf("%.3f", maximum(alloc_trace)))")
        println(io, "  regression: present, raw_length=$(length(reg_trace_raw)), kept_length=$(length(reg_trace)), final=$(@sprintf("%.3f", reg_trace[end])), max=$(@sprintf("%.3f", maximum(reg_trace)))")

        println(io)
        println(io, "Feature identification:")
        if gs === nothing
            println(io, "  not available: gamma_chain and/or gamma_true was not found.")
        else
            println(io, "  gamma_samples_post_burn: ", gs.nsamples)
            println(io, "  nfeatures: ", gs.nfeatures)
            println(io, "  true_active: ", gs.ntrue_active)
            println(io, "  true_inactive: ", gs.ntrue_inactive)
            println(io, "  P(gamma_sampled = 1 | gamma_true = 1): ", @sprintf("%.6f", gs.p_sampled1_true1))
            println(io, "  P(gamma_sampled = 0 | gamma_true = 1): ", @sprintf("%.6f", gs.p_sampled0_true1))
            println(io, "  P(gamma_sampled = 1 | gamma_true = 0): ", @sprintf("%.6f", gs.p_sampled1_true0))
            println(io, "  P(gamma_sampled = 0 | gamma_true = 0): ", @sprintf("%.6f", gs.p_sampled0_true0))
            println(io, "  threshold_for_counts: ", gs.threshold)
            println(io, "  true_positives_count: ", gs.tp)
            println(io, "  true_negatives_count: ", gs.tn)
            println(io, "  false_positives_count: ", gs.fp)
            println(io, "  false_negatives_count: ", gs.fn)
        end
    end
    path
end

function main()
    args = parse_args(ARGS)
    params = TOML.parsefile(args.toml_file)
    inputs = load_inputs(args.jls_files)
    burns = burn_settings(args, params)

    alloc_trace_raw = logpost_trace(inputs.allocation, "allocation")
    reg_trace_raw = logpost_trace(inputs.regression, "regression")
    alloc_trace = discard_burn(alloc_trace_raw, burns.allocation, "allocation")
    reg_trace = discard_burn(reg_trace_raw, burns.regression, "regression")

    gchain = gamma_chain(inputs.regression)
    gtrue = gamma_truth(inputs)
    gs = if gchain === nothing || gtrue === nothing
        nothing
    else
        gamma_summary(discard_burn_rows(gchain, burns.regression, "gamma_chain"), gtrue)
    end

    out_prefix = default_out_prefix(args, inputs, params)
    mkpath(dirname(out_prefix))
    pdf_path = abspath("$(out_prefix).pdf")
    txt_path = abspath("$(out_prefix).txt")

    make_pdf(pdf_path, alloc_trace, reg_trace, gs, burns)
    write_text_summary(txt_path, args, inputs, alloc_trace_raw, reg_trace_raw, alloc_trace, reg_trace, gs, burns)

    println("validated allocation logpost trace length: ", length(alloc_trace_raw), " raw, ", length(alloc_trace), " kept")
    println("validated regression logpost trace length: ", length(reg_trace_raw), " raw, ", length(reg_trace), " kept")
    println("saved plot pdf: ", pdf_path)
    println("saved text summary: ", txt_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
