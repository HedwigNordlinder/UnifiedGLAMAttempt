using Dates, LinearAlgebra, Random, Serialization, Statistics, TOML
using Distributions

include("simulate_data.jl")

const EMPTY_TABLE = Dict{String,Any}()

section(params, name::AbstractString) = begin
    value = get(params, name, EMPTY_TABLE)
    value isa AbstractDict || error("Parameter section [$name] must be a table.")
    value
end

cfgget(table::AbstractDict, key::AbstractString, default) = haskey(table, key) ? table[key] : default
as_int(x, name) = x isa Integer ? Int(x) : error("$name must be an integer, got $(repr(x)).")
as_float(x, name, ::Type{T} = Float64) where {T<:AbstractFloat} = T(x)
as_bool(x, name) = x isa Bool ? x : error("$name must be boolean, got $(repr(x)).")
as_string(x, name) = x isa AbstractString ? String(x) : error("$name must be a string, got $(repr(x)).")
lower_string(x, name) = lowercase(as_string(x, name))

function progress_setting(params::AbstractDict, table::AbstractDict, key::AbstractString, name::AbstractString)
    run = section(params, "run")
    as_bool(cfgget(table, key, cfgget(run, "progress", true)), "$name.$key")
end

function phase_log(enabled::Bool, message::AbstractString)
    enabled || return nothing
    stamp = Dates.format(now(), dateformat"HH:MM:SS")
    println("[$stamp] $message")
    flush(stdout)
    nothing
end

function parse_vector(value, p::Int, name::AbstractString, ::Type{T} = Float64) where {T<:AbstractFloat}
    if value isa Real
        return fill(T(value), p)
    elseif value isa AbstractVector
        length(value) == p || error("$name must be scalar or length $p, got length $(length(value)).")
        return collect(T, value)
    else
        error("$name must be scalar or vector, got $(typeof(value)).")
    end
end

function parse_two_scales(value, name::AbstractString, ::Type{T} = Float64) where {T<:AbstractFloat}
    if value isa Real
        return (T(value), T(value))
    elseif value isa AbstractVector && length(value) == 2
        return (T(value[1]), T(value[2]))
    else
        error("$name must be scalar or length-2 vector.")
    end
end

function parse_nobs_range(sim::AbstractDict)
    if haskey(sim, "nobs_range")
        r = sim["nobs_range"]
        (r isa AbstractVector && length(r) == 2) || error("simulation.nobs_range must be [min, max].")
        lo, hi = as_int(r[1], "simulation.nobs_range[1]"), as_int(r[2], "simulation.nobs_range[2]")
    else
        lo = as_int(cfgget(sim, "nobs_min", 8), "simulation.nobs_min")
        hi = as_int(cfgget(sim, "nobs_max", 14), "simulation.nobs_max")
    end
    lo <= hi || error("nobs minimum must be <= maximum.")
    lo:hi
end

function parse_distribution(table::AbstractDict, p::Int, name::AbstractString, ::Type{T} = Float64) where {T<:AbstractFloat}
    dist = lower_string(cfgget(table, "dist", "normal"), "$name.dist")
    if dist in ("normal", "univariate_normal")
        mu = as_float(cfgget(table, "mean", 0.0), "$name.mean", T)
        sd = as_float(cfgget(table, "sd", 1.0), "$name.sd", T)
        sd > zero(T) || error("$name.sd must be positive.")
        return Normal(mu, sd)
    elseif dist in ("mvnormal_diag", "mvn_diag")
        mu = parse_vector(cfgget(table, "mean", 0.0), p, "$name.mean", T)
        sd = parse_vector(cfgget(table, "sd", 1.0), p, "$name.sd", T)
        all(>(zero(T)), sd) || error("All entries of $name.sd must be positive.")
        return MvNormal(mu, Diagonal(sd .^ 2))
    elseif dist in ("mvnormal_iso", "mvn_iso")
        mu = parse_vector(cfgget(table, "mean", 0.0), p, "$name.mean", T)
        sd = as_float(cfgget(table, "sd", 1.0), "$name.sd", T)
        sd > zero(T) || error("$name.sd must be positive.")
        return MvNormal(mu, (sd^2) * Matrix{T}(I, p, p))
    else
        error("Unsupported $name.dist = $(repr(dist)). Supported: normal, mvnormal_diag, mvnormal_iso.")
    end
end

function parse_t_prior(table::AbstractDict, ::Type{T} = Float64) where {T<:AbstractFloat}
    dist = lower_string(cfgget(table, "dist", "uniform"), "t_prior.dist")
    dist == "uniform" || error("Only uniform t_prior is currently supported.")
    lo = as_float(cfgget(table, "low", 0.0), "t_prior.low", T)
    hi = as_float(cfgget(table, "high", 1.0), "t_prior.high", T)
    lo < hi || error("Require t_prior.low < t_prior.high.")
    Uniform(lo, hi)
end

function build_latent_prior(params::AbstractDict, p::Int, ::Type{T} = Float64) where {T<:AbstractFloat}
    latent = section(params, "latent_prior")
    m0 = parse_vector(cfgget(latent, "m0", 0.0), p, "latent_prior.m0", T)
    lambda0_scale = as_float(cfgget(latent, "lambda0_scale", 2.0), "latent_prior.lambda0_scale", T)
    lambda0_scale > zero(T) || error("latent_prior.lambda0_scale must be positive.")
    s0_scales = parse_two_scales(cfgget(latent, "S0_scale", 1.0), "latent_prior.S0_scale", T)
    all(>(zero(T)), s0_scales) || error("latent_prior.S0_scale entries must be positive.")
    nu0 = haskey(latent, "nu0") ?
        as_float(latent["nu0"], "latent_prior.nu0", T) :
        T(p + as_float(cfgget(latent, "nu0_offset", 3.0), "latent_prior.nu0_offset", T))

    LatentPrior(
        as_float(cfgget(latent, "alpha0", 2.0), "latent_prior.alpha0", T),
        as_float(cfgget(latent, "beta0", 2.0), "latent_prior.beta0", T),
        m0,
        lambda0_scale * Matrix{T}(I, p, p),
        as_float(cfgget(latent, "a0", 2.0), "latent_prior.a0", T),
        as_float(cfgget(latent, "b0", 2.0), "latent_prior.b0", T),
        nu0,
        (s0_scales[1] * Matrix{T}(I, p, p), s0_scales[2] * Matrix{T}(I, p, p)),
    )
end

function build_simulation_config(params::AbstractDict, ::Type{T} = Float64) where {T<:AbstractFloat}
    sim = section(params, "simulation")
    p = as_int(cfgget(sim, "p", 60), "simulation.p")
    npatients = as_int(cfgget(sim, "npatients", 1000), "simulation.npatients")
    p > 0 || error("simulation.p must be positive.")
    npatients > 0 || error("simulation.npatients must be positive.")
    SimulationConfig(
        npatients,
        parse_nobs_range(sim),
        build_latent_prior(params, p, T),
        parse_distribution(section(params, "beta_prior"), p, "beta_prior", T),
        parse_t_prior(section(params, "t_prior"), T),
    )
end

function supervised_from_state(data::LatentData{T}, state::GibbsState{T}, y::AbstractVector{<:Real}) where {T}
    n = length(data.X)
    p = size(data.X[1], 1)
    A = Matrix{T}(undef, n, p)
    B = Matrix{T}(undef, n, p)
    for i in 1:n
        A[i, :] .= state.model.mu[i][:, 1]
        B[i, :] .= state.model.mu[i][:, 2]
    end
    SupervisedData(A, B, y)
end

function supervised_from_posterior_mean(states::Vector{GibbsState{T}}, y::AbstractVector{<:Real}) where {T}
    isempty(states) && error("Need at least one saved allocation state for posterior_mean regression data.")
    n = length(states[1].model.mu)
    p = size(states[1].model.mu[1], 1)
    A = zeros(T, n, p)
    B = zeros(T, n, p)
    for state in states, i in 1:n
        A[i, :] .+= state.model.mu[i][:, 1]
        B[i, :] .+= state.model.mu[i][:, 2]
    end
    A ./= T(length(states))
    B ./= T(length(states))
    SupervisedData(A, B, y)
end

keep_iterations(nsweeps::Int, burn::Int, thin::Int) =
    [it for it in 1:nsweeps if it > burn && (it - burn) % thin == 0]

function artifact_prefix(parameter_file::AbstractString, prefix::AbstractString)
    parameter_stem = splitext(basename(parameter_file))[1]
    if prefix == parameter_stem || startswith(prefix, "$(parameter_stem)_")
        return prefix
    end
    "$(parameter_stem)_$(prefix)"
end

function save_artifact(stem::AbstractString, payload)
    mkpath(dirname(stem))
    jls_path = abspath("$(stem).jls")
    serialize(jls_path, payload)
    jld2_path = nothing
    if HAVE_JLD2
        jld2_path = abspath("$(stem).jld2")
        JLD2.jldopen(jld2_path, "w") do f
            f["payload"] = payload
        end
    end
    (; jls_path, jld2_path)
end

function gamma_recovery(gamma_chain::AbstractMatrix{Bool}, gamma_true::AbstractVector{Bool})
    true0 = .!gamma_true
    true1 = gamma_true
    [any(true0) ? mean(.!gamma_chain[:, true0]) : NaN  any(true0) ? mean(gamma_chain[:, true0]) : NaN;
     any(true1) ? mean(.!gamma_chain[:, true1]) : NaN  any(true1) ? mean(gamma_chain[:, true1]) : NaN]
end

function allocation_summary(alloc_fit, keep_iters)
    saved_logpost = isempty(keep_iters) ? Float64[] : alloc_fit.logpost[keep_iters]
    imap_saved = isempty(saved_logpost) ? nothing : argmax(saved_logpost)
    imap_iter = imap_saved === nothing ? nothing : keep_iters[imap_saved]
    (; nsweeps = length(alloc_fit.logpost),
       nkept = length(keep_iters),
       final_logpost = alloc_fit.logpost[end],
       max_logpost = maximum(alloc_fit.logpost),
       map_saved_index = imap_saved,
       map_iteration = imap_iter)
end

function regression_summary(reg_fit, truth)
    t_chain = hasproperty(reg_fit, :t_chain) ? getproperty(reg_fit, :t_chain) : nothing
    t_samples = t_chain === nothing ? reg_fit.t_trace : t_chain
    beta_chain = getproperty(reg_fit, :β_chain)
    beta_mean = beta_chain === nothing ? nothing : vec(mean(beta_chain, dims = 1))
    beta_rmse = beta_mean === nothing ? nothing : sqrt(mean((beta_mean .- truth.β).^2))
    beta_cor = beta_mean === nothing || length(beta_mean) < 2 ? nothing : cor(beta_mean, truth.β)
    gamma_chain = hasproperty(reg_fit, :gamma_chain) ? getproperty(reg_fit, :gamma_chain) : nothing
    gamma_stats = gamma_chain !== nothing && hasproperty(truth, :gamma) ?
        gamma_recovery(gamma_chain, truth.gamma) :
        nothing

    (; t_mean = mean(t_samples),
       t_sd = std(t_samples),
       t_true = truth.t,
       beta_rmse,
       beta_cor,
       true_active = hasproperty(truth, :gamma) ? count(truth.gamma) : nothing,
       posterior_active_mean = hasproperty(reg_fit, :active_trace) ? mean(reg_fit.active_trace) : nothing,
       gamma_recovery = gamma_stats,
       mean_acceptance = reg_fit.mean_acceptance,
       divergence_rate = reg_fit.divergence_rate,
       mean_tree_depth = reg_fit.mean_tree_depth,
       initial_step_size = hasproperty(reg_fit, :initial_step_size) ? reg_fit.initial_step_size : nothing,
       mean_step_size = hasproperty(reg_fit, :mean_step_size) ? reg_fit.mean_step_size : nothing)
end

function plain_regression_truth(truth::RegressionTruth)
    (; kind = "dense", beta = truth.β, t = truth.t, eta = truth.η, p = truth.p)
end

function plain_regression_truth(truth::GammaRegressionTruth)
    (; kind = "gamma", beta = truth.β, gamma = truth.gamma, t = truth.t, eta = truth.η, p = truth.p)
end

function select_regression_data(strategy::AbstractString, sim_bundle, alloc_fit, keep_iters)
    y = sim_bundle.supervised_data.y
    if strategy == "truth"
        return sim_bundle.supervised_data, (; strategy, map_saved_index = nothing, map_iteration = nothing, nstates_used = 0)
    end
    alloc_fit.states === nothing && error("allocation_mcmc.save_states must be true for regression_data = $strategy.")
    length(alloc_fit.states) == length(keep_iters) ||
        error("Allocation state count does not match saved iteration count.")

    if strategy == "map"
        !isempty(keep_iters) || error("No saved allocation states are available for MAP regression data.")
        saved_logpost = alloc_fit.logpost[keep_iters]
        imap_saved = argmax(saved_logpost)
        state = alloc_fit.states[imap_saved]
        fit_data = supervised_from_state(sim_bundle.latent_data, state, y)
        return fit_data, (; strategy, map_saved_index = imap_saved, map_iteration = keep_iters[imap_saved], nstates_used = 1)
    elseif strategy == "posterior_mean"
        fit_data = supervised_from_posterior_mean(alloc_fit.states, y)
        return fit_data, (; strategy, map_saved_index = nothing, map_iteration = nothing, nstates_used = length(alloc_fit.states))
    else
        error("Unsupported fitting.regression_data = $(repr(strategy)). Use truth, map, or posterior_mean.")
    end
end

function build_regression_model(params::AbstractDict, config::SimulationConfig, sim_model)
    reg = section(params, "regression_mcmc")
    model_kind = lower_string(cfgget(reg, "model", "auto"), "regression_mcmc.model")
    if model_kind == "auto"
        return sim_model
    elseif model_kind == "dense"
        return RegressionModel(config.β_prior)
    elseif model_kind == "gamma"
        gamma = section(params, "gamma")
        gamma_prior = as_float(cfgget(gamma, "active_probability", 0.1), "gamma.active_probability")
        slab = parse_distribution(section(params, "beta_prior"), length(config.latent_prior.m0), "beta_prior")
        slab isa UnivariateDistribution || error("Gamma regression currently requires a univariate beta_prior slab.")
        return GammaRegressionModel(slab, gamma_prior)
    else
        error("Unsupported regression_mcmc.model = $(repr(model_kind)). Use auto, dense, or gamma.")
    end
end

function parse_init_gamma(rng::AbstractRNG, value, p::Int, gamma_prior::Real)
    if value isa AbstractString
        mode = lowercase(value)
        mode == "all_on" && return trues(p)
        mode == "all_off" && return falses(p)
        mode == "prior" && return BitVector(rand(rng, Bernoulli(gamma_prior), p) .== 1)
        error("Unsupported regression_mcmc.init_gamma = $(repr(value)).")
    elseif value isa AbstractVector
        length(value) == p || error("regression_mcmc.init_gamma must have length $p.")
        return BitVector(Bool.(value))
    else
        error("regression_mcmc.init_gamma must be all_on, all_off, prior, or a boolean vector.")
    end
end

function run_regression(rng::AbstractRNG, params::AbstractDict, model, fit_data::SupervisedData)
    reg = section(params, "regression_mcmc")
    p = size(fit_data.cluster_a_μ, 2)
    init_beta = parse_vector(cfgget(reg, "init_beta", 0.0), p, "regression_mcmc.init_beta")
    init_t = as_float(cfgget(reg, "init_t", 0.5), "regression_mcmc.init_t")
    target_accept = as_float(cfgget(reg, "target_accept", 0.9), "regression_mcmc.target_accept")
    max_depth = as_int(cfgget(reg, "max_depth", 10), "regression_mcmc.max_depth")
    save_states = as_bool(cfgget(reg, "save_states", false), "regression_mcmc.save_states")
    save_chain = as_bool(cfgget(reg, "save_chain", true), "regression_mcmc.save_chain")
    progress = progress_setting(params, reg, "progress", "regression_mcmc")
    verbose = as_bool(cfgget(reg, "verbose", false), "regression_mcmc.verbose")
    progress_every = as_int(cfgget(reg, "progress_every", 0), "regression_mcmc.progress_every")

    if model isa GammaRegressionModel
        init_gamma = parse_init_gamma(rng, cfgget(reg, "init_gamma", "all_on"), p, model.gamma_prior)
        return gamma_hmc(rng, model, fit_data;
                         nsamples = as_int(cfgget(reg, "nsamples", 500), "regression_mcmc.nsamples"),
                         initial_hmc = as_int(cfgget(reg, "initial_hmc", 300), "regression_mcmc.initial_hmc"),
                         initial_adapts = as_int(cfgget(reg, "initial_adapts", 250), "regression_mcmc.initial_adapts"),
                         hmc_steps_per_gamma = as_int(cfgget(reg, "hmc_steps_per_gamma", 5), "regression_mcmc.hmc_steps_per_gamma"),
                         hmc_adapts_per_gamma = as_int(cfgget(reg, "hmc_adapts_per_gamma", 0), "regression_mcmc.hmc_adapts_per_gamma"),
                         reuse_initial_metric = as_bool(cfgget(reg, "reuse_initial_metric", true), "regression_mcmc.reuse_initial_metric"),
                         step_size_adapt_rate = as_float(cfgget(reg, "step_size_adapt_rate", 0.02), "regression_mcmc.step_size_adapt_rate"),
                         step_size_min_factor = as_float(cfgget(reg, "step_size_min_factor", 0.25), "regression_mcmc.step_size_min_factor"),
                         step_size_max_factor = as_float(cfgget(reg, "step_size_max_factor", 2.0), "regression_mcmc.step_size_max_factor"),
                         init_β = init_beta, init_gamma, init_t,
                         target_accept, max_depth, save_states, save_chain, progress, verbose,
                         progress_every = progress_every > 0 ? progress_every : max(1, as_int(cfgget(reg, "nsamples", 500), "regression_mcmc.nsamples") ÷ 100))
    elseif model isa RegressionModel
        return hmc(rng, model, fit_data;
                   nsweeps = as_int(cfgget(reg, "nsweeps", 3000), "regression_mcmc.nsweeps"),
                   n_adapts = as_int(cfgget(reg, "n_adapts", 750), "regression_mcmc.n_adapts"),
                   burn = as_int(cfgget(reg, "burn", 750), "regression_mcmc.burn"),
                   thin = as_int(cfgget(reg, "thin", 3), "regression_mcmc.thin"),
                   init_β = init_beta, init_t,
                   target_accept, max_depth, save_states, save_chain, progress, verbose,
                   progress_every = progress_every > 0 ? progress_every : max(1, as_int(cfgget(reg, "nsweeps", 3000), "regression_mcmc.nsweeps") ÷ 100))
    else
        error("Unsupported regression model type: $(typeof(model)).")
    end
end

function run_engine(parameter_file::AbstractString)
    started_at = now()
    params = TOML.parsefile(parameter_file)
    run = section(params, "run")
    sim = section(params, "simulation")
    fit = section(params, "fitting")
    alloc = section(params, "allocation_mcmc")

    seed = as_int(cfgget(run, "seed", 20260516), "run.seed")
    rng = MersenneTwister(seed)
    prefix = as_string(cfgget(run, "prefix", splitext(basename(parameter_file))[1]), "run.prefix")
    artifact_name_prefix = artifact_prefix(parameter_file, prefix)
    outdir = abspath(as_string(cfgget(run, "output_dir", joinpath("runs", prefix)), "run.output_dir"))
    mkpath(outdir)
    run_progress = as_bool(cfgget(run, "progress", true), "run.progress")

    T = Float64
    config = build_simulation_config(params, T)
    sim_kind = lower_string(cfgget(sim, "regression", "dense"), "simulation.regression")
    cluster_mean_shift = cfgget(sim, "cluster_mean_shift", 1.0)
    simulation_stem = joinpath(outdir, "$(artifact_name_prefix)_simulation")
    phase_log(run_progress, "starting run prefix=$prefix artifact_prefix=$artifact_name_prefix seed=$seed output_dir=$outdir")
    phase_log(run_progress, "simulation: kind=$sim_kind patients=$(config.npatients) p=$(length(config.latent_prior.m0)) nobs=$(first(config.nobs_range)):$(last(config.nobs_range)) cluster_mean_shift=$(cluster_mean_shift)")

    sim_out = if sim_kind in ("dense", "standard")
        simulate_dataset(rng, config; stem = simulation_stem, cluster_mean_shift)
    elseif sim_kind in ("gamma", "spike_slab", "spike-and-slab")
        gamma = section(params, "gamma")
        gamma_prior = as_float(cfgget(gamma, "active_probability", 0.1), "gamma.active_probability", T)
        slab = parse_distribution(section(params, "beta_prior"), length(config.latent_prior.m0), "beta_prior", T)
        slab isa UnivariateDistribution || error("Gamma simulation requires a univariate beta_prior slab.")
        simulate_gamma_dataset(rng, config; stem = simulation_stem,
                               cluster_mean_shift, gamma_prior, β_prior = slab)
    else
        error("Unsupported simulation.regression = $(repr(sim_kind)).")
    end
    active_msg = hasproperty(sim_out.bundle.regression_truth, :gamma) ?
        " true_active=$(count(sim_out.bundle.regression_truth.gamma))/$(length(sim_out.bundle.regression_truth.gamma))" :
        ""
    phase_log(run_progress, "simulation complete: outcome_rate=$(round(sim_out.bundle.stats.outcome_rate; digits=3)) true_t=$(round(sim_out.bundle.regression_truth.t; digits=3))$active_msg")

    alloc_nsweeps = as_int(cfgget(alloc, "nsweeps", 250), "allocation_mcmc.nsweeps")
    alloc_burn = as_int(cfgget(alloc, "burn", 0), "allocation_mcmc.burn")
    alloc_thin = as_int(cfgget(alloc, "thin", 1), "allocation_mcmc.thin")
    alloc_nsweeps >= 1 || error("allocation_mcmc.nsweeps must be positive.")
    0 <= alloc_burn < alloc_nsweeps || error("allocation_mcmc.burn must satisfy 0 <= burn < nsweeps.")
    alloc_thin >= 1 || error("allocation_mcmc.thin must be positive.")
    strategy = lower_string(cfgget(fit, "regression_data", "map"), "fitting.regression_data")
    save_alloc_states = as_bool(cfgget(alloc, "save_states", strategy != "truth"), "allocation_mcmc.save_states")
    alloc_progress = progress_setting(params, alloc, "progress", "allocation_mcmc")
    alloc_progress_every = as_int(cfgget(alloc, "progress_every", max(1, alloc_nsweeps ÷ 100)), "allocation_mcmc.progress_every")
    phase_log(run_progress, "allocation MCMC: nsweeps=$alloc_nsweeps burn=$alloc_burn thin=$alloc_thin save_states=$save_alloc_states")
    alloc_fit = gibbs(rng, sim_out.bundle.latent_data, config.latent_prior;
                      nsweeps = alloc_nsweeps,
                      burn = alloc_burn,
                      thin = alloc_thin,
                      save_states = save_alloc_states,
                      postprocess = as_bool(cfgget(alloc, "postprocess", true), "allocation_mcmc.postprocess"),
                      cluster_mean_shift = cfgget(alloc, "cluster_mean_shift", cluster_mean_shift),
                      progress = alloc_progress,
                      progress_every = alloc_progress_every)
    keep_iters = keep_iterations(alloc_nsweeps, alloc_burn, alloc_thin)
    phase_log(run_progress, "allocation complete: final_logpost=$(round(alloc_fit.logpost[end]; digits=2)) max_logpost=$(round(maximum(alloc_fit.logpost); digits=2)) kept=$(length(keep_iters))")
    fit_data, fit_data_meta = select_regression_data(strategy, sim_out.bundle, alloc_fit, keep_iters)
    phase_log(run_progress, "regression data: strategy=$(fit_data_meta.strategy) states_used=$(fit_data_meta.nstates_used) map_iteration=$(fit_data_meta.map_iteration)")

    reg_model = build_regression_model(params, config, sim_out.bundle.regression_model)
    reg = section(params, "regression_mcmc")
    if reg_model isa GammaRegressionModel
        nsamples = as_int(cfgget(reg, "nsamples", 500), "regression_mcmc.nsamples")
        initial_hmc = as_int(cfgget(reg, "initial_hmc", 300), "regression_mcmc.initial_hmc")
        hmc_steps = as_int(cfgget(reg, "hmc_steps_per_gamma", 5), "regression_mcmc.hmc_steps_per_gamma")
        phase_log(run_progress, "gamma regression MCMC: high_level_samples=$nsamples total_hmc_transitions=$(initial_hmc + nsamples * hmc_steps)")
    else
        nsweeps = as_int(cfgget(reg, "nsweeps", 3000), "regression_mcmc.nsweeps")
        n_adapts = as_int(cfgget(reg, "n_adapts", 750), "regression_mcmc.n_adapts")
        burn = as_int(cfgget(reg, "burn", 750), "regression_mcmc.burn")
        thin = as_int(cfgget(reg, "thin", 3), "regression_mcmc.thin")
        phase_log(run_progress, "dense regression HMC: nsweeps=$nsweeps n_adapts=$n_adapts burn=$burn thin=$thin")
    end
    reg_fit = run_regression(rng, params, reg_model, fit_data)
    phase_log(run_progress, "regression complete: mean_accept_prob=$(round(reg_fit.mean_acceptance; digits=3)) divergence_rate=$(round(reg_fit.divergence_rate; digits=3))")

    alloc_payload = (; fit = alloc_fit,
                     logpost_trace = alloc_fit.logpost,
                     keep_iterations = keep_iters,
                     config = section(params, "allocation_mcmc"),
                     summary = allocation_summary(alloc_fit, keep_iters))
    fit_data_payload = (; supervised_data = fit_data, metadata = fit_data_meta)
    reg_payload = (; fit = reg_fit,
                   logpost_trace = reg_fit.logpost,
                   model = reg_model,
                   fit_data_metadata = fit_data_meta,
                   config = section(params, "regression_mcmc"),
                   summary = regression_summary(reg_fit, sim_out.bundle.regression_truth))

    paths = (;
        simulation = (; jls_path = abspath(sim_out.paths.jls_path), jld2_path = sim_out.paths.jld2_path),
        allocation_chain = save_artifact(joinpath(outdir, "$(artifact_name_prefix)_allocation_chain"), alloc_payload),
        fit_data = save_artifact(joinpath(outdir, "$(artifact_name_prefix)_fit_data"), fit_data_payload),
        regression_chain = save_artifact(joinpath(outdir, "$(artifact_name_prefix)_regression_chain"), reg_payload),
    )

    raw_parameter_file = read(parameter_file, String)
    finished_at = now()
    run_payload = (;
        metadata = (;
            started_at = Dates.format(started_at, dateformat"yyyy-mm-ddTHH:MM:SS"),
            finished_at = Dates.format(finished_at, dateformat"yyyy-mm-ddTHH:MM:SS"),
            elapsed_seconds = Dates.value(finished_at - started_at) / 1000,
            julia_version = string(VERSION),
            parameter_file = abspath(parameter_file),
            output_dir = outdir,
            prefix,
            artifact_prefix = artifact_name_prefix,
            seed,
        ),
        parameter_toml = raw_parameter_file,
        parsed_parameters = params,
        paths,
        summaries = (;
            simulation = sim_out.bundle.stats,
            allocation = alloc_payload.summary,
            fit_data = fit_data_meta,
            regression = reg_payload.summary,
        ),
        plotting_metadata = (;
            summary_plot = sim_out.bundle.summary_plot,
            latent_cluster_mean_shift = sim_out.bundle.latent_cluster_mean_shift,
            latent_cluster_centers = sim_out.bundle.latent_cluster_centers,
            regression_truth = plain_regression_truth(sim_out.bundle.regression_truth),
            regression_data = fit_data_meta,
        ),
    )
    run_paths = save_artifact(joinpath(outdir, "$(artifact_name_prefix)_run"), run_payload)
    phase_log(run_progress, "artifacts saved")

    println("saved simulation: ", paths.simulation.jls_path)
    println("saved allocation chain: ", paths.allocation_chain.jls_path)
    println("saved fitted supervised data: ", paths.fit_data.jls_path)
    println("saved regression chain: ", paths.regression_chain.jls_path)
    println("saved run metadata: ", run_paths.jls_path)
    println("posterior mean t: ", round(reg_payload.summary.t_mean, digits = 4))
    println("true t: ", round(reg_payload.summary.t_true, digits = 4))
    println("mean HMC accept prob: ", round(reg_payload.summary.mean_acceptance, digits = 3))
    println("divergence rate: ", round(reg_payload.summary.divergence_rate, digits = 3))

    (; paths = merge(paths, (; run = run_paths)), run = run_payload)
end

function main()
    length(ARGS) == 1 || error("Usage: julia --project=. --startup-file=no run_simulation_fit.jl PARAMS.toml")
    run_engine(ARGS[1])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
