using LinearAlgebra, Random, Statistics, Distributions, Logging
using AdvancedHMC, LogDensityProblems, LogDensityProblemsAD
using ADTypes, DifferentiationInterface, Mooncake
include("progress_helpers.jl")

const REGRESSION_AD_BACKEND = ADTypes.AutoMooncake()

struct SupervisedData{T<:AbstractFloat,Y<:Real}
    cluster_a_μ::Matrix{T}   # n x p
    cluster_b_μ::Matrix{T}   # n x p
    y::Vector{Y}             # 0/1
end

SupervisedData(a::Vector{<:AbstractVector{T}}, b::Vector{<:AbstractVector{T}}, y::AbstractVector{<:Real}) where {T<:AbstractFloat} =
    SupervisedData(Matrix(reduce(hcat, a)'), Matrix(reduce(hcat, b)'), collect(y))

struct RegressionModel{P<:Distribution}
    β_prior::P
end

struct GammaRegressionModel{P<:UnivariateDistribution,T<:AbstractFloat}
    β_prior::P
    gamma_prior::T
end

const LOG2PI = log(2 * pi)

softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))
logsigmoid(x) = -softplus(-x)
log1msigmoid(x) = -softplus(x)
sigmoid(x) = x >= 0 ? inv(one(x) + exp(-x)) : exp(x) / (one(x) + exp(x))
logit(x) = log(x) - log1p(-x)

function β_logprior(prior::Normal, β::AbstractVector)
    μ = mean(prior)
    σ = std(prior)
    z = (β .- μ) ./ σ
    -0.5 * sum(abs2, z) - length(β) * (log(σ) + 0.5 * LOG2PI)
end

function β_logprior(prior::MvNormal, β::AbstractVector)
    δ = β .- mean(prior)
    -0.5 * (dot(δ, invcov(prior) * δ) + length(β) * LOG2PI + logdetcov(prior))
end

β_logprior(prior::UnivariateDistribution, β::AbstractVector) = sum(logpdf.(Ref(prior), β))
β_logprior(prior::MultivariateDistribution, β::AbstractVector) = logpdf(prior, β)
t_logprior(t) = 0.0 <= t <= 1.0 ? 0.0 : -Inf

function logits(data::SupervisedData, β::AbstractVector, t::Real)
    t .* (data.cluster_a_μ * β) .+ (1 - t) .* (data.cluster_b_μ * β)
end

function logits(data::SupervisedData, β::AbstractVector, gamma::AbstractVector{Bool}, t::Real)
    logits(data, β .* gamma, t)
end

ℓ_prior(model::RegressionModel, β::AbstractVector) = β_logprior(model.β_prior, β)

function ℓ_likelihood(data::SupervisedData, β::AbstractVector, t::Real)
    η = logits(data, β, t)
    s = zero(promote_type(eltype(η), eltype(data.y)))
    @inbounds for i in eachindex(η, data.y)
        s += data.y[i] * logsigmoid(η[i]) + (1 - data.y[i]) * log1msigmoid(η[i])
    end
    s
end

function ℓ_posterior(model::RegressionModel, data::SupervisedData, β::AbstractVector, t::Real)
    t_logprior(t) + ℓ_prior(model, β) + ℓ_likelihood(data, β, t)
end

function gamma_logprior(model::GammaRegressionModel, gamma::AbstractVector{Bool})
    p = clamp(model.gamma_prior, eps(typeof(model.gamma_prior)), one(model.gamma_prior) - eps(typeof(model.gamma_prior)))
    count(gamma) * log(p) + (length(gamma) - count(gamma)) * log1p(-p)
end

function ℓ_prior(model::GammaRegressionModel, β::AbstractVector, gamma::AbstractVector{Bool})
    β_logprior(model.β_prior, β[gamma]) + gamma_logprior(model, gamma)
end

function ℓ_likelihood(data::SupervisedData, β::AbstractVector, gamma::AbstractVector{Bool}, t::Real)
    η = logits(data, β, gamma, t)
    s = zero(promote_type(eltype(η), eltype(data.y)))
    @inbounds for i in eachindex(η, data.y)
        s += data.y[i] * logsigmoid(η[i]) + (1 - data.y[i]) * log1msigmoid(η[i])
    end
    s
end

function ℓ_posterior(model::GammaRegressionModel, data::SupervisedData, β::AbstractVector, gamma::AbstractVector{Bool}, t::Real)
    t_logprior(t) + ℓ_prior(model, β, gamma) + ℓ_likelihood(data, β, gamma, t)
end

function transformed_logtarget(model::RegressionModel, data::SupervisedData, β::AbstractVector, u::Real)
    t = sigmoid(u)
    ℓ_posterior(model, data, β, t) + logsigmoid(u) + log1msigmoid(u)
end

supports_hmc_prior(::Normal) = true
supports_hmc_prior(::MvNormal) = true
supports_hmc_prior(::Distribution) = false

struct RegressionTarget{M<:RegressionModel,D<:SupervisedData}
    model::M
    data::D
end

struct GammaActiveTarget{P<:UnivariateDistribution,T<:AbstractFloat,Y<:Real}
    β_prior::P
    A::Matrix{T}
    B::Matrix{T}
    y::Vector{Y}
end

LogDensityProblems.dimension(target::RegressionTarget) = size(target.data.cluster_a_μ, 2) + 1
LogDensityProblems.capabilities(::Type{<:RegressionTarget}) = LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(target::RegressionTarget, θ::AbstractVector{<:Real})
    p = size(target.data.cluster_a_μ, 2)
    length(θ) == p + 1 || error("Expected parameter vector of length $(p + 1), got $(length(θ)).")
    β = θ[1:p]
    u = θ[p + 1]
    transformed_logtarget(target.model, target.data, β, u)
end

LogDensityProblems.dimension(target::GammaActiveTarget) = size(target.A, 2) + 1
LogDensityProblems.capabilities(::Type{<:GammaActiveTarget}) = LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(target::GammaActiveTarget, θ::AbstractVector{<:Real})
    p = size(target.A, 2)
    length(θ) == p + 1 || error("Expected parameter vector of length $(p + 1), got $(length(θ)).")
    β = θ[1:p]
    u = θ[p + 1]
    tt = sigmoid(u)
    η = tt .* (target.A * β) .+ (1 - tt) .* (target.B * β)
    s = β_logprior(target.β_prior, β) + logsigmoid(u) + log1msigmoid(u)
    @inbounds for i in eachindex(η, target.y)
        s += target.y[i] * logsigmoid(η[i]) + (1 - target.y[i]) * log1msigmoid(η[i])
    end
    s
end

mutable struct RegressionHMCState{T<:AbstractFloat}
    β::Vector{T}
    u::T
    loglik::T
    logpost::T
end

mutable struct GammaRegressionHMCState{T<:AbstractFloat}
    β::Vector{T}
    gamma::BitVector
    u::T
    loglik::T
    logpost::T
end

t(state::RegressionHMCState) = sigmoid(state.u)
t(state::GammaRegressionHMCState) = sigmoid(state.u)
snapshot(state::RegressionHMCState{T}) where {T} = RegressionHMCState(copy(state.β), state.u, state.loglik, state.logpost)
snapshot(state::GammaRegressionHMCState{T}) where {T} =
    GammaRegressionHMCState(copy(state.β), copy(state.gamma), state.u, state.loglik, state.logpost)

function hmc_state(model::RegressionModel, data::SupervisedData, θ::AbstractVector)
    p = size(data.cluster_a_μ, 2)
    β = collect(float.(θ[1:p]))
    u = float(θ[p + 1])
    tt = sigmoid(u)
    RegressionHMCState(β, u, ℓ_likelihood(data, β, tt), ℓ_posterior(model, data, β, tt))
end

function regression_logdensity_model(model::RegressionModel, data::SupervisedData)
    supports_hmc_prior(model.β_prior) || error("Joint HMC currently supports Normal or MvNormal priors for β.")
    target = RegressionTarget(model, data)
    AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(REGRESSION_AD_BACKEND, target))
end

statfield(stat, name::Symbol, default) = hasproperty(stat, name) ? getproperty(stat, name) : default

function hmc_acceptance_probability(stat)
    raw = Float64(statfield(stat, :acceptance_rate, NaN))
    isfinite(raw) || return raw
    clamp(raw, 0.0, 1.0)
end

function push_hmc_stats!(accept::Vector{Float64}, tree_depth::Vector{Int},
                         numerical_error::BitVector, transition)
    stat = transition.stat
    push!(accept, hmc_acceptance_probability(stat))
    push!(tree_depth, Int(statfield(stat, :tree_depth, 0)))
    push!(numerical_error, Bool(statfield(stat, :numerical_error, false)))
    nothing
end

function run_hmc_steps(rng::AbstractRNG, density_model, sampler, θ0::AbstractVector;
                       nsteps::Int, n_adapts::Int, progress::Bool, verbose::Bool,
                       progress_label::AbstractString = "",
                       progress_every::Int = max(1, nsteps ÷ 100),
                       save_transitions::Bool = false)
    accept = Float64[]
    tree_depth = Int[]
    numerical_error = BitVector()
    transitions = save_transitions ? Vector{Any}(undef, nsteps) : nothing
    prog = SimpleProgress(progress_label, nsteps; enabled = progress && !isempty(progress_label), every = progress_every)
    progress_update!(prog, 0; force = true)
    with_logger(NullLogger()) do
        transition, hmc_state = AdvancedHMC.AbstractMCMC.step(
            rng,
            density_model,
            sampler;
            initial_params = θ0,
            n_adapts,
            progress,
            verbose,
        )
        push_hmc_stats!(accept, tree_depth, numerical_error, transition)
        save_transitions && (transitions[1] = transition)
        progress_update!(prog, 1; suffix = "accept_prob=$(round(accept[end]; digits=3)) depth=$(tree_depth[end])")
        for _ in 2:nsteps
            transition, hmc_state = AdvancedHMC.AbstractMCMC.step(
                rng,
                density_model,
                sampler,
                hmc_state;
                n_adapts,
                progress,
                verbose,
            )
            push_hmc_stats!(accept, tree_depth, numerical_error, transition)
            i = length(accept)
            save_transitions && (transitions[i] = transition)
            progress_update!(prog, i; suffix = "accept_prob=$(round(accept[end]; digits=3)) depth=$(tree_depth[end])")
        end
        (; transition, hmc_state, accept, tree_depth, numerical_error, transitions)
    end
end

function inverse_metric_diag(metric, dim::Int)
    Minv = getproperty(metric, Symbol("M⁻¹"))
    if Minv isa UniformScaling
        fill(Float64(Minv.λ), dim)
    elseif Minv isa AbstractVector
        collect(Float64, Minv)
    elseif Minv isa AbstractMatrix
        collect(Float64, diag(Minv))
    else
        error("Unsupported metric inverse type: $(typeof(Minv)).")
    end
end

function gamma_density_model(model::GammaRegressionModel, data::SupervisedData, active::AbstractVector{Int})
    target = GammaActiveTarget(model.β_prior,
                               Matrix(data.cluster_a_μ[:, active]),
                               Matrix(data.cluster_b_μ[:, active]),
                               data.y)
    AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(REGRESSION_AD_BACKEND, target))
end

function gamma_hmc_state(model::GammaRegressionModel, data::SupervisedData, β::AbstractVector, gamma::BitVector, u::Real)
    tt = sigmoid(u)
    GammaRegressionHMCState(collect(float.(β)), copy(gamma),
                            float(u),
                            ℓ_likelihood(data, β, gamma, tt),
                            ℓ_posterior(model, data, β, gamma, tt))
end

function sample_gamma!(rng::AbstractRNG, model::GammaRegressionModel, data::SupervisedData,
                       β::AbstractVector, gamma::BitVector, u::Real, j::Int)
    tt = sigmoid(u)
    η = logits(data, β, gamma, tt)
    xj = tt .* view(data.cluster_a_μ, :, j) .+ (1 - tt) .* view(data.cluster_b_μ, :, j)
    δ = β[j] .* xj
    η_on = gamma[j] ? η : η .+ δ
    η_off = gamma[j] ? η .- δ : η

    loglik_on = zero(eltype(η))
    loglik_off = zero(eltype(η))
    @inbounds for i in eachindex(η, data.y)
        y = data.y[i]
        loglik_on += y * logsigmoid(η_on[i]) + (1 - y) * log1msigmoid(η_on[i])
        loglik_off += y * logsigmoid(η_off[i]) + (1 - y) * log1msigmoid(η_off[i])
    end

    pγ = clamp(model.gamma_prior, eps(typeof(model.gamma_prior)), one(model.gamma_prior) - eps(typeof(model.gamma_prior)))
    log_on = log(pγ) + logpdf(model.β_prior, β[j]) + loglik_on
    log_off = log1p(-pγ) + loglik_off
    prob_on = sigmoid(log_on - log_off)
    gamma[j] = rand(rng) < prob_on
    prob_on
end

function gamma_initial_warmup!(rng::AbstractRNG, model::GammaRegressionModel, data::SupervisedData,
                               β::Vector{Float64}, u::Float64;
                               nsteps = 300, n_adapts = min(250, nsteps - 1),
                               target_accept = 0.8, max_depth = 10,
                               progress = false, verbose = false)
    p = length(β)
    active = collect(1:p)
    density_model = gamma_density_model(model, data, active)
    sampler = NUTS(target_accept; max_depth)
    θ0 = vcat(β, u)
    draws = run_hmc_steps(rng, density_model, sampler, θ0;
                          nsteps, n_adapts, progress, verbose)

    θ = draws.transition.z.θ
    β .= θ[1:p]
    post = min(n_adapts + 1, nsteps):nsteps
    metric_diag = inverse_metric_diag(AdvancedHMC.getmetric(draws.hmc_state), p + 1)
    step_size = Float64(AdvancedHMC.step_size(AdvancedHMC.getintegrator(draws.hmc_state)))
    (; u = Float64(θ[p + 1]),
       metric_diag,
       step_size,
       accept = mean(draws.accept[post]),
       tree_depth = draws.tree_depth[end],
       numerical_error = any(draws.numerical_error[post]))
end

function gamma_hmc_chunk!(rng::AbstractRNG, model::GammaRegressionModel, data::SupervisedData,
                          β::Vector{Float64}, gamma::BitVector, u::Float64;
                          nsteps = 5, n_adapts = 0, target_accept = 0.8, max_depth = 10,
                          base_metric_diag = nothing, base_step_size = nothing,
                          progress = false, verbose = false)
    active = findall(gamma)
    density_model = gamma_density_model(model, data, active)
    if base_metric_diag === nothing
        sampler = NUTS(target_accept; max_depth)
    else
        p = length(β)
        θ_index = vcat(active, p + 1)
        metric = AdvancedHMC.DiagEuclideanMetric(collect(Float64, base_metric_diag[θ_index]))
        integrator = base_step_size === nothing ? :leapfrog : AdvancedHMC.Leapfrog(Float64(base_step_size))
        sampler = NUTS(target_accept; max_depth, metric, integrator)
    end
    θ0 = vcat(β[active], u)
    draws = run_hmc_steps(rng, density_model, sampler, θ0;
                          nsteps, n_adapts, progress, verbose)

    θ = draws.transition.z.θ
    β[active] .= θ[1:length(active)]
    post = min(n_adapts + 1, nsteps):nsteps
    (; u = Float64(θ[length(active) + 1]),
       step_size = Float64(AdvancedHMC.step_size(AdvancedHMC.getintegrator(draws.hmc_state))),
       accept = mean(draws.accept[post]),
       tree_depth = draws.tree_depth[end],
       numerical_error = any(draws.numerical_error[post]))
end

function gamma_hmc(rng::AbstractRNG, model::GammaRegressionModel, data::SupervisedData;
                   nsamples = 500, initial_hmc = 300,
                   initial_adapts = min(250, initial_hmc - 1),
                   hmc_steps_per_gamma = 5, hmc_adapts_per_gamma = 0,
                   reuse_initial_metric = true,
                   step_size_adapt_rate = 0.0,
                   step_size_min_factor = 0.25,
                   step_size_max_factor = 2.0,
                   init_β = zeros(size(data.cluster_a_μ, 2)), init_gamma = trues(size(data.cluster_a_μ, 2)),
                   init_t = 0.5, target_accept = 0.8, max_depth = 10, save_states = false,
                   save_chain = true, progress = false, verbose = false,
                   progress_every = max(1, nsamples ÷ 100))
    0.0 < init_t < 1.0 || error("Initial t must lie strictly inside (0, 1).")
    nsamples >= 1 || error("nsamples must be positive.")
    initial_hmc >= 1 || error("initial_hmc must be positive.")
    0 <= initial_adapts < initial_hmc || error("initial_adapts must satisfy 0 <= initial_adapts < initial_hmc.")
    hmc_steps_per_gamma >= 1 || error("hmc_steps_per_gamma must be positive.")
    0 <= hmc_adapts_per_gamma < hmc_steps_per_gamma || error("hmc_adapts_per_gamma must satisfy 0 <= hmc_adapts_per_gamma < hmc_steps_per_gamma.")
    step_size_adapt_rate >= 0 || error("step_size_adapt_rate must be non-negative.")
    0 < step_size_min_factor <= step_size_max_factor || error("Require 0 < step_size_min_factor <= step_size_max_factor.")
    p = size(data.cluster_a_μ, 2)
    length(init_β) == p || error("Expected init_β to have length $p, got $(length(init_β)).")
    length(init_gamma) == p || error("Expected init_gamma to have length $p, got $(length(init_gamma)).")

    β = collect(Float64, init_β)
    gamma = BitVector(init_gamma)
    u = logit(float(init_t))
    gamma .= true
    if progress
        println("gamma regression warmup: all gammas on, initial_hmc=$initial_hmc, initial_adapts=$initial_adapts")
        flush(stdout)
    end
    warm = gamma_initial_warmup!(rng, model, data, β, u;
                                 nsteps = initial_hmc, n_adapts = initial_adapts,
                                 target_accept, max_depth, progress, verbose)
    u = warm.u
    gamma .= init_gamma
    base_metric_diag = reuse_initial_metric ? warm.metric_diag : nothing
    base_step_size = reuse_initial_metric ? warm.step_size : nothing
    adaptive_step_size = base_step_size
    min_step_size = base_step_size === nothing ? nothing : step_size_min_factor * base_step_size
    max_step_size = base_step_size === nothing ? nothing : step_size_max_factor * base_step_size

    loglik = Vector{Float64}(undef, nsamples)
    logpost = Vector{Float64}(undef, nsamples)
    t_trace = Vector{Float64}(undef, nsamples)
    accept_trace = Vector{Float64}(undef, nsamples)
    step_size_trace = Vector{Float64}(undef, nsamples)
    tree_depth = Vector{Int}(undef, nsamples)
    numerical_error = BitVector(undef, nsamples)
    gamma_index = Vector{Int}(undef, nsamples)
    gamma_prob = Vector{Float64}(undef, nsamples)
    active_trace = Vector{Int}(undef, nsamples)
    states = save_states ? Vector{GammaRegressionHMCState{Float64}}(undef, nsamples) : nothing
    β_chain = save_chain ? Matrix{Float64}(undef, nsamples, p) : nothing
    gamma_chain = save_chain ? BitMatrix(undef, nsamples, p) : nothing
    t_chain = save_chain ? Vector{Float64}(undef, nsamples) : nothing
    prog = SimpleProgress("gamma regression", nsamples; enabled = progress, every = progress_every)
    progress_update!(prog, 0; force = true, suffix = "p=$p hmc_steps_per_gamma=$hmc_steps_per_gamma")

    for it in 1:nsamples
        j = rand(rng, 1:p)
        gamma_index[it] = j
        gamma_prob[it] = sample_gamma!(rng, model, data, β, gamma, u, j)
        hmc_stats = gamma_hmc_chunk!(rng, model, data, β, gamma, u;
                                     nsteps = hmc_steps_per_gamma, n_adapts = hmc_adapts_per_gamma,
                                     target_accept, max_depth, base_metric_diag, base_step_size = adaptive_step_size,
                                     progress, verbose)
        u = hmc_stats.u
        if adaptive_step_size !== nothing && step_size_adapt_rate > 0
            adaptive_step_size = clamp(adaptive_step_size * exp(step_size_adapt_rate * (hmc_stats.accept - target_accept)),
                                       min_step_size, max_step_size)
        end
        tt = sigmoid(u)
        loglik[it] = ℓ_likelihood(data, β, gamma, tt)
        logpost[it] = ℓ_posterior(model, data, β, gamma, tt)
        t_trace[it] = tt
        accept_trace[it] = hmc_stats.accept
        step_size_trace[it] = hmc_stats.step_size
        tree_depth[it] = hmc_stats.tree_depth
        numerical_error[it] = hmc_stats.numerical_error
        active_trace[it] = count(gamma)
        progress_update!(prog, it; suffix = "t=$(round(tt; digits=3)) active=$(active_trace[it]) accept_prob=$(round(accept_trace[it]; digits=3))")
        if save_states
            states[it] = GammaRegressionHMCState(copy(β), copy(gamma), u, loglik[it], logpost[it])
        end
        if save_chain
            β_chain[it, :] .= β
            gamma_chain[it, :] .= gamma
            t_chain[it] = tt
        end
    end

    state = gamma_hmc_state(model, data, β, gamma, u)
    (; state, loglik, logpost, t_trace, states, β_chain, gamma_chain, t_chain,
       gamma_index, gamma_prob, accept_trace, step_size_trace, tree_depth, numerical_error,
       initial_metric_diag = warm.metric_diag,
       initial_step_size = warm.step_size,
       warmup_acceptance = warm.accept,
       warmup_numerical_error = warm.numerical_error,
       mean_acceptance = mean(accept_trace),
       divergence_rate = mean(Float64.(numerical_error)),
       mean_step_size = mean(step_size_trace),
       mean_tree_depth = mean(Float64.(tree_depth)),
       active_trace)
end

function hmc(rng::AbstractRNG, model::RegressionModel, data::SupervisedData;
             nsweeps = 1_000, n_adapts = min(div(nsweeps, 2), 1_000), burn = n_adapts, thin = 1,
             init_β = zeros(size(data.cluster_a_μ, 2)), init_t = 0.5,
             target_accept = 0.8, max_depth = 10, save_states = false, save_chain = true,
             progress = false, verbose = false, progress_every = max(1, nsweeps ÷ 100))
    0.0 < init_t < 1.0 || error("Initial t must lie strictly inside (0, 1).")
    1 <= nsweeps || error("nsweeps must be positive.")
    0 <= n_adapts < nsweeps || error("n_adapts must satisfy 0 <= n_adapts < nsweeps.")
    0 <= burn < nsweeps || error("burn must satisfy 0 <= burn < nsweeps.")
    thin >= 1 || error("thin must be at least 1.")

    p = size(data.cluster_a_μ, 2)
    length(init_β) == p || error("Expected init_β to have length $p, got $(length(init_β)).")

    density_model = regression_logdensity_model(model, data)
    sampler = NUTS(target_accept; max_depth = max_depth)
    θ0 = vcat(collect(float.(init_β)), logit(float(init_t)))
    hmc_run = run_hmc_steps(rng, density_model, sampler, θ0;
                            nsteps = nsweeps, n_adapts,
                            progress, verbose,
                            progress_label = "dense regression",
                            progress_every,
                            save_transitions = true)
    draws = hmc_run.transitions

    loglik = Vector{Float64}(undef, nsweeps)
    logpost = Vector{Float64}(undef, nsweeps)
    logtarget = Vector{Float64}(undef, nsweeps)
    t_trace = Vector{Float64}(undef, nsweeps)
    accept_trace = Vector{Float64}(undef, nsweeps)
    step_size = Vector{Float64}(undef, nsweeps)
    nom_step_size = Vector{Float64}(undef, nsweeps)
    tree_depth = Vector{Int}(undef, nsweeps)
    numerical_error = BitVector(undef, nsweeps)
    is_adapt = BitVector(undef, nsweeps)

    nkeep = cld(nsweeps - burn, thin)
    states = save_states ? Vector{RegressionHMCState{Float64}}(undef, nkeep) : nothing
    β_chain = save_chain ? Matrix{Float64}(undef, nkeep, p) : nothing
    t_chain = save_chain ? Vector{Float64}(undef, nkeep) : nothing
    saved = 0

    for it in 1:nsweeps
        draw = draws[it]
        θ = draw.z.θ
        β = θ[1:p]
        u = θ[p + 1]
        tt = sigmoid(u)
        stat = draw.stat

        loglik[it] = ℓ_likelihood(data, β, tt)
        logpost[it] = ℓ_posterior(model, data, β, tt)
        logtarget[it] = transformed_logtarget(model, data, β, u)
        t_trace[it] = tt
        accept_trace[it] = hmc_acceptance_probability(stat)
        step_size[it] = Float64(statfield(stat, :step_size, NaN))
        nom_step_size[it] = Float64(statfield(stat, :nom_step_size, NaN))
        tree_depth[it] = Int(statfield(stat, :tree_depth, 0))
        numerical_error[it] = Bool(statfield(stat, :numerical_error, false))
        is_adapt[it] = Bool(statfield(stat, :is_adapt, false))

        if it > burn && (it - burn) % thin == 0
            saved += 1
            if save_states
                states[saved] = RegressionHMCState(copy(β), u, loglik[it], logpost[it])
            end
            if save_chain
                β_chain[saved, :] .= β
                t_chain[saved] = tt
            end
        end
    end

    post_start = n_adapts < nsweeps ? n_adapts + 1 : nsweeps
    state = hmc_state(model, data, draws[end].z.θ)
    (; state, loglik, logpost, logtarget, t_trace, states, β_chain, t_chain,
       accept_trace, step_size, nom_step_size, tree_depth, numerical_error, is_adapt,
       mean_acceptance = mean(accept_trace[post_start:end]),
       divergence_rate = mean(Float64.(numerical_error[post_start:end])),
       mean_tree_depth = mean(Float64.(tree_depth[post_start:end])))
end
