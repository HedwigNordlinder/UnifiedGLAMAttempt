using LinearAlgebra, Random, Statistics, Distributions
using AdvancedHMC, LogDensityProblems, LogDensityProblemsAD, Zygote

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

LogDensityProblems.dimension(target::RegressionTarget) = size(target.data.cluster_a_μ, 2) + 1
LogDensityProblems.capabilities(::Type{<:RegressionTarget}) = LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(target::RegressionTarget, θ::AbstractVector{<:Real})
    p = size(target.data.cluster_a_μ, 2)
    length(θ) == p + 1 || error("Expected parameter vector of length $(p + 1), got $(length(θ)).")
    β = θ[1:p]
    u = θ[p + 1]
    transformed_logtarget(target.model, target.data, β, u)
end

mutable struct RegressionHMCState{T<:AbstractFloat}
    β::Vector{T}
    u::T
    loglik::T
    logpost::T
end

t(state::RegressionHMCState) = sigmoid(state.u)
snapshot(state::RegressionHMCState{T}) where {T} = RegressionHMCState(copy(state.β), state.u, state.loglik, state.logpost)

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
    AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(Val(:Zygote), target))
end

statfield(stat, name::Symbol, default) = hasproperty(stat, name) ? getproperty(stat, name) : default

function hmc(rng::AbstractRNG, model::RegressionModel, data::SupervisedData;
             nsweeps = 1_000, n_adapts = min(div(nsweeps, 2), 1_000), burn = n_adapts, thin = 1,
             init_β = zeros(size(data.cluster_a_μ, 2)), init_t = 0.5,
             target_accept = 0.8, max_depth = 10, save_states = false, save_chain = true,
             progress = false, verbose = false)
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
    draws = AdvancedHMC.AbstractMCMC.sample(
        rng,
        density_model,
        sampler,
        nsweeps;
        n_adapts = n_adapts,
        initial_params = θ0,
        progress = progress,
        verbose = verbose,
    )

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
        accept_trace[it] = Float64(statfield(stat, :acceptance_rate, NaN))
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
