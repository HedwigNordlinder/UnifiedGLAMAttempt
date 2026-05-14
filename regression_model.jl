using LinearAlgebra, Random, Distributions

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

softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))
logsigmoid(x) = -softplus(-x)
log1msigmoid(x) = -softplus(x)
sigmoid(x) = x >= 0 ? inv(one(x) + exp(-x)) : exp(x) / (one(x) + exp(x))
logit(x) = log(x) - log1p(-x)

β_logprior(prior::UnivariateDistribution, β) = sum(logpdf.(Ref(prior), β))
β_logprior(prior::MultivariateDistribution, β) = logpdf(prior, β)
t_logprior(t) = 0.0 <= t <= 1.0 ? 0.0 : -Inf

function logits(data::SupervisedData, β::AbstractVector, t::Real)
    t .* (data.cluster_a_μ * β) .+ (1 - t) .* (data.cluster_b_μ * β)
end

function ℓ_prior(model::RegressionModel, β::AbstractVector)
    β_logprior(model.β_prior, β)
end

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

function ∇β_ℓ_prior(prior::Normal, β::AbstractVector)
    @. -(β - mean(prior)) / var(prior)
end

function ∇β_ℓ_prior(prior::MvNormal, β::AbstractVector)
    -(Matrix(invcov(prior)) * (β .- mean(prior)))
end

∇β_ℓ_prior(prior::Distribution, β::AbstractVector) =
    error("No analytic β-prior gradient implemented for $(typeof(prior)).")

function ∇β_ℓ_likelihood(data::SupervisedData, β::AbstractVector, t::Real)
    η = logits(data, β, t)
    r = similar(η)
    @inbounds for i in eachindex(r, η, data.y)
        r[i] = data.y[i] - sigmoid(η[i])
    end
    t .* (data.cluster_a_μ' * r) .+ (1 - t) .* (data.cluster_b_μ' * r)
end

function ∇t_ℓ_likelihood(data::SupervisedData, β::AbstractVector, t::Real)
    aβ = data.cluster_a_μ * β
    bβ = data.cluster_b_μ * β
    s = zero(promote_type(eltype(aβ), eltype(data.y)))
    @inbounds for i in eachindex(aβ, bβ, data.y)
        η = t * aβ[i] + (1 - t) * bβ[i]
        s += (data.y[i] - sigmoid(η)) * (aβ[i] - bβ[i])
    end
    s
end

mutable struct MALAState{T<:AbstractFloat}
    β::Vector{T}
    u::T
    loglik::T
    logpost::T
end

t(state::MALAState) = sigmoid(state.u)
snapshot(state::MALAState{T}) where {T} = MALAState(copy(state.β), state.u, state.loglik, state.logpost)

function mala_state(model::RegressionModel, data::SupervisedData, β::AbstractVector, t0::Real)
    0.0 < t0 < 1.0 || error("Initial t must lie strictly inside (0, 1).")
    β0 = collect(float.(β))
    t1 = float(t0)
    MALAState(β0, logit(t1), ℓ_likelihood(data, β0, t1), ℓ_posterior(model, data, β0, t1))
end

function transformed_logtarget(model::RegressionModel, data::SupervisedData, β::AbstractVector, u::Real)
    t = sigmoid(u)
    ℓ_posterior(model, data, β, t) + logsigmoid(u) + log1msigmoid(u)
end

function transformed_gradient(model::RegressionModel, data::SupervisedData, β::AbstractVector, u::Real)
    t = sigmoid(u)
    gβ = ∇β_ℓ_prior(model.β_prior, β) + ∇β_ℓ_likelihood(data, β, t)
    gu = t * (1 - t) * ∇t_ℓ_likelihood(data, β, t) + (1 - 2t)
    gβ, gu
end

gaussian_logkernel(x::AbstractVector, m::AbstractVector, step::Real) = -0.5 * sum(abs2, x .- m) / step^2
gaussian_logkernel(x::Real, m::Real, step::Real) = -0.5 * abs2(x - m) / step^2

function mala_step_β!(rng::AbstractRNG, model::RegressionModel, data::SupervisedData, state::MALAState; step::Real = 0.05)
    gβ, _ = transformed_gradient(model, data, state.β, state.u)
    m = state.β .+ 0.5 * step^2 .* gβ
    βp = m .+ step .* randn(rng, length(state.β))
    lp = transformed_logtarget(model, data, βp, state.u)
    gβp, _ = transformed_gradient(model, data, βp, state.u)
    mp = βp .+ 0.5 * step^2 .* gβp
    logα = lp - transformed_logtarget(model, data, state.β, state.u) +
           gaussian_logkernel(state.β, mp, step) - gaussian_logkernel(βp, m, step)
    if log(rand(rng)) < logα
        state.β .= βp
        state.loglik = ℓ_likelihood(data, state.β, t(state))
        state.logpost = ℓ_posterior(model, data, state.β, t(state))
        return true
    end
    false
end

function mala_step_t!(rng::AbstractRNG, model::RegressionModel, data::SupervisedData, state::MALAState; step::Real = 0.1)
    _, gu = transformed_gradient(model, data, state.β, state.u)
    m = state.u + 0.5 * step^2 * gu
    up = m + step * randn(rng)
    lp = transformed_logtarget(model, data, state.β, up)
    _, gup = transformed_gradient(model, data, state.β, up)
    mp = up + 0.5 * step^2 * gup
    logα = lp - transformed_logtarget(model, data, state.β, state.u) +
           gaussian_logkernel(state.u, mp, step) - gaussian_logkernel(up, m, step)
    if log(rand(rng)) < logα
        state.u = up
        state.loglik = ℓ_likelihood(data, state.β, t(state))
        state.logpost = ℓ_posterior(model, data, state.β, t(state))
        return true
    end
    false
end

function mala(rng::AbstractRNG, model::RegressionModel, data::SupervisedData;
              nsweeps = 1_000, burn = 0, thin = 1, init_β = zeros(size(data.cluster_a_μ, 2)),
              init_t = 0.5, step_β = 0.05, step_t = 0.1, save_states = false)
    state = mala_state(model, data, init_β, init_t)
    loglik = Vector{Float64}(undef, nsweeps)
    logpost = Vector{Float64}(undef, nsweeps)
    nkeep = burn < nsweeps ? cld(nsweeps - burn, thin) : 0
    states = save_states ? Vector{MALAState{Float64}}(undef, nkeep) : nothing
    saved = 0
    acc_β = 0
    acc_t = 0
    for it in 1:nsweeps
        acc_β += mala_step_β!(rng, model, data, state; step = step_β)
        acc_t += mala_step_t!(rng, model, data, state; step = step_t)
        loglik[it] = state.loglik
        logpost[it] = state.logpost
        if save_states && it > burn && (it - burn) % thin == 0
            saved += 1
            states[saved] = snapshot(state)
        end
    end
    (; state, loglik, logpost, states, accept_β = acc_β / nsweeps, accept_t = acc_t / nsweeps)
end
