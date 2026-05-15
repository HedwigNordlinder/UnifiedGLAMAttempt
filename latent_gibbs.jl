using LinearAlgebra, Random, Statistics, Distributions
using PDMats: PDMat
include("glam_stable.jl")

struct LatentPrior{T<:AbstractFloat}
    alpha0::T
    beta0::T
    m0::Vector{T}
    Lambda0::Matrix{T}     # prior precision for each patient/cluster mean
    a0::T
    b0::T                  # rate prior for lambda
    nu0::T
    S0::NTuple{2,Matrix{T}}
end

struct GibbsState{T<:AbstractFloat}
    model::LatentGMM{T}
    z::Vector{BitVector}   # true => cluster 1, false => cluster 2
end

empirical_cov(X) = begin
    n = size(X, 2)
    C = X .- mean(X, dims = 2)
    C * C' / max(n - 1, 1)
end

function projection_basis(data::LatentData{T}; ndirs::Int = min(3, size(data.X[1], 1))) where {T}
    E = eigen(Symmetric(empirical_cov(reduce(hcat, data.X))))
    ord = sortperm(E.values; rev = true)
    q = min(ndirs, length(ord))
    V = Matrix{T}(undef, size(E.vectors, 1), q)
    for ℓ in 1:q
        v = copy(E.vectors[:, ord[ℓ]])
        j = argmax(abs.(v))
        v[j] < 0 && (v .*= -one(T))
        V[:, ℓ] .= v
    end
    V
end

projection_direction(data::LatentData{T}) where {T} = vec(projection_basis(data; ndirs = 1))

function default_prior(data::LatentData{T}; alpha0 = one(T), beta0 = one(T), a0 = T(2), b0 = T(2)) where {T}
    Xall = reduce(hcat, data.X)
    p = size(Xall, 1)
    S = empirical_cov(Xall) + T(1e-6) * Matrix{T}(I, p, p)
    scale = max(tr(S) / p, eps(T))
    LatentPrior(alpha0, beta0, vec(mean(Xall, dims = 2)), Matrix{T}(I, p, p) * (T(0.01) / scale), a0, b0, T(p + 2), (copy(S), copy(S)))
end

function cluster_mean_centers(prior::LatentPrior{T}, cluster_mean_shift = zero(T)) where {T}
    p = length(prior.m0)
    shift = cluster_mean_shift isa AbstractVector ? collect(T, cluster_mean_shift) : fill(T(cluster_mean_shift), p)
    length(shift) == p || error("cluster_mean_shift must be scalar or length $p, got length $(length(shift)).")
    (prior.m0 .- shift, prior.m0 .+ shift)
end

priorcache(prior::LatentPrior{T}; cluster_mean_shift = zero(T)) where {T} = begin
    F0 = cholesky(Hermitian(prior.Lambda0))
    centers = cluster_mean_centers(prior, cluster_mean_shift)
    (; F0,
       centers,
       eta0 = ntuple(k -> prior.Lambda0 * centers[k], 2),
       logdet0 = 2sum(log, diag(F0.L)),
       beta = Beta(prior.alpha0, prior.beta0),
       gamma = Gamma(prior.a0, inv(prior.b0)),
       iw = ntuple(k -> InverseWishart(prior.nu0, PDMat(Symmetric(prior.S0[k]))), 2))
end

function rand_precision_normal(rng, Q, b)
    F = cholesky(Hermitian(Q))
    (F \ b) + (F.U \ randn(rng, length(b)))
end

function init_state(rng::AbstractRNG, data::LatentData{T}, prior::LatentPrior{T}) where {T}
    Xall = reduce(hcat, data.X)
    p = size(Xall, 1)
    S = empirical_cov(Xall) + T(1e-6) * Matrix{T}(I, p, p)
    d = T(0.25) .* sqrt.(diag(S))
    pi = Vector{T}(undef, length(data.X))
    mu = Vector{Matrix{T}}(undef, length(data.X))
    z = Vector{BitVector}(undef, length(data.X))
    for i in eachindex(data.X)
        Xi = data.X[i]
        ni = size(Xi, 2)
        zi = BitVector(vec(Xi[1, :]) .> median(vec(Xi[1, :])))
        n1 = count(zi)
        if n1 == 0 || n1 == ni
            zi .= falses(ni)
            zi[1:cld(ni, 2)] .= true
            n1 = count(zi)
        end
        z[i] = zi
        pi[i] = (n1 + T(0.5)) / (ni + one(T))
        m = vec(mean(Xi, dims = 2))
        M = Matrix{T}(undef, p, 2)
        M[:, 1] .= n1 > 0 ? vec(mean(Xi[:, zi], dims = 2)) : m .- d
        M[:, 2] .= n1 < ni ? vec(mean(Xi[:, .!zi], dims = 2)) : m .+ d
        mu[i] = M
    end
    GibbsState(LatentGMM(pi, mu, ones(T, length(data.X), 2), (copy(S), copy(S))), z)
end

snapshot(state::GibbsState{T}) where {T} = GibbsState(
    LatentGMM(copy(state.model.pi), [copy(M) for M in state.model.mu], copy(state.model.lambda), (copy(state.model.Sigma[1]), copy(state.model.Sigma[2]))),
    copy.(state.z),
)

function swap_labels!(state::GibbsState{T}) where {T}
    model = state.model
    model.pi .= one(T) .- model.pi
    tmpΣ = copy(model.Sigma[1])
    model.Sigma[1] .= model.Sigma[2]
    model.Sigma[2] .= tmpΣ
    tmpλ = copy(model.lambda[:, 1])
    model.lambda[:, 1] .= model.lambda[:, 2]
    model.lambda[:, 2] .= tmpλ
    for i in eachindex(model.mu)
        M = model.mu[i]
        tmpμ = copy(M[:, 1])
        M[:, 1] .= M[:, 2]
        M[:, 2] .= tmpμ
        state.z[i] .= .!state.z[i]
    end
    state
end

function contrast_preference(δ::AbstractVector{T}; tol = sqrt(eps(T))) where {T}
    npos = count(>(tol), δ)
    nneg = count(<(-tol), δ)
    npos != nneg && return sign(T(npos - nneg))
    scale = max(std(δ), one(T))
    med = median(δ)
    abs(med) > tol * scale && return sign(med)
    mn = mean(δ)
    abs(mn) > tol * scale && return sign(mn)
    zero(T)
end

function canonicalize!(state::GibbsState{T}, v::AbstractVector{T}; tol = sqrt(eps(T))) where {T}
    canonicalize!(state, reshape(v, :, 1); tol)
end

function canonicalize!(state::GibbsState{T}, V::AbstractMatrix{T}; tol = sqrt(eps(T))) where {T}
    ndirs = size(V, 2)
    npatients = length(state.model.mu)
    Δ = Matrix{T}(undef, ndirs, npatients)
    for i in 1:npatients
        Δ[:, i] .= V' * (view(state.model.mu[i], :, 2) - view(state.model.mu[i], :, 1))
    end

    pref = zero(T)
    for d in 1:ndirs
        pref = contrast_preference(view(Δ, d, :); tol)
        pref != 0 && break
    end

    if pref == 0
        p = size(state.model.mu[1], 1)
        m1, m2 = zeros(T, p), zeros(T, p)
        for M in state.model.mu
            m1 .+= view(M, :, 1)
            m2 .+= view(M, :, 2)
        end
        m1 ./= npatients
        m2 ./= npatients
        for d in 1:ndirs
            δ = dot(view(V, :, d), m2 - m1)
            if abs(δ) > tol * max(norm(view(V, :, d)) * max(norm(m1), norm(m2)), one(T))
                pref = sign(δ)
                break
            end
        end
        if pref == 0
            δ = tr(state.model.Sigma[2]) - tr(state.model.Sigma[1])
            if abs(δ) > tol * max(tr(state.model.Sigma[1]) + tr(state.model.Sigma[2]), one(T))
                pref = sign(δ)
            else
                for j in eachindex(m1)
                    δ = m2[j] - m1[j]
                    abs(δ) > tol || continue
                    pref = sign(δ)
                    break
                end
            end
        end
    end

    pref < 0 && swap_labels!(state)
    state
end

function complete_loglik(data::LatentData{T}, state::GibbsState{T}; cache = covcache(state.model)) where {T}
    s = zero(T)
    for i in eachindex(data.X)
        Xi, zi, mui = data.X[i], state.z[i], state.model.mu[i]
        p = clamp(state.model.pi[i], eps(T), one(T) - eps(T))
        n1 = count(zi)
        s += n1 * log(p) + (length(zi) - n1) * log1p(-p)
        @views for j in axes(Xi, 2)
            k = zi[j] ? 1 : 2
            s += gaussian_logpdf(Xi[:, j], mui[:, k], state.model.lambda[i, k], cache[k])
        end
    end
    s
end

function logprior(model::LatentGMM{T}, prior::LatentPrior{T}; pcache = priorcache(prior)) where {T}
    s = zero(T)
    for p in model.pi
        s += logpdf(pcache.beta, clamp(p, eps(T), one(T) - eps(T)))
    end
    for M in model.mu, k in 1:2
        r = M[:, k] - pcache.centers[k]
        v = pcache.F0.U * r
        s += 0.5 * (pcache.logdet0 - length(r) * log(2π) - dot(v, v))
    end
    for λ in model.lambda
        s += logpdf(pcache.gamma, λ)
    end
    for k in 1:2
        s += logpdf(pcache.iw[k], PDMat(Symmetric(model.Sigma[k])))
    end
    s
end

complete_logpost(data, state, prior; cache = covcache(state.model), pcache = priorcache(prior)) =
    complete_loglik(data, state; cache) + logprior(state.model, prior; pcache)

function gibbs_step!(rng::AbstractRNG, data::LatentData{T}, state::GibbsState{T}, prior::LatentPrior{T}; pcache = priorcache(prior)) where {T}
    model = state.model
    p = size(data.X[1], 1)
    cache = covcache(model)

    for i in eachindex(data.X)
        Xi, zi, mui = data.X[i], state.z[i], model.mu[i]
        q = clamp(model.pi[i], eps(T), one(T) - eps(T))
        @views for j in axes(Xi, 2)
            a = log(q) + gaussian_logpdf(Xi[:, j], mui[:, 1], model.lambda[i, 1], cache[1])
            b = log1p(-q) + gaussian_logpdf(Xi[:, j], mui[:, 2], model.lambda[i, 2], cache[2])
            zi[j] = rand(rng) < exp(a - logsumexp2(a, b))
        end
        n1 = count(zi)
        model.pi[i] = rand(rng, Beta(prior.alpha0 + n1, prior.beta0 + length(zi) - n1))
    end

    Sinv = ntuple(k -> Matrix(cache[k][1] \ Matrix{T}(I, p, p)), 2)
    for i in eachindex(data.X)
        Xi, zi, Mui = data.X[i], state.z[i], model.mu[i]
        for k in 1:2
            n = 0
            sx = zeros(T, p)
            @views for j in axes(Xi, 2)
                (zi[j] == (k == 1)) || continue
                sx .+= Xi[:, j]
                n += 1
            end
            Q = prior.Lambda0 + (n * model.lambda[i, k]) * Sinv[k]
            b = pcache.eta0[k] + model.lambda[i, k] * (Sinv[k] * sx)
            Mui[:, k] .= rand_precision_normal(rng, Q, b)
        end
    end

    for i in eachindex(data.X)
        Xi, zi, Mui = data.X[i], state.z[i], model.mu[i]
        for k in 1:2
            qsum, n = zero(T), 0
            @views for j in axes(Xi, 2)
                (zi[j] == (k == 1)) || continue
                r = Xi[:, j] - Mui[:, k]
                qsum += dot(r, cache[k][1] \ r)
                n += 1
            end
            shape = prior.a0 + T(0.5 * p * n)
            rate = prior.b0 + T(0.5) * qsum
            model.lambda[i, k] = rand(rng, Gamma(shape, inv(rate)))
        end
    end

    for k in 1:2
        S = copy(prior.S0[k])
        n = 0
        for i in eachindex(data.X)
            Xi, zi, Mui = data.X[i], state.z[i], model.mu[i]
            λ = model.lambda[i, k]
            @views for j in axes(Xi, 2)
                (zi[j] == (k == 1)) || continue
                r = Xi[:, j] - Mui[:, k]
                BLAS.ger!(λ, r, r, S)
                n += 1
            end
        end
        model.Sigma[k] .= rand(rng, InverseWishart(prior.nu0 + n, PDMat(Symmetric(S))))
    end
    state
end

function gibbs(rng::AbstractRNG, data::LatentData{T}, prior::LatentPrior{T};
               nsweeps = 1_000, burn = 0, thin = 1, init = nothing, save_states = false,
               postprocess = true, direction = nothing, cluster_mean_shift = zero(T)) where {T}
    state = init === nothing ? init_state(rng, data, prior) : snapshot(init)
    pcache = priorcache(prior; cluster_mean_shift)
    V = direction === nothing ? projection_basis(data) :
        direction isa AbstractVector ? reshape(T.(direction), :, 1) : Matrix{T}(direction)
    postprocess && canonicalize!(state, V)
    loglik = Vector{T}(undef, nsweeps)
    logpost = Vector{T}(undef, nsweeps)
    nkeep = burn < nsweeps ? cld(nsweeps - burn, thin) : 0
    states = save_states ? Vector{GibbsState{T}}(undef, nkeep) : nothing
    saved = 0
    for it in 1:nsweeps
        gibbs_step!(rng, data, state, prior; pcache)
        postprocess && canonicalize!(state, V)
        cache = covcache(state.model)
        loglik[it] = latent_loglik(data, state.model; cache)
        logpost[it] = complete_logpost(data, state, prior; cache, pcache)
        if save_states && it > burn && (it - burn) % thin == 0
            saved += 1
            states[saved] = snapshot(state)
        end
    end
    (; state, loglik, logpost, states)
end
