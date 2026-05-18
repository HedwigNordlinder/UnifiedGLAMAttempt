using LinearAlgebra

struct LatentData{T<:AbstractFloat}
    X::Vector{Matrix{T}}      # patient i: p x n_i, one observation per column
end

struct LatentGMM{T<:AbstractFloat}
    pi::Vector{T}             # patient-specific Pr(cluster 1)
    mu::Vector{Matrix{T}}     # patient i: p x 2 means
    lambda::Matrix{T}         # patient i, cluster k scale; Cov = Sigma[k] / lambda[i,k]
    Sigma::NTuple{2,Matrix{T}}
end

logsumexp2(a, b) = max(a, b) + log1p(exp(-abs(a - b)))

covcache(model::LatentGMM) = ntuple(k -> begin
    F = cholesky(Hermitian(model.Sigma[k]))
    (F, adjoint(UpperTriangular(F.factors)), 2sum(log, diag(F.factors)))
end, 2)

function gaussian_logpdf(x, mu, lambda, cache)
    _, L, logdet = cache
    z = L \ (x - mu)
    d = length(x)
    0.5 * d * log(lambda) - 0.5 * (d * log(2π) + logdet + lambda * dot(z, z))
end

function residual_col!(work::AbstractVector, X::AbstractMatrix, j::Integer, mu::AbstractVector)
    @inbounds for r in eachindex(work, mu)
        work[r] = X[r, j] - mu[r]
    end
    work
end

function residual_col!(work::AbstractVector, X::AbstractMatrix, j::Integer, M::AbstractMatrix, k::Integer)
    @inbounds for r in eachindex(work)
        work[r] = X[r, j] - M[r, k]
    end
    work
end

function quadform_col!(work::AbstractVector, X::AbstractMatrix, j::Integer, mu::AbstractVector, cache)
    L = cache[2]
    residual_col!(work, X, j, mu)
    ldiv!(L, work)
    dot(work, work)
end

function quadform_col!(work::AbstractVector, X::AbstractMatrix, j::Integer,
                       M::AbstractMatrix, k::Integer, cache)
    L = cache[2]
    residual_col!(work, X, j, M, k)
    ldiv!(L, work)
    dot(work, work)
end

function gaussian_logpdf_col!(work::AbstractVector, X::AbstractMatrix, j::Integer,
                              mu::AbstractVector, lambda, cache)
    logdet = cache[3]
    q = quadform_col!(work, X, j, mu, cache)
    d = length(work)
    0.5 * d * log(lambda) - 0.5 * (d * log(2π) + logdet + lambda * q)
end

function gaussian_logpdf_col!(work::AbstractVector, X::AbstractMatrix, j::Integer,
                              M::AbstractMatrix, k::Integer, lambda, cache)
    logdet = cache[3]
    q = quadform_col!(work, X, j, M, k, cache)
    d = length(work)
    0.5 * d * log(lambda) - 0.5 * (d * log(2π) + logdet + lambda * q)
end

function responsibilities(X, pi, mu, lambda, cache)
    T = eltype(X)
    p = clamp(pi, eps(T), one(T) - eps(T))
    r = Matrix{T}(undef, 2, size(X, 2))
    work1 = Vector{T}(undef, size(X, 1))
    work2 = similar(work1)
    lp1, lp2 = log(p), log1p(-p)
    for j in axes(X, 2)
        a = lp1 + gaussian_logpdf_col!(work1, X, j, mu, 1, lambda[1], cache[1])
        b = lp2 + gaussian_logpdf_col!(work2, X, j, mu, 2, lambda[2], cache[2])
        z = logsumexp2(a, b)
        r[1, j] = exp(a - z)
        r[2, j] = exp(b - z)
    end
    r
end

function latent_loglik(data::LatentData{T}, model::LatentGMM{T}; cache = covcache(model)) where {T}
    s = zero(T)
    work1 = Vector{T}(undef, size(data.X[1], 1))
    work2 = similar(work1)
    for i in eachindex(data.X)
        Xi, mui = data.X[i], model.mu[i]
        p = clamp(model.pi[i], eps(T), one(T) - eps(T))
        lp1, lp2 = log(p), log1p(-p)
        for j in axes(Xi, 2)
            a = lp1 + gaussian_logpdf_col!(work1, Xi, j, mui, 1, model.lambda[i, 1], cache[1])
            b = lp2 + gaussian_logpdf_col!(work2, Xi, j, mui, 2, model.lambda[i, 2], cache[2])
            s += logsumexp2(a, b)
        end
    end
    s
end

soft_cluster_means(X, r) = (r * X') ./ max.(sum(r, dims = 2), eps(eltype(X)))

function latent_summaries(data::LatentData{T}, model::LatentGMM{T}; cache = covcache(model)) where {T}
    [soft_cluster_means(data.X[i], responsibilities(data.X[i], model.pi[i], model.mu[i], view(model.lambda, i, :), cache))
     for i in eachindex(data.X)]
end
