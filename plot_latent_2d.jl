using Random, LinearAlgebra, Printf, Statistics, Distributions, Plots
include("latent_gibbs.jl")

gr()

rand_mvn(rng, μ, Σ) = μ + cholesky(Hermitian(Σ)).L * randn(rng, length(μ))
random_on_frame(rng, p, q = 2) = Matrix(qr(randn(rng, p, q)).Q)[:, 1:q]

function simulate_latent(rng; p = 100, npatients = 4, nobs = 50)
    B = random_on_frame(rng, p, 4)
    μ0 = (2.6 .* B[:, 1] .+ 1.2 .* B[:, 2], -2.3 .* B[:, 1] .+ 0.9 .* B[:, 3])
    Σ = ntuple(_ -> begin
        A = randn(rng, p, 4)
        Matrix(Symmetric(0.28 .* Matrix{Float64}(I, p, p) + 0.10 .* (A * A') / 4))
    end, 2)
    pi = rand(rng, Beta(2, 2), npatients)
    lambda = 0.7 .+ rand(rng, Gamma(4, 0.25), npatients, 2)
    X = Vector{Matrix{Float64}}(undef, npatients)
    z = Vector{BitVector}(undef, npatients)
    mu = Vector{Matrix{Float64}}(undef, npatients)
    for i in 1:npatients
        shift = 0.4 .* randn(rng) .* B[:, 4]
        M = hcat(μ0[1] .+ shift .+ 0.10 .* randn(rng, p), μ0[2] .+ shift .+ 0.10 .* randn(rng, p))
        Xi = Matrix{Float64}(undef, p, nobs)
        zi = BitVector(undef, nobs)
        for j in 1:nobs
            zi[j] = rand(rng) < pi[i]
            k = zi[j] ? 1 : 2
            Xi[:, j] = rand_mvn(rng, M[:, k], Σ[k] / lambda[i, k])
        end
        X[i], z[i], mu[i] = Xi, zi, M
    end
    data = LatentData(X)
    truth = GibbsState(LatentGMM(pi, mu, lambda, Σ), z)
    canonicalize!(truth, projection_basis(data))
    data, truth
end

function ellipse_points(μ, Σ; level = 0.8, nθ = 200)
    θ = range(0, 2π; length = nθ)
    r = sqrt(quantile(Chisq(2), level))
    A = r .* cholesky(Hermitian(Σ)).L
    hcat([μ .+ A * [cos(t), sin(t)] for t in θ]...)
end

function plot_sample(data, state, truth, U; path = "latent_100d_projection.png")
    npatients = length(data.X)
    rows = ceil(Int, npatients / 2)
    plt = plot(layout = (rows, 2), size = (900, 320 * rows), legend = false)
    levels = (0.5, 0.8, 0.95)
    colors = [:dodgerblue3, :tomato3]
    for i in 1:npatients
        Xi, zi = U' * data.X[i], state.z[i]
        title = @sprintf("Patient %d, pi=%.2f", i, state.model.pi[i])
        plot!(plt[i]; xlabel = "u1'x", ylabel = "u2'x", aspect_ratio = :equal, title)
        for k in 1:2
            keep = k == 1 ? zi : .!zi
            keep_true = k == 1 ? truth.z[i] : .!truth.z[i]
            scatter!(plt[i], Xi[1, keep_true], Xi[2, keep_true]; ms = 2, alpha = 0.15, color = colors[k])
            scatter!(plt[i], Xi[1, keep], Xi[2, keep]; ms = 4, alpha = 0.85, color = colors[k])
        end
        for k in 1:2
            Σhat = Symmetric(U' * (state.model.Sigma[k] / state.model.lambda[i, k]) * U)
            μhat = U' * state.model.mu[i][:, k]
            for lev in levels
                E = ellipse_points(μhat, Σhat; level = lev)
                plot!(plt[i], E[1, :], E[2, :], color = colors[k], lw = 2)
            end
            scatter!(plt[i], [μhat[1]], [μhat[2]], marker = :x, ms = 7, color = colors[k])
            Etrue = ellipse_points(U' * truth.model.mu[i][:, k], Symmetric(U' * (truth.model.Sigma[k] / truth.model.lambda[i, k]) * U); level = 0.8)
            plot!(plt[i], Etrue[1, :], Etrue[2, :], color = colors[k], ls = :dash, alpha = 0.4, lw = 1.5)
        end
    end
    savefig(plt, path)
    path
end

function main()
    rng = MersenneTwister(12)
    data, truth = simulate_latent(rng; p = 100)
    prior = default_prior(data)
    out = gibbs(rng, data, prior; nsweeps = 1_000)
    U = random_on_frame(rng, size(data.X[1], 1), 2)
    path = plot_sample(data, out.state, truth, U)
    @printf("saved %s\n", abspath(path))
    @printf("final observed-data loglik = %.2f\n", out.loglik[end])
    @printf("final complete-data logpost = %.2f\n", out.logpost[end])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
