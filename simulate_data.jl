using LinearAlgebra, Random, Statistics, Serialization, Distributions, Plots
include("latent_gibbs.jl")
include("regression_model.jl")

const HAVE_JLD2 = Base.find_package("JLD2") !== nothing
if HAVE_JLD2
    @eval import JLD2
end

struct SimulationConfig{T<:AbstractFloat,P<:Distribution}
    npatients::Int
    nobs_range::UnitRange{Int}
    latent_prior::LatentPrior{T}
    β_prior::P
    t_prior::Uniform{T}
end

struct RegressionTruth{T<:AbstractFloat}
    β::Vector{T}
    t::T
    η::Vector{T}
    p::Vector{T}
end

function default_simulation_config(p::Int; npatients = 120, nobs_range = 8:20)
    T = Float64
    SimulationConfig(
        npatients,
        nobs_range,
        LatentPrior(T(2), T(2), zeros(T, p), T(0.5) * Matrix{T}(I, p, p), T(2), T(2), T(p + 3), (Matrix{T}(I, p, p), Matrix{T}(I, p, p))),
        Normal(T(0), T(1)),
        Uniform(T(0), T(1)),
    )
end

rand_mvn(rng, μ, Σ) = μ + cholesky(Hermitian(Σ)).L * randn(rng, length(μ))

rand_β(rng::AbstractRNG, prior::UnivariateDistribution, p::Int) = rand(rng, prior, p)
rand_β(rng::AbstractRNG, prior::MultivariateDistribution, p::Int) = begin
    β = rand(rng, prior)
    length(β) == p || error("Multivariate β prior dimension $(length(β)) does not match p=$p.")
    collect(β)
end
rand_β(rng::AbstractRNG, prior::Distribution, p::Int) =
    error("No β sampler implemented for $(typeof(prior)).")

function simulate_latent_truth(rng::AbstractRNG, config::SimulationConfig{T}) where {T}
    prior = config.latent_prior
    p = length(prior.m0)
    Sigma = ntuple(k -> Matrix(rand(rng, InverseWishart(prior.nu0, PDMat(Symmetric(prior.S0[k]))))), 2)
    pi = rand(rng, Beta(prior.alpha0, prior.beta0), config.npatients)
    lambda = rand(rng, Gamma(prior.a0, inv(prior.b0)), config.npatients, 2)
    mu = Vector{Matrix{T}}(undef, config.npatients)
    z = Vector{BitVector}(undef, config.npatients)
    X = Vector{Matrix{T}}(undef, config.npatients)
    for i in 1:config.npatients
        ni = rand(rng, config.nobs_range)
        zi = rand(rng, Bernoulli(pi[i]), ni) .== 1
        z[i] = BitVector(zi)
        Mi = Matrix{T}(undef, p, 2)
        for k in 1:2
            Mi[:, k] .= rand_precision_normal(rng, prior.Lambda0, prior.Lambda0 * prior.m0)
        end
        mu[i] = Mi
        Xi = Matrix{T}(undef, p, ni)
        @views for j in 1:ni
            k = z[i][j] ? 1 : 2
            Xi[:, j] = rand_mvn(rng, Mi[:, k], Sigma[k] / lambda[i, k])
        end
        X[i] = Xi
    end
    data = LatentData(X)
    truth = GibbsState(LatentGMM(pi, mu, lambda, Sigma), z)
    canonicalize!(truth, projection_direction(data))
    data, truth
end

function simulate_regression_truth(rng::AbstractRNG, config::SimulationConfig{T}, truth::GibbsState{T}) where {T}
    p = size(truth.model.mu[1], 1)
    β = collect(T, rand_β(rng, config.β_prior, p))
    t = rand(rng, config.t_prior)
    η = Vector{T}(undef, length(truth.model.mu))
    q = similar(η)
    y = Vector{Int}(undef, length(η))
    A = Matrix{T}(undef, length(η), p)
    B = similar(A)
    for i in eachindex(truth.model.mu)
        A[i, :] .= truth.model.mu[i][:, 1]
        B[i, :] .= truth.model.mu[i][:, 2]
        η[i] = t * dot(β, truth.model.mu[i][:, 1]) + (1 - t) * dot(β, truth.model.mu[i][:, 2])
        q[i] = sigmoid(η[i])
        y[i] = rand(rng, Bernoulli(q[i]))
    end
    SupervisedData(A, B, y), RegressionTruth(β, t, η, q)
end

function patient_stats(data::LatentData{T}, truth::GibbsState{T}, regtruth::RegressionTruth{T}, supervised::SupervisedData{T}) where {T}
    nobs = [size(Xi, 2) for Xi in data.X]
    realized_share = [mean(zi) for zi in truth.z]
    y = Float64.(supervised.y)
    (; nobs,
       realized_share,
       mean_nobs = mean(nobs),
       sd_nobs = std(nobs),
       mean_pi = mean(truth.model.pi),
       sd_pi = std(truth.model.pi),
       outcome_rate = mean(y),
       lambda_mean = vec(mean(truth.model.lambda, dims = 1)),
       lambda_sd = vec(std(truth.model.lambda, dims = 1)),
       eta_mean = mean(regtruth.η),
       eta_sd = std(regtruth.η))
end

function plot_simulation_summary(data::LatentData, truth::GibbsState, regtruth::RegressionTruth, supervised::SupervisedData;
                                 path = "simulation_summary.png")
    stats = patient_stats(data, truth, regtruth, supervised)
    plt = plot(layout = (2, 3), size = (1200, 700), legend = false)

    histogram!(plt[1], truth.model.pi; bins = 20, color = :steelblue3,
               title = "Patient pi\nmean=$(round(stats.mean_pi, digits=2)), sd=$(round(stats.sd_pi, digits=2))",
               xlabel = "pi_i", ylabel = "count")

    scatter!(plt[2], truth.model.pi, stats.realized_share; color = :black, alpha = 0.7,
             title = "Realized share vs pi", xlabel = "pi_i", ylabel = "mean(z_i)")
    plot!(plt[2], [0, 1], [0, 1], color = :tomato3, lw = 2, ls = :dash)

    counts = [count(==(0), supervised.y), count(==(1), supervised.y)]
    bar!(plt[3], ["y=0", "y=1"], counts; color = [:gray65, :seagreen3],
         title = "Outcome counts\nmean(y)=$(round(stats.outcome_rate, digits=2))", ylabel = "count")

    histogram!(plt[4], truth.model.lambda[:, 1]; bins = 20, alpha = 0.55, color = :dodgerblue3,
               title = "Lambda by cluster", xlabel = "lambda", ylabel = "count")
    histogram!(plt[4], truth.model.lambda[:, 2]; bins = 20, alpha = 0.55, color = :tomato3)

    histogram!(plt[5], stats.nobs; bins = length(unique(stats.nobs)), color = :mediumpurple3,
               title = "Observations per patient\nmean=$(round(stats.mean_nobs, digits=1)), sd=$(round(stats.sd_nobs, digits=1))",
               xlabel = "n_i", ylabel = "count")

    histogram!(plt[6], regtruth.p; bins = 20, color = :darkorange2,
               title = "Outcome probabilities\nmean logit=$(round(stats.eta_mean, digits=2)), sd=$(round(stats.eta_sd, digits=2))",
               xlabel = "Pr(y_i=1)", ylabel = "count")

    savefig(plt, path)
    path
end

function save_bundle(bundle; stem = "simulation_bundle")
    jls_path = "$(stem).jls"
    serialize(jls_path, bundle)
    jld2_path = nothing
    if HAVE_JLD2
        jld2_path = "$(stem).jld2"
        JLD2.jldopen(jld2_path, "w") do f
            f["bundle"] = bundle
        end
    end
    (; jls_path, jld2_path)
end

function load_bundle(path::AbstractString)
    endswith(path, ".jls") && return deserialize(path)
    if endswith(path, ".jld2")
        HAVE_JLD2 || error("JLD2 is not installed in this environment.")
        return JLD2.jldopen(path, "r") do f
            f["bundle"]
        end
    end
    error("Unrecognized bundle format: $path")
end

function simulate_dataset(rng::AbstractRNG, config::SimulationConfig{T}; stem = "simulation_bundle") where {T}
    latent_data, latent_truth = simulate_latent_truth(rng, config)
    supervised_data, regression_truth = simulate_regression_truth(rng, config, latent_truth)
    plot_path = plot_simulation_summary(latent_data, latent_truth, regression_truth, supervised_data; path = "$(stem)_summary.png")
    stats = patient_stats(latent_data, latent_truth, regression_truth, supervised_data)
    bundle = (;
        config,
        latent_data,
        latent_truth,
        supervised_data,
        regression_model = RegressionModel(config.β_prior),
        regression_truth,
        stats,
        summary_plot = abspath(plot_path),
    )
    paths = save_bundle(bundle; stem)
    (; bundle, plot_path = abspath(plot_path), paths)
end

function main()
    rng = MersenneTwister(20260514)
    config = default_simulation_config(20; npatients = 120, nobs_range = 8:20)
    out = simulate_dataset(rng, config; stem = "simulation_bundle")
    println("saved summary plot: ", out.plot_path)
    println("saved bundle (.jls): ", abspath(out.paths.jls_path))
    out.paths.jld2_path === nothing || println("saved bundle (.jld2): ", abspath(out.paths.jld2_path))
    println("mean(pi) = ", round(out.bundle.stats.mean_pi, digits = 3))
    println("mean(y) = ", round(out.bundle.stats.outcome_rate, digits = 3))
    println("mean(lambda) = ", round.(out.bundle.stats.lambda_mean; digits = 3))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
