using LinearAlgebra, Printf, Random, Statistics
using LogDensityProblems, LogDensityProblemsAD

include("simulate_data.jl")

const SEED = 20260518
const P = 5
const NPATIENTS = 30
const NOBS_RANGE = 2:5
const CLUSTER_MEAN_SHIFT = 0.7
const GRAD_REL_TOL = 1e-4
const GRAD_ABS_TOL = 1e-5
const VALUE_ABS_TOL = 1e-8
const MAX_DIVERGENCE_RATE = 0.25
const MIN_TRACE_RANGE = 1e-10

function check(condition::Bool, msg::AbstractString)
    if condition
        println("[pass] ", msg)
    else
        error("[fail] $msg")
    end
end

function finite_difference_gradient(f, θ::AbstractVector; relstep = 1e-5)
    g = similar(float.(θ))
    θplus = collect(float.(θ))
    θminus = collect(float.(θ))
    for j in eachindex(θ)
        h = relstep * max(1.0, abs(float(θ[j])))
        θplus[j] += h
        θminus[j] -= h
        g[j] = (f(θplus) - f(θminus)) / (2h)
        θplus[j] = θ[j]
        θminus[j] = θ[j]
    end
    g
end

function gradient_check(label::AbstractString, target, θ::AbstractVector)
    wrapped = LogDensityProblemsAD.ADgradient(REGRESSION_AD_BACKEND, target)
    value, mooncake_grad = LogDensityProblemsAD.logdensity_and_gradient(wrapped, θ)
    direct_value = LogDensityProblems.logdensity(target, θ)
    fd_grad = finite_difference_gradient(x -> LogDensityProblems.logdensity(target, x), θ)

    value_err = abs(value - direct_value)
    grad_diff = mooncake_grad .- fd_grad
    max_abs_grad_err = maximum(abs.(grad_diff))
    rel_grad_err = norm(grad_diff) / max(norm(fd_grad), 1.0)
    inf_tol = GRAD_ABS_TOL + GRAD_REL_TOL * max(maximum(abs.(fd_grad)), 1.0)

    @printf("%s gradient check:\n", label)
    @printf("  value_err=%.3e max_abs_grad_err=%.3e rel_grad_err=%.3e inf_tol=%.3e\n",
            value_err, max_abs_grad_err, rel_grad_err, inf_tol)

    check(isfinite(value), "$label Mooncake log density is finite")
    check(value_err <= VALUE_ABS_TOL, "$label Mooncake value matches direct log density")
    check(all(isfinite, mooncake_grad), "$label Mooncake gradient is finite")
    check(max_abs_grad_err <= inf_tol, "$label Mooncake gradient matches finite differences")
end

finite_vector(x) = all(isfinite, x)
bounded_prob_vector(x) = all(v -> isfinite(v) && 0.0 <= v <= 1.0, x)
trace_range(x) = maximum(x) - minimum(x)

function hmc_smoke_check(label::AbstractString, fit)
    @printf("%s HMC smoke:\n", label)
    @printf("  mean_accept=%.3g divergence_rate=%.3g t_range=%.3e logpost_range=%.3e\n",
            fit.mean_acceptance, fit.divergence_rate, trace_range(fit.t_trace), trace_range(fit.logpost))

    check(finite_vector(fit.logpost), "$label log posterior trace is finite")
    check(finite_vector(fit.t_trace), "$label t trace is finite")
    check(bounded_prob_vector(fit.accept_trace), "$label accept_prob trace is in [0, 1]")
    check(0.0 <= fit.divergence_rate <= MAX_DIVERGENCE_RATE, "$label divergence rate is not systematic")
    check(trace_range(fit.t_trace) > MIN_TRACE_RANGE, "$label t trace moves")
end

function build_validation_data(rng::AbstractRNG)
    config = default_simulation_config(P; npatients = NPATIENTS, nobs_range = NOBS_RANGE)
    _, latent_truth = simulate_latent_truth(rng, config; cluster_mean_shift = CLUSTER_MEAN_SHIFT)
    dense_data, _ = simulate_regression_truth(rng, config, latent_truth)
    gamma_data, _ = simulate_gamma_regression_truth(rng, config, latent_truth; gamma_prior = 0.4)
    dense_model = RegressionModel(config.β_prior)
    gamma_model = GammaRegressionModel(Normal(0.0, 1.0), 0.4)
    (; dense_data, gamma_data, dense_model, gamma_model)
end

function main()
    rng = MersenneTwister(SEED)
    data = build_validation_data(rng)

    println("Mooncake AD validity check")
    println("backend: ", REGRESSION_AD_BACKEND)

    θ_dense = vcat(0.2 .* randn(rng, P), logit(0.45))
    dense_target = RegressionTarget(data.dense_model, data.dense_data)
    gradient_check("dense regression", dense_target, θ_dense)

    active = [1, 3, 5]
    θ_gamma = vcat([0.15, -0.25, 0.35], logit(0.55))
    gamma_target = GammaActiveTarget(
        data.gamma_model.β_prior,
        Matrix(data.gamma_data.cluster_a_μ[:, active]),
        Matrix(data.gamma_data.cluster_b_μ[:, active]),
        data.gamma_data.y,
    )
    gradient_check("gamma active-subset regression", gamma_target, θ_gamma)

    dense_fit = hmc(
        rng,
        data.dense_model,
        data.dense_data;
        nsweeps = 30,
        n_adapts = 15,
        burn = 0,
        thin = 1,
        init_β = zeros(P),
        init_t = 0.45,
        target_accept = 0.9,
        max_depth = 6,
        save_chain = true,
        progress = false,
    )
    hmc_smoke_check("dense regression", dense_fit)

    gamma_fit = gamma_hmc(
        rng,
        data.gamma_model,
        data.gamma_data;
        nsamples = 30,
        initial_hmc = 20,
        initial_adapts = 10,
        hmc_steps_per_gamma = 3,
        hmc_adapts_per_gamma = 0,
        init_β = zeros(P),
        init_gamma = trues(P),
        init_t = 0.45,
        target_accept = 0.9,
        max_depth = 6,
        save_chain = true,
        progress = false,
    )
    hmc_smoke_check("gamma regression", gamma_fit)
    @printf("  active_range=%d gamma_changes=%d\n",
            maximum(gamma_fit.active_trace) - minimum(gamma_fit.active_trace),
            count(!iszero, vec(sum(abs.(diff(Int.(gamma_fit.gamma_chain); dims = 1)); dims = 2))))
    check(maximum(gamma_fit.active_trace) - minimum(gamma_fit.active_trace) > 0,
          "gamma active count moves")

    println("VALIDATION PASSED")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
