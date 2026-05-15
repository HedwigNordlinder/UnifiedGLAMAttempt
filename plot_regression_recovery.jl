using Random, LinearAlgebra, Statistics, Printf, Plots
include("simulate_data.jl")

gr()

function recovery_plot(β_true, β_hat, t_true, t_chain; path = "regression_recovery.png", accept_β = NaN, accept_t = NaN)
    plt = plot(layout = (1, 2), size = (1100, 420), legend = false)

    xr = extrema(vcat(β_true, β_hat))
    pad = 0.05 * max(abs(xr[1]), abs(xr[2]), 1.0)
    lo, hi = xr[1] - pad, xr[2] + pad
    X = hcat(ones(length(β_true)), β_true)
    α, b = X \ β_hat
    corrβ = cor(β_true, β_hat)
    rmseβ = sqrt(mean((β_hat .- β_true).^2))
    scatter!(plt[1], β_true, β_hat; color = :midnightblue, alpha = 0.8, ms = 5,
             xlabel = "true beta", ylabel = "posterior mean beta",
             title = @sprintf("Beta recovery\ncor=%.2f, rmse=%.2f", corrβ, rmseβ))
    plot!(plt[1], [lo, hi], [lo, hi]; color = :gray45, ls = :dash, lw = 2)
    plot!(plt[1], [lo, hi], α .+ b .* [lo, hi]; color = :tomato3, lw = 2.5)

    t_mean = mean(t_chain)
    t_sd = std(t_chain)
    histogram!(plt[2], t_chain; bins = 30, color = :darkorange2, alpha = 0.8,
               xlabel = "t", ylabel = "count",
               title = @sprintf("t posterior\ntrue=%.3f, post=%.3f, sd=%.3f", t_true, t_mean, t_sd))
    vline!(plt[2], [t_true]; color = :black, lw = 2.5, ls = :dash)
    vline!(plt[2], [t_mean]; color = :seagreen4, lw = 2.5)
    plot!(plt[2], titlefont = 11)

    savefig(plt, path)
    path
end

function main()
    rng = MersenneTwister(20260515)
    p = 20
    T = Float64
    config = SimulationConfig(
        300,
        8:20,
        LatentPrior(T(2), T(2), zeros(T, p), T(2) * Matrix{T}(I, p, p), T(2), T(2), T(p + 3), (Matrix{T}(I, p, p), Matrix{T}(I, p, p))),
        Normal(T(0), T(0.5)),
        Uniform(T(0), T(1)),
    )
    sim = simulate_dataset(rng, config; stem = "regression_recovery_bundle")
    fit = mala(rng, sim.bundle.regression_model, sim.bundle.supervised_data;
               nsweeps = 4_000, burn = 1_000, thin = 4,
               step_β = 0.20, step_t = 0.80, save_chain = true)
    β_hat = vec(mean(fit.β_chain, dims = 1))
    plot_path = recovery_plot(sim.bundle.regression_truth.β, β_hat, sim.bundle.regression_truth.t, fit.t_chain;
                              path = "regression_recovery.png", accept_β = fit.accept_β, accept_t = fit.accept_t)
    println("saved recovery plot: ", abspath(plot_path))
    println("saved bundle (.jls): ", abspath(sim.paths.jls_path))
    println("posterior mean t: ", round(mean(fit.t_chain), digits = 4))
    println("true t: ", round(sim.bundle.regression_truth.t, digits = 4))
    println("acceptance: beta=", round(fit.accept_β, digits = 3), ", t=", round(fit.accept_t, digits = 3))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
