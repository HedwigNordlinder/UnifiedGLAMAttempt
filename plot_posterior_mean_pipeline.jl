using Random, Statistics, Printf, LinearAlgebra, Plots
include("simulate_data.jl")

gr()

function supervised_from_posterior_mean(states::Vector{GibbsState{T}}, y::AbstractVector{<:Real}) where {T}
    isempty(states) && error("Need at least one saved latent state.")
    n = length(states[1].model.mu)
    p = size(states[1].model.mu[1], 1)
    A = zeros(T, n, p)
    B = zeros(T, n, p)
    for state in states, i in 1:n
        A[i, :] .+= state.model.mu[i][:, 1]
        B[i, :] .+= state.model.mu[i][:, 2]
    end
    scale = inv(T(length(states)))
    A .*= scale
    B .*= scale
    SupervisedData(A, B, y)
end

function recovery_axes!(sp, β_true, β_hat)
    xr = extrema(vcat(β_true, β_hat))
    pad = 0.05 * max(abs(xr[1]), abs(xr[2]), 1.0)
    lo, hi = xr[1] - pad, xr[2] + pad
    X = hcat(ones(length(β_true)), β_true)
    α, b = X \ β_hat
    scatter!(sp, β_true, β_hat; color = :midnightblue, alpha = 0.8, ms = 5,
             xlabel = "true beta", ylabel = "posterior mean beta",
             title = @sprintf("Beta recovery\ncor=%.2f, rmse=%.2f", cor(β_true, β_hat), sqrt(mean((β_hat .- β_true).^2))))
    plot!(sp, [lo, hi], [lo, hi]; color = :gray45, ls = :dash, lw = 2)
    plot!(sp, [lo, hi], α .+ b .* [lo, hi]; color = :tomato3, lw = 2.5)
end

function t_posterior_axes!(sp, t_true, t_chain)
    t_mean, t_sd = mean(t_chain), std(t_chain)
    histogram!(sp, t_chain; bins = 30, color = :darkorange2, alpha = 0.8,
               xlabel = "t", ylabel = "count",
               title = @sprintf("t posterior\ntrue=%.3f, post=%.3f, sd=%.3f", t_true, t_mean, t_sd))
    vline!(sp, [t_true]; color = :black, lw = 2.5, ls = :dash)
    vline!(sp, [t_mean]; color = :seagreen4, lw = 2.5)
end

function trace_axes!(sp, x, y; burn = nothing, color = :royalblue4, xlabel = "iteration", ylabel = "", title = "")
    plot!(sp, x, y; color, lw = 1.3, xlabel, ylabel, title)
    burn === nothing || vline!(sp, [burn]; color = :black, ls = :dash, lw = 2)
end

function pipeline_plot(alloc_logpost, reg_fit, β_true; latent_burn, latent_ndraws, t_true, burn = 0, path = "posterior_mean_pipeline.png")
    plt = plot(layout = (3, 2), size = (1200, 980), legend = false)

    trace_axes!(plt[1], 1:length(alloc_logpost), alloc_logpost;
                burn = latent_burn, color = :purple4, ylabel = "alloc logpost",
                title = "Allocation log posterior")

    recovery_axes!(plt[2], β_true, vec(mean(reg_fit.β_chain, dims = 1)))
    t_posterior_axes!(plt[3], t_true, reg_fit.t_chain)
    trace_axes!(plt[4], 1:length(reg_fit.t_trace), reg_fit.t_trace;
                burn = burn, color = :royalblue4, ylabel = "t",
                title = @sprintf("t trace\npost-burn mean=%.3f", mean(reg_fit.t_trace[burn + 1:end])))
    trace_axes!(plt[5], 1:length(reg_fit.logpost), reg_fit.logpost;
                burn = burn, color = :firebrick3, ylabel = "reg logpost",
                title = @sprintf("Regression log posterior\nfinal=%.1f", reg_fit.logpost[end]))

    plot!(plt[6]; framestyle = :none, grid = false, ticks = false, xlims = (0, 1), ylims = (0, 1))
    annotate!(plt[6], 0.02, 0.92, text("Posterior-Mean Pipeline", 13, :left))
    annotate!(plt[6], 0.02, 0.76, text(@sprintf("latent mean draws: %d", latent_ndraws), 11, :left))
    annotate!(plt[6], 0.02, 0.62, text(@sprintf("reg mean accept prob: %.3f", reg_fit.mean_acceptance), 11, :left))
    annotate!(plt[6], 0.02, 0.48, text(@sprintf("reg divergence: %.3f", reg_fit.divergence_rate), 11, :left))
    annotate!(plt[6], 0.02, 0.32, text(@sprintf("true t: %.3f", t_true), 11, :left))
    annotate!(plt[6], 0.02, 0.18, text(@sprintf("post mean t: %.3f", mean(reg_fit.t_chain)), 11, :left))

    savefig(plt, path)
    path
end

function main()
    rng = MersenneTwister(20260515)
    p = 100
    T = Float64
    latent_nsweeps = 300
    latent_burn = 150
    cluster_mean_shift = one(T)
    config = SimulationConfig(
        100,
        8:14,
        LatentPrior(T(2), T(2), zeros(T, p), T(2) * Matrix{T}(I, p, p), T(2), T(2), T(p + 3), (Matrix{T}(I, p, p), Matrix{T}(I, p, p))),
        Normal(T(0), T(0.5)),
        Uniform(T(0), T(1)),
    )
    sim = simulate_dataset(rng, config; stem = "posterior_mean_pipeline_bundle", cluster_mean_shift)

    alloc_fit = gibbs(rng, sim.bundle.latent_data, config.latent_prior;
                      nsweeps = latent_nsweeps, burn = 0, thin = 1, save_states = true, cluster_mean_shift)
    latent_states = alloc_fit.states[latent_burn + 1:end]
    mean_supervised = supervised_from_posterior_mean(latent_states, sim.bundle.supervised_data.y)

    reg_fit = hmc(rng, sim.bundle.regression_model, mean_supervised;
                  nsweeps = 3_000, n_adapts = 750, burn = 750, thin = 3,
                  target_accept = 0.9, max_depth = 10, save_chain = true)

    plot_path = pipeline_plot(alloc_fit.logpost, reg_fit, sim.bundle.regression_truth.β;
                              latent_burn = latent_burn, latent_ndraws = length(latent_states),
                              t_true = sim.bundle.regression_truth.t, burn = 750,
                              path = "posterior_mean_pipeline.png")

    println("saved pipeline plot: ", abspath(plot_path))
    println("saved bundle (.jls): ", abspath(sim.paths.jls_path))
    println("latent posterior-mean draws: ", length(latent_states))
    println("posterior mean t: ", round(mean(reg_fit.t_chain), digits = 4))
    println("true t: ", round(sim.bundle.regression_truth.t, digits = 4))
    println("mean HMC accept prob: ", round(reg_fit.mean_acceptance, digits = 3))
    println("divergence rate: ", round(reg_fit.divergence_rate, digits = 3))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
