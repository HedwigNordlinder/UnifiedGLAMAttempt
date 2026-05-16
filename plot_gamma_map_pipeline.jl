using Random, Statistics, Printf, LinearAlgebra, Plots
include("simulate_data.jl")

gr()

function supervised_from_map(data::LatentData{T}, state::GibbsState{T}, y::AbstractVector{<:Real}) where {T}
    n = length(data.X)
    p = size(data.X[1], 1)
    A = Matrix{T}(undef, n, p)
    B = Matrix{T}(undef, n, p)
    for i in 1:n
        Mi = state.model.mu[i]
        A[i, :] .= Mi[:, 1]
        B[i, :] .= Mi[:, 2]
    end
    SupervisedData(A, B, y)
end

function gamma_recovery(gamma_chain::AbstractMatrix{Bool}, gamma_true::AbstractVector{Bool})
    true0 = .!gamma_true
    true1 = gamma_true
    p00 = any(true0) ? mean(.!gamma_chain[:, true0]) : NaN
    p10 = any(true0) ? mean(gamma_chain[:, true0]) : NaN
    p01 = any(true1) ? mean(.!gamma_chain[:, true1]) : NaN
    p11 = any(true1) ? mean(gamma_chain[:, true1]) : NaN
    [p00 p10; p01 p11]
end

function gamma_recovery_axes!(sp, gamma_chain::AbstractMatrix{Bool}, gamma_true::AbstractVector{Bool})
    G = gamma_recovery(gamma_chain, gamma_true)
    heatmap!(sp, ["sampled=0", "sampled=1"], ["true=0", "true=1"], G;
             c = :viridis, clims = (0, 1), aspect_ratio = :equal,
             title = @sprintf("Gamma recovery\ntrue active=%d/%d", count(gamma_true), length(gamma_true)))
    for i in 1:2, j in 1:2
        annotate!(sp, j, i, text(@sprintf("%.3f", G[i, j]), 12, :white))
    end
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

function pipeline_plot(alloc_logpost, imap, reg_fit, truth; t_true, path = "gamma_map_pipeline.png")
    plt = plot(layout = (3, 2), size = (1200, 980), legend = false)

    trace_axes!(plt[1], 1:length(alloc_logpost), alloc_logpost;
                color = :purple4, ylabel = "alloc logpost",
                title = "Allocation log posterior")
    scatter!(plt[1], [imap], [alloc_logpost[imap]]; color = :goldenrod2, markerstrokecolor = :black, ms = 7)

    gamma_recovery_axes!(plt[2], reg_fit.gamma_chain, truth.gamma)
    t_posterior_axes!(plt[3], t_true, reg_fit.t_chain)
    trace_axes!(plt[4], 1:length(reg_fit.t_trace), reg_fit.t_trace;
                color = :royalblue4, ylabel = "t",
                title = @sprintf("t trace\nmean=%.3f", mean(reg_fit.t_trace)))
    trace_axes!(plt[5], 1:length(reg_fit.logpost), reg_fit.logpost;
                color = :firebrick3, ylabel = "reg logpost",
                title = @sprintf("Regression log posterior\nfinal=%.1f", reg_fit.logpost[end]))

    plot!(plt[6]; framestyle = :none, grid = false, ticks = false, xlims = (0, 1), ylims = (0, 1))
    annotate!(plt[6], 0.02, 0.92, text("Gamma MAP pipeline", 13, :left))
    annotate!(plt[6], 0.02, 0.78, text(@sprintf("allocation MAP iter: %d", imap), 11, :left))
    annotate!(plt[6], 0.02, 0.64, text(@sprintf("true active: %d/%d", count(truth.gamma), length(truth.gamma)), 11, :left))
    annotate!(plt[6], 0.02, 0.50, text(@sprintf("post active mean: %.1f", mean(reg_fit.active_trace)), 11, :left))
    annotate!(plt[6], 0.02, 0.36, text(@sprintf("reg mean accept: %.3f", reg_fit.mean_acceptance), 11, :left))
    annotate!(plt[6], 0.02, 0.22, text(@sprintf("reg divergence: %.3f", reg_fit.divergence_rate), 11, :left))
    annotate!(plt[6], 0.02, 0.08, text(@sprintf("true t: %.3f, post t: %.3f", t_true, mean(reg_fit.t_chain)), 11, :left))

    savefig(plt, path)
    path
end

function main()
    rng = MersenneTwister(20260516)
    p = 60
    T = Float64
    cluster_mean_shift = one(T)
    gamma_prior = T(0.1)
    config = SimulationConfig(
        1000,
        8:14,
        LatentPrior(T(2), T(2), zeros(T, p), T(2) * Matrix{T}(I, p, p), T(2), T(2), T(p + 3), (Matrix{T}(I, p, p), Matrix{T}(I, p, p))),
        Normal(T(0), T(1)),
        Uniform(T(0), T(1)),
    )
    sim = simulate_gamma_dataset(rng, config; stem = "gamma_map_pipeline_bundle",
                                 cluster_mean_shift, gamma_prior, β_prior = Normal(T(0), T(1)))

    alloc_fit = gibbs(rng, sim.bundle.latent_data, config.latent_prior;
                      nsweeps = 250, burn = 0, thin = 1, save_states = true, cluster_mean_shift)
    imap = argmax(alloc_fit.logpost)
    map_state = alloc_fit.states[imap]
    map_supervised = supervised_from_map(sim.bundle.latent_data, map_state, sim.bundle.supervised_data.y)

    reg_fit = gamma_hmc(rng, sim.bundle.regression_model, map_supervised;
                        nsamples = 500, initial_hmc = 300, initial_adapts = 250,
                        hmc_steps_per_gamma = 5, hmc_adapts_per_gamma = 0,
                        reuse_initial_metric = true, step_size_adapt_rate = 0.02,
                        init_gamma = trues(p), init_t = 0.5,
                        target_accept = 0.95, max_depth = 10, save_chain = true)

    plot_path = pipeline_plot(alloc_fit.logpost, imap, reg_fit, sim.bundle.regression_truth;
                              t_true = sim.bundle.regression_truth.t, path = "gamma_map_pipeline.png")

    G = gamma_recovery(reg_fit.gamma_chain, sim.bundle.regression_truth.gamma)
    println("saved pipeline plot: ", abspath(plot_path))
    println("saved bundle (.jls): ", abspath(sim.paths.jls_path))
    println("allocation MAP iteration: ", imap)
    println("posterior mean t: ", round(mean(reg_fit.t_chain), digits = 4))
    println("true t: ", round(sim.bundle.regression_truth.t, digits = 4))
    println("true active betas: ", count(sim.bundle.regression_truth.gamma), "/", p)
    println("posterior mean active betas: ", round(mean(reg_fit.active_trace), digits = 2))
    println("gamma recovery [P(s0|t0) P(s1|t0); P(s0|t1) P(s1|t1)]: ", round.(G; digits = 3))
    println("initial step size: ", round(reg_fit.initial_step_size, digits = 5))
    println("mean chunk step size: ", round(reg_fit.mean_step_size, digits = 5))
    println("mean acceptance: ", round(reg_fit.mean_acceptance, digits = 3))
    println("divergence rate: ", round(reg_fit.divergence_rate, digits = 3))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
