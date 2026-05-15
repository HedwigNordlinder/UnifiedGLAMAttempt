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

function pipeline_plot(alloc_logpost, imap, reg_fit, β_true; t_true, burn = 0, path = "map_pipeline.png")
    plt = plot(layout = (3, 2), size = (1200, 980), legend = false)

    trace_axes!(plt[1], 1:length(alloc_logpost), alloc_logpost;
                color = :purple4, ylabel = "alloc logpost",
                title = "Allocation log posterior")
    scatter!(plt[1], [imap], [alloc_logpost[imap]]; color = :goldenrod2, markerstrokecolor = :black, ms = 7)

    recovery_axes!(plt[2], β_true, vec(mean(reg_fit.β_chain, dims = 1)))
    t_posterior_axes!(plt[3], t_true, reg_fit.t_chain)
    trace_axes!(plt[4], 1:length(reg_fit.t_trace), reg_fit.t_trace;
                burn = burn, color = :royalblue4, ylabel = "t",
                title = @sprintf("t trace\npost-burn mean=%.3f", mean(reg_fit.t_trace[burn + 1:end])))
    trace_axes!(plt[5], 1:length(reg_fit.logpost), reg_fit.logpost;
                burn = burn, color = :firebrick3, ylabel = "reg logpost",
                title = @sprintf("Regression log posterior\nfinal=%.1f", reg_fit.logpost[end]))

    plot!(plt[6]; framestyle = :none, grid = false, ticks = false, xlims = (0, 1), ylims = (0, 1))
    annotate!(plt[6], 0.02, 0.92, text("MAP pipeline summary", 13, :left))
    annotate!(plt[6], 0.02, 0.76, text(@sprintf("allocation MAP iter: %d", imap), 11, :left))
    annotate!(plt[6], 0.02, 0.60, text(@sprintf("reg acceptance beta: %.3f", reg_fit.accept_β), 11, :left))
    annotate!(plt[6], 0.02, 0.46, text(@sprintf("reg acceptance t: %.3f", reg_fit.accept_t), 11, :left))
    annotate!(plt[6], 0.02, 0.30, text(@sprintf("true t: %.3f", t_true), 11, :left))
    annotate!(plt[6], 0.02, 0.16, text(@sprintf("post mean t: %.3f", mean(reg_fit.t_chain)), 11, :left))

    savefig(plt, path)
    path
end

function main()
    rng = MersenneTwister(20260515)
    p = 100
    T = Float64
    config = SimulationConfig(
        100,
        8:14,
        LatentPrior(T(2), T(2), zeros(T, p), T(2) * Matrix{T}(I, p, p), T(2), T(2), T(p + 3), (Matrix{T}(I, p, p), Matrix{T}(I, p, p))),
        Normal(T(0), T(0.5)),
        Uniform(T(0), T(1)),
    )
    sim = simulate_dataset(rng, config; stem = "map_pipeline_bundle")

    alloc_fit = gibbs(rng, sim.bundle.latent_data, config.latent_prior;
                      nsweeps = 250, burn = 0, thin = 1, save_states = true)
    imap = argmax(alloc_fit.logpost)
    map_state = alloc_fit.states[imap]
    map_supervised = supervised_from_map(sim.bundle.latent_data, map_state, sim.bundle.supervised_data.y)

    reg_fit = mala(rng, sim.bundle.regression_model, map_supervised;
                   nsweeps = 3_000, burn = 750, thin = 3,
                   step_β = 0.32, step_t = 1.20, save_chain = true)

    plot_path = pipeline_plot(alloc_fit.logpost, imap, reg_fit, sim.bundle.regression_truth.β;
                              t_true = sim.bundle.regression_truth.t, burn = 750, path = "map_pipeline.png")

    println("saved pipeline plot: ", abspath(plot_path))
    println("saved bundle (.jls): ", abspath(sim.paths.jls_path))
    println("allocation MAP iteration: ", imap)
    println("posterior mean t: ", round(mean(reg_fit.t_chain), digits = 4))
    println("true t: ", round(sim.bundle.regression_truth.t, digits = 4))
    println("acceptance: beta=", round(reg_fit.accept_β, digits = 3), ", t=", round(reg_fit.accept_t, digits = 3))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
