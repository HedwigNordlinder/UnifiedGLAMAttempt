using Random, Printf, Plots
include("simulate_data.jl")

gr()

function diagnostic_plot(t_trace, logpost; burn = 0, path = "regression_chain_diagnostics.png")
    plt = plot(layout = (2, 1), size = (1000, 650), legend = false)
    its = eachindex(t_trace)

    plot!(plt[1], its, t_trace; color = :royalblue4, lw = 1.5,
          xlabel = "iteration", ylabel = "t",
          title = @sprintf("t trace (mean post-burn = %.3f)", mean(t_trace[max(burn + 1, 1):end])))
    burn > 0 && vline!(plt[1], [burn]; color = :black, ls = :dash, lw = 2)

    plot!(plt[2], eachindex(logpost), logpost; color = :firebrick3, lw = 1.5,
          xlabel = "iteration", ylabel = "log posterior",
          title = @sprintf("log posterior trace (final = %.1f)", logpost[end]))
    burn > 0 && vline!(plt[2], [burn]; color = :black, ls = :dash, lw = 2)

    savefig(plt, path)
    path
end

function get_bundle(rng, path::AbstractString, config)
    if isfile(path)
        return load_bundle(path)
    end
    simulate_dataset(rng, config; stem = splitext(path)[1]).bundle
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
    bundle = get_bundle(rng, "regression_recovery_bundle.jls", config)
    burn = 1_000
    fit = hmc(rng, bundle.regression_model, bundle.supervised_data;
              nsweeps = 4_000, n_adapts = burn, burn = burn, thin = 4,
              target_accept = 0.85, max_depth = 10, save_chain = false)
    path = diagnostic_plot(fit.t_trace, fit.logpost; burn = burn, path = "regression_chain_diagnostics.png")
    println("saved diagnostics plot: ", abspath(path))
    println("posterior mean t after burn: ", round(mean(fit.t_trace[burn + 1:end]), digits = 4))
    println("mean acceptance: ", round(fit.mean_acceptance, digits = 3))
    println("divergence rate: ", round(fit.divergence_rate, digits = 3))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
