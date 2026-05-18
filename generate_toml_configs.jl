const P_VALUES = [50,100,200,400]
const NPATIENTS_VALUES = [500,1000,2000]
const NOBS_RANGE_VALUES = [(1, 40),(20,40)]
const CLUSTER_MEAN_SHIFT_VALUES = [0.2,0.5,1.0]
const ACTIVE_PROBABILITY_VALUES = [0.1,0.2,0.3]
const CLUSTER_PRIOR_VALUES = [(2.0, 2.0),(4.0,2.0)]

const CONFIG_OUTPUT_DIR = "generated_configs"
const BASE_SEED = 20260516

tag(x::Integer) = string(x)
tag(x::AbstractFloat) = replace(string(x), "-" => "m", "." => "p")
tag(x::Tuple{<:Integer,<:Integer}) = "$(x[1])-$(x[2])"

function config_stem(p, npatients, nobs_range, cluster_mean_shift, active_probability, cluster_alpha0, cluster_beta0)
    join([
        "gamma",
        "p$(tag(p))",
        "npatients$(tag(npatients))",
        "nobs$(tag(nobs_range))",
        "shift$(tag(cluster_mean_shift))",
        "active$(tag(active_probability))",
        "alpha$(tag(cluster_alpha0))",
        "beta$(tag(cluster_beta0))",
    ], "_")
end

function config_text(; p, npatients, nobs_range, cluster_mean_shift, active_probability,
                     cluster_alpha0, cluster_beta0, stem)
    """
    [run]
    seed = $BASE_SEED
    prefix = "$stem"
    output_dir = "runs/$stem"
    progress = true

    [simulation]
    regression = "gamma"
    p = $p
    npatients = $npatients
    nobs_range = [$(nobs_range[1]),$(nobs_range[2])]
    cluster_mean_shift = $cluster_mean_shift

    [latent_prior]
    alpha0 = $cluster_alpha0
    beta0 = $cluster_beta0
    lambda0_scale = 2.0
    a0 = 2.0
    b0 = 2.0
    nu0_offset = 3.0
    S0_scale = 1.0
    m0 = 0.0

    [beta_prior]
    dist = "normal"
    mean = 0.0
    sd = 1.0

    [t_prior]
    dist = "uniform"
    low = 0.0
    high = 1.0

    [gamma]
    active_probability = $active_probability

    [allocation_mcmc]
    nsweeps = 500
    burn = 0
    thin = 1
    save_states = true
    postprocess = true
    progress = true
    progress_every = 10

    [plotting]
    burn_in = 0
    allocation_burn_in = 50
    regression_burn_in = 400

    [fitting]
    regression_data = "map"

    [regression_mcmc]
    model = "auto"
    nsamples = 800
    initial_hmc = 400
    initial_adapts = 200
    hmc_steps_per_gamma = 5
    hmc_adapts_per_gamma = 0
    reuse_initial_metric = true
    step_size_adapt_rate = 0.02
    init_gamma = "all_on"
    init_t = 0.5
    target_accept = 0.9
    max_depth = 8
    save_chain = true
    save_states = false
    progress = true
    progress_every = 10
    """
end

function generate_configs()
    mkpath(CONFIG_OUTPUT_DIR)
    foreach(rm, filter(endswith(".toml"), readdir(CONFIG_OUTPUT_DIR; join = true)))
    paths = String[]
    seen = Set{String}()

    for p in P_VALUES,
        npatients in NPATIENTS_VALUES,
        nobs_range in NOBS_RANGE_VALUES,
        cluster_mean_shift in CLUSTER_MEAN_SHIFT_VALUES,
        active_probability in ACTIVE_PROBABILITY_VALUES,
        cluster_prior in CLUSTER_PRIOR_VALUES

        cluster_alpha0, cluster_beta0 = cluster_prior
        stem = config_stem(p, npatients, nobs_range, cluster_mean_shift, active_probability,
                           cluster_alpha0, cluster_beta0)
        stem in seen && error("Duplicate generated config stem: $stem")
        push!(seen, stem)

        path = joinpath(CONFIG_OUTPUT_DIR, "$stem.toml")
        text = config_text(;
            p,
            npatients,
            nobs_range,
            cluster_mean_shift,
            active_probability,
            cluster_alpha0,
            cluster_beta0,
            stem,
        )
        write(path, text)
        push!(paths, path)
    end

    println("generated $(length(paths)) config file(s) in $(abspath(CONFIG_OUTPUT_DIR))")
    foreach(path -> println("  ", path), paths)
    paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_configs()
end
