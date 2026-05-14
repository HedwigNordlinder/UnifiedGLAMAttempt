using Distributions

struct LatentData
    patient_features::Vector{Vector{Vector{Float64}}}
    patient_outcomes::Vector{Real}
    patient_dimensionality::Vector{Integer} # How many observations per patient
end

struct GMMPriors
    allocation_prior::Distribution # Beta distribution
    μ_hyperprior::Distribution
    λ_hyperprior::Distribution # Hyperprior for concentration
    cluster_covariance_prior::Vector{Distribution} # Hyperprior for clusters
end

function ℓ_prior(priors::GMMPriors, allocation_probabilities::Vector{Float64}, 
                    cluster_μ::Vector{Vector{Float64}}, 
                    cluster_λ::Vector{Vector{Float64}}, 
                    global_cluster_Σ::Vector{Matrix{Float64}})
    allocation_ℓprior = sum(logpdf.((priors.allocation_prior,),allocation_probabilities))
    μ_ℓprior = sum(logpdf.((priors.μ_hyperprior,),reduce(vcat,cluster_μ)))
    λ_ℓprior = sum(logpdf.((priors.λ_hyperprior,), reduce(vcat, cluster_λ)))
    Σ_global_ℓprior = sum(logpdf.((priors.cluster_covariance_prior,),global_cluster_Σ))
    return allocation_ℓprior+μ_ℓprior+λ_ℓprior+Σ_global_ℓprior
end

function ℓ_likelihood(data::LatentData,allocation_probabilities::Vector{Float64},
                        cluster_μ::Vector{Vector{Float64}},
                        cluster_λ::Vector{Vector{Float64}},
                        global_cluster_Σ::Vector{Matrix{Float64}}, 
                        sampled_allocations::Vector{Vector{Integer}},
                        sampled_allocation_probabilities::Vector{Float64},
                        cluster_patient_μ::Vector{Vector{Vector{Float64}}})
    patient_cluster_covariances = reduce(hcat, cluster_λ)' .* reshape(global_cluster_Σ, 1, :)

    allocation_counts = sum.(sampled_allocations)
    allocation_probability_ℓ_likelihood = sum(logpdf.(Binomial.(data.patient_dimensionality,sampled_allocation_probabilities),allocation_counts))
    cluster_masks = [Bool.(m) for m in sampled_allocations]
    cluster_a = [@view data.patient_features[i][cluster_masks[i]] for i in eachindex(data.patient_features)]
    cluster_b = [@view data.patient_features[i][.!cluster_masks[i]] for i in eachindex(data.patient_features)]

    cluster_a_covariances = getindex.(patient_cluster_covariances,1)
    cluster_b_covariances = getindex.(patient_cluster_covariances, 2)
    cluster_a_μ = getindex.(cluster_patient_μ, 1)
    cluster_b_μ = getindex.(cluster_patient_μ,2)
    cluster_a_ℓ_likelihood = sum(logpdf.(MvNormal.(cluster_a_μ, cluster_a_covariances),cluster_a))
    cluster_b_ℓ_likelihood = sum(logpdf.(MvNormal.(cluster_b_μ, cluster_b_covariances),cluster_b))

end