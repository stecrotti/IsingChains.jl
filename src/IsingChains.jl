module IsingChains

import Base: zero, convert, show
import LogExpFunctions: logaddexp
import OffsetArrays: OffsetVector, fill
import UnPack: @unpack
import LinearAlgebra: dot
import StatsBase: sample
import Random: AbstractRNG, GLOBAL_RNG

export IsingChain, accumulate_all!, nspins, normalization, energy, free_energy, pdf,
        site_marginals!, site_marginals, site_magnetizations!, site_magnetizations,
        neighbor_marginals!, neighbor_marginals, 
        neighbor_magnetizations!, neighbor_magnetizations,
        avg_energy, entropy,
        sample!, sample

include("accumulate.jl")

struct IsingChain{T,U}
    J :: Vector{T}      # couplings
    h :: Vector{T}      # external fields
    β :: T              # inverse temperature
    l :: OffsetVector{U, Vector{U}}     # messages from the left
    r :: OffsetVector{U, Vector{U}}     # messages from the right   

    function IsingChain(J::Vector{T}, h::Vector{T}, β::T) where T
        l = accumulate_left(J, h, β)
        r = accumulate_right(J, h, β)
        U = eltype(l)
        new{T,U}(J, h, β, l, r)
    end

end

IsingChain(N::Int; T = Float64) = IsingChain(randn(T,N-1), randn(T,N), 1.0)

function show(io::IO, x::IsingChain)
    @unpack J, h, β, l, r = x
    println(io, "IsingChain with N = $(nspins(x)) variables at temperature β = $β")
end

nspins(x::IsingChain) = length(x.h)

accumulate_all!(x::IsingChain) = accumulate_all!(x.l, x.r, x.J, x.h, x.β)

function free_energy(x::IsingChain)
    @unpack J, h, β, l, r = x
    return -1/β * logaddexp( β*(h[1] + r[2][:p]), β*(-h[1] + r[2][:m]) )
end

normalization(x::IsingChain) = exp(-x.β * free_energy(x))


function energy(x::IsingChain, σ)
    @unpack J, h, β, l, r = x
    e_fields = - dot(h, σ)
    e_pairs = 0.0
    for i in eachindex(J)
        e_pairs -= J[i]*σ[i]*σ[i+1]
    end
    return e_fields + e_pairs
end

pdf(x::IsingChain, σ) = exp(-x.β*(energy(x, σ) - free_energy(x)))

function site_marginals!(m::Vector{Bin{T}}, x::IsingChain) where T
    @unpack J, h, β, l, r = x
    F = free_energy(x)
    for i in eachindex(m)
        m[i] = ( exp( β*(l[i-1][:p] + h[i] + r[i+1][:p] + F )),
                 exp( β*(l[i-1][:m] - h[i] + r[i+1][:m] + F )) )
    end
    m
end
function site_marginals(x::IsingChain{T,U}) where {T,U}
    m = zero(x.l.parent)
    site_marginals!(m, x)
end

function site_magnetizations!(m::Vector{T}, x::IsingChain) where {T<:Real}
    @unpack J, h, β, l, r = x
    F = free_energy(x)
    for i in eachindex(m)
        p_i = Bin{T}(( exp( β*(l[i-1][:p] + h[i] + r[i+1][:p] + F )),
                       exp( β*(l[i-1][:m] - h[i] + r[i+1][:m] + F )) ))
        m[i] = magnetization(p_i)
    end
    m
end
function site_magnetizations(x::IsingChain{T,U}) where {T,U}
    m = zeros(T, nspins(x))
    site_magnetizations!(m, x)
end

function neighbor_marginals!(p::Vector{Bin2{T}}, x::IsingChain) where T
    @unpack J, h, β, l, r = x
    F = free_energy(x)
    for i in eachindex(p)
        p[i] = ( exp( β*(l[i-1][:p] + h[i] + J[i] + h[i+1] + r[i+2][:p] + F )),     # pp 
                 exp( β*(l[i-1][:p] + h[i] - J[i] - h[i+1] + r[i+2][:m] + F )),     # pm
                 exp( β*(l[i-1][:m] - h[i] - J[i] + h[i+1] + r[i+2][:p] + F )),     # mp 
                 exp( β*(l[i-1][:m] - h[i] + J[i] - h[i+1] + r[i+2][:m] + F )) )    # mm
    end
    p
end
function neighbor_marginals(x::IsingChain{T,U}) where {T,U}
    p = zeros(Bin2{T}, length(x.J))
    neighbor_marginals!(p, x)
end

function neighbor_magnetizations!(p::Vector{T}, x::IsingChain) where {T<:Real}
    @unpack J, h, β, l, r = x
    F = free_energy(x)
    for i in eachindex(p)
        p_i = Bin2{T}(( exp( β*(l[i-1][:p] + h[i] + J[i] + h[i+1] + r[i+2][:p] + F )),     # pp 
                        exp( β*(l[i-1][:p] + h[i] - J[i] - h[i+1] + r[i+2][:m] + F )),     # pm
                        exp( β*(l[i-1][:m] - h[i] - J[i] + h[i+1] + r[i+2][:p] + F )),     # mp 
                        exp( β*(l[i-1][:m] - h[i] + J[i] - h[i+1] + r[i+2][:m] + F )) ))
        p[i] = magnetization(p_i)
    end
    p
end
function neighbor_magnetizations(x::IsingChain{T,U}) where {T,U}
    p = zeros(T, nspins(x)-1)
    neighbor_magnetizations!(p, x)
end

function avg_energy(x::IsingChain; m = site_magnetizations(x),
        p = neighbor_magnetizations(x))
    @unpack J, h, β, l, r = x
    e_fields = - dot(h, m)
    e_pairs = - dot(J, p)
    return e_fields + e_pairs
end

entropy(x::IsingChain; kw...) = x.β * (avg_energy(x; kw...) - free_energy(x)) 

function sample_spin(rng::AbstractRNG, pplus::Real)
    @assert 0 ≤ pplus ≤ 1 "$pplus"
    r = rand(rng)
    r < pplus ? 1 : -1
end

# return a sample along with its log-probability
function sample!(rng::AbstractRNG, σ, x::IsingChain)
    @unpack J, h, β, l, r = x
    F = free_energy(x)
    # sample σ₁ from its marginal 
    pplus = exp(β*( + h[1] + r[2][:p] + F))
    σ1 = sample_spin(rng, pplus)
    σ[1] = σ1
    # `u` accumulates the energy contribution of already extracted spins
    u = h[1]*σ1

    for i in 2:lastindex(σ)
        # sample σᵢ|σ₁,…,σᵢ₋₁
        logpplus = β*( u + h[i] + J[i-1]*σ[i-1] + r[i+1][:p] )
        logpminus = β*( u - h[i] - J[i-1]*σ[i-1] + r[i+1][:m] )
        pplus = exp(logpplus - logaddexp(logpplus, logpminus))
        σi = sample_spin(rng, pplus)
        σ[i] = σi
        u += h[i]*σi + J[i-1]*σ[i-1]*σi
    end

    logp = β*(u + F)
    @assert exp(logp) ≈ pdf(x, σ)
    σ, logp
end
sample!(σ, x::IsingChain) = sample!(GLOBAL_RNG, σ, x)
sample(rng::AbstractRNG, x::IsingChain) = sample!(rng, zeros(Int, nspins(x)), x)
sample(x::IsingChain) = sample(GLOBAL_RNG, x)

end # end module