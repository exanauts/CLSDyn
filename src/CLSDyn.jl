module CLSDyn
using ForwardDiff

import PowerModels

const _WS = 1.0

struct PSystem
    # number of buses
    nbus::Int
    # number of generators
    ngen::Int
    # number of loads
    nload::Int
    # voltage magnitude vector
    vmag::Vector{Float64}
    # reduced admittance matrix
    yred::Matrix{ComplexF64}
    # mechanical power (ngen x 1
    pmec::Vector{Float64}
    # generator inertia (ngen x 1)
    gen_inertia::Vector{Float64}
    # generator damping (ngen x 1)
    gen_damping::Vector{Float64}
end

include("ivp.jl")
include("psys.jl")

end # module CLSDyn
