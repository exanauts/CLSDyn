module CLSDyn
using ForwardDiff

import PowerModels

const _WS = 1.0

checkarguments(f, argtypes) = length(methods(f, argtypes)) > 0

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

mutable struct SystemDynamics
    psys::PSystem
    pvec::AbstractArray
    rhs!::Function

    function SystemDynamics(psys::PSystem, input_rhs::Function)
        # in this function we assemble the parameters into a vector
        # and store it in the SystemDynamics struct. This is so that
        # the interfaces to gradients and hessians become clean.
        pvec = vcat(psys.vmag, psys.pmec, psys.gen_inertia, psys.gen_damping)

        # the input rhs function should be of the form
        #   rhs!(dx, x, p, t, psys)
        # and here we create a wrapper.
        # we might need to enforce type signature with checkarguments.
        function rhs!(dx, x, p, t)
            input_rhs(dx, x, p, t, psys)
        end
        new(psys, pvec, rhs!)
    end

end

# IVP struct
abstract type IVPAbstract end
struct IVP <: IVPAbstract
    # number of steps
    nsteps::Int
    # step size
    dt::Float64
    # initial condition
    x0::AbstractArray
    # time span
    tspan::Tuple{Float64,Float64}
    # method
    method::ODEMethod
    # system dynamics
    sys::SystemDynamics

    function IVP(
        nsteps::Int,
        dt::Float64,
        x0::AbstractArray,
        tspan::Tuple{Float64,Float64},
        method::ODEMethod,
        sys::SystemDynamics
    )
        new(nsteps, dt, x0, tspan, method, sys)
    end
end

# ODE method struct
abstract type ODEMethod end
mutable struct RK <: ODEMethod
    # number of stages
    #nstages::Int
    # coefficients
    #a::Matrix{Float64}
    #b::Vector{Float64}
    #c::Vector{Float64}
    # temporary storage
    k::Union{Vector{Vector{Float64}}, Nothing}
    function RK()
        k = nothing
        new(k)
    end
end

function preallocate!(ivp::IVP)
    # preallocate storage for k vectors
    ivp.method.k = [zeros(size(ivp.x0)) for i in 1:4]
end

include("ivp.jl")
include("psys.jl")

end # module CLSDyn
