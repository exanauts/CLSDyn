module CLSDyn
using ForwardDiff

import PowerModels

const _WS = 1.0

checkarguments(f, argtypes) = length(methods(f, argtypes)) > 0

abstract type SystemContext end

struct PSystem <: SystemContext
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
    ctx::SystemContext
    pvec::AbstractArray
    rhs!::Function

    fx!::Union{Function, Nothing}
    fp!::Union{Function, Nothing}
    fx_transpose!::Union{Function, Nothing}
    fp_transpose!::Union{Function, Nothing}
    fxx!::Union{Function, Nothing}
    fpp!::Union{Function, Nothing}
    fxp!::Union{Function, Nothing}

    function SystemDynamics(ctx::SystemContext, pvec::AbstractArray, input_rhs::Function)
        function rhs!(dx, x, p, t)
            input_rhs(dx, x, p, t, ctx)
        end
        new(ctx, pvec, rhs!, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end

    function SystemDynamics(ctx::PSystem, input_rhs::Function)
        # in this function we assemble the parameters into a vector
        # and store it in the SystemDynamics struct. This is so that
        # the interfaces to gradients and hessians become clean.
        pvec = vcat(ctx.vmag, ctx.pmec, ctx.gen_inertia, ctx.gen_damping)

        # the input rhs function should be of the form
        #   rhs!(dx, x, p, t, ctx)
        # and here we create a wrapper.
        # we might need to enforce type signature with checkarguments.
        function rhs!(dx, x, p, t)
            input_rhs(dx, x, p, t, ctx)
        end
        new(ctx, pvec, rhs!, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end

end

function set_fx!(sys::SystemDynamics, input_fx::Function)
    function fx!(fx, dx, x, p, t)
        input_fx(fx, dx, x, p, t, sys.ctx)
    end
    sys.fx! = fx!
end

function set_fp!(sys::SystemDynamics, input_fp::Function)
    function fp!(fp, dx, x, p, t)
        input_fp(fp, dx, x, p, t, sys.ctx)
    end
    sys.fp! = fp!
end


abstract type IVPAbstract end
abstract type ODEMethod end

struct IVP <: IVPAbstract
    # number of steps
    nsteps::Int
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
        x0::AbstractArray,
        tspan::Tuple{Float64,Float64},
        method::ODEMethod,
        sys::SystemDynamics
    )
        new(nsteps, x0, tspan, method, sys)
    end
end

mutable struct RK <: ODEMethod
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
include("sensitivities.jl")

# Export symbols
export PSystem, SystemDynamics
export IVP, RK

end # module CLSDyn
