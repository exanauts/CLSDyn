module CLSDyn
using ForwardDiff
using NLPModels

import PowerModels

const _WS = 1.0

checkarguments(f, argtypes) = length(methods(f, argtypes)) > 0

abstract type SystemContext end

"""
    PSystem

Holds information related to a power system, including the number of buses, generators,
and loads, as well as the voltage magnitude vector, reduced admittance matrix, 
mechanical power, generator inertia, and generator damping.

# Fields

- `nbus::Int`: Number of buses
- `ngen::Int`: Number of generators
- `nload::Int`: Number of loads
- `vmag::Vector{Float64}`: Voltage magnitude vector
- `yred::Matrix{ComplexF64}`: Reduced admittance matrix
- `pmec::Vector{Float64}`: Mechanical power vector
- `gen_inertia::Vector{Float64}`: Generator inertia vector
- `gen_damping::Vector{Float64}`: Generator damping vector

"""
struct PSystem <: SystemContext
    nbus::Int
    ngen::Int
    nload::Int
    vmag::Vector{Float64}
    yred::Matrix{ComplexF64}
    pmec::Vector{Float64}
    gen_inertia::Vector{Float64}
    gen_damping::Vector{Float64}
end

"""
    SystemDynamics

Represents the system dynamics, with functions for the right-hand side of the differential 
equations, their first and second derivatives, and the associated parameters vector.

# Fields

- `ctx::SystemContext`: The context of the system
- `pvec::AbstractArray`: Vector of parameters
- `rhs!::Function`: Function for the right-hand side of the system's dynamics
- Other derivative functions

"""
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
    fpx!::Union{Function, Nothing}

    function SystemDynamics(ctx::SystemContext, pvec::AbstractArray, input_rhs::Function)
        function rhs!(dx, x, p, t)
            input_rhs(dx, x, p, t, ctx)
        end
        new(ctx, pvec, rhs!, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
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
        new(ctx, pvec, rhs!, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end

end

"""
    CostFunctional

Encapsulates the cost functional associated with the system dynamics, including the integral term 
and terminal condition, and their respective gradient computations.

# Fields

- `sys::SystemDynamics`: Reference to the system dynamics
- `r!::Union{Function, Nothing}`: Function for the integral term
- `w!::Union{Function, Nothing}`: Function for the terminal condition
- Other gradient functions

"""
mutable struct CostFunctional
    # system dynamics reference
    # NOTE: It seems we need to have a reference to the system dynamics
    # within the cost functional struct. Otherwise, after creating the wrapper
    # functions, when calling cost.rx!(dr, x0, pvec, t), we get an error
    #  "ERROR: type CostFunctional has no field sys"
    #  WHY?
    sys::SystemDynamics
    # integral term
    r!::Union{Function, Nothing}
    # terminal condition
    w!::Union{Function, Nothing}

    # integral term gradients
    rx!::Union{Function, Nothing}
    rp!::Union{Function, Nothing}
    rxx!::Union{Function, Nothing}
    rpp!::Union{Function, Nothing}
    rxp!::Union{Function, Nothing}
    rpx!::Union{Function, Nothing}

    function CostFunctional(
            sys::SystemDynamics, 
            r_input::Union{Function, Nothing},
            w_input::Union{Function, Nothing}
    )
        # cost functional is associated to system dynamics.
        # context of system dynamics is used to create the
        # wrapped functions.
        ctx = sys.ctx

        if r_input != nothing
            function r!(x, p, t)
                r_input(x, p, t, ctx)
            end
        else
            r! = nothing
        end

        if w_input != nothing
            function w!(x, p, t)
                w_input(x, p, t, ctx)
            end
        else
            w! = nothing
        end
        new(sys, r!, w!, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

# setters for gradients in SystemDynamics and CostFunctional.
# This is a lot of boilerplate code. I am not sure if there is a
# better way to do this (that does not involve unnecessary complexity).

function set_fx!(sys::SystemDynamics, input_fx::Function)
    function fx!(df, dx, x, p, t)
        input_fx(df, dx, x, p, t, sys.ctx)
    end
    sys.fx! = fx!
end

function set_fp!(sys::SystemDynamics, input_fp::Function)
    function fp!(df, dx, x, p, t)
        input_fp(df, dx, x, p, t, sys.ctx)
    end
    sys.fp! = fp!
end

function set_fx_trans!(sys::SystemDynamics, input_fx_trans::Function)
    function fx_trans!(df, dx, x, p, t)
        input_fx_trans(df, dx, x, p, t, sys.ctx)
    end
    sys.fx_transpose! = fx_trans!
end

function set_fp_trans!(sys::SystemDynamics, input_fp_trans::Function)
    function fp_trans!(df, dx, x, p, t)
        input_fp_trans(df, dx, x, p, t, sys.ctx)
    end
    sys.fp_transpose! = fp_trans!
end

function set_fxx!(sys::SystemDynamics, input_fxx::Function)
    function fxx!(df, u, z, x, p, t)
        input_fxx(df, u, z, x, p, t, sys.ctx)
    end
    sys.fxx! = fxx!
end

function set_fpp!(sys::SystemDynamics, input_fpp::Function)
    function fpp!(df, u, z, x, p, t)
        input_fpp(df, u, z, x, p, t, sys.ctx)
    end
    sys.fpp! = fpp!
end

function set_fxp!(sys::SystemDynamics, input_fxp::Function)
    function fxp!(df, u, z, x, p, t)
        input_fxp(df, u, z, x, p, t, sys.ctx)
    end
    sys.fxp! = fxp!
end

function set_fpx!(sys::SystemDynamics, input_fpx::Function)
    function fpx!(df, u, z, x, p, t)
        input_fpx(df, u, z, x, p, t, sys.ctx)
    end
    sys.fpx! = fpx!
end

function set_rx!(cf::CostFunctional, input_rx::Function)
    function rx!(dr, x, p, t)
        input_rx(dr, x, p, t, cf.sys.ctx)
    end
    cf.rx! = rx!
end

function set_rp!(cf::CostFunctional, input_rp::Function)
    function rp!(dr, x, p, t)
        input_rp(dr, x, p, t, cf.sys.ctx)
    end
    cf.rp! = rp!
end

function set_rxx!(cf::CostFunctional, input_rxx::Function)
    function rxx!(dr, dx, x, p, t)
        input_rxx(dr, dx, x, p, t, cf.sys.ctx)
    end
    cf.rxx! = rxx!
end

function set_rpp!(cf::CostFunctional, input_rpp::Function)
    function rpp!(dr, dx, x, p, t)
        input_rpp(dr, dx, x, p, t, cf.sys.ctx)
    end
    cf.rpp! = rpp!
end

function set_rxp!(cf::CostFunctional, input_rxp::Function)
    function rxp!(dr, dx, x, p, t)
        input_rxp(dr, dx, x, p, t, cf.sys.ctx)
    end
    cf.rxp! = rxp!
end

function set_rpx!(cf::CostFunctional, input_rpx::Function)
    function rpx!(dr, dx, x, p, t)
        input_rpx(dr, dx, x, p, t, cf.sys.ctx)
    end
    cf.rpx! = rpx!
end


# IVP and ODE solver types
abstract type IVPAbstract end
abstract type ODEMethod end

"""
    IVP

Represents an initial value problem (IVP) that consists of the number of steps, initial condition,
time span, method of ODE solver, and system dynamics.

# Fields

- `nsteps::Int`: Number of steps
- `x0::AbstractArray`: Initial condition
- `tspan::Tuple{Float64,Float64}`: Time span
- `method::ODEMethod`: ODE solver method
- `sys::SystemDynamics`: System dynamics

"""
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

function preallocate!(rk::RK, dim::Int)
    rk.k = [zeros(dim) for i in 1:4]
end

function preallocate!(ivp::IVP)
    # preallocate storage for k vectors
    #ivp.method.k = [zeros(size(ivp.x0)) for i in 1:4]
    preallocate!(ivp.method, size(ivp.x0, 1))
end

function preallocate!(ivp::IVP, dim::Int)
    # preallocate storage for k vectors
    ivp.method.k = [zeros(dim) for i in 1:4]
end

include("ivp.jl")
include("psys.jl")
include("sensitivities.jl")
include("nlp.jl")

# Export symbols
export PSystem, SystemDynamics
export DynamicNLP
export IVP, RK

end # module CLSDyn
