using LinearAlgebra
using ForwardDiff
using Plots
using Calculus
using FiniteDifferences
using Enzyme

# Lorenz system
abstract type Dynamics{FT} end

rhs!(::Dynamics, dx, x, p, t) = error("rhs! not implemented")

function rk4_step!(
    dynamics::Dynamics,
)
    # Note: this is a bit inefficient because I allocate the k's each time
    # i take a step. the best way might be to pass a data structure
    # where the k's are already allocated.
    k = dynamics.k
    x = dynamics.x
    t = dynamics.t
    p = dynamics.p
    dt = dynamics.dt
    step = dynamics.step
    tvec = dynamics.tvec[step]
    xold = @view(dynamics.traj[:,step])
    xnew = @view(dynamics.traj[:,step+1])

    t[1] = tvec
    t[2] = tvec + 0.5*dt
    t[3] = tvec + 0.5*dt
    t[4] = tvec + dt
    x[1] = xold
    rhs!(k[1], x[1], p, t[1])
    x[2] = xold + (dt/2)*k[1]
    rhs!(k[2], x[2], p, t[2])
    x[3] = xold + (dt/2)*k[2]
    rhs!(k[3], x[3], p, t[3])
    x[4] = xold + dt*k[3]
    rhs!(k[4], x[4], p, t[4])
    xnew .= xold .+ (dt/6)*k[1] .+ (dt/3)*k[2] .+ (dt/3)*k[3] .+ (dt/6)*k[4]
    return nothing
end

function rk4!(dynamics::Dynamics)
    """
        The function f is of the form
        f(dx, x, p, t)
    """
    dynamics.traj[:,1] .= dynamics.x0
    for dynamics.step in 1:(dynamics.nsteps - 1)
        rk4_step!(dynamics)
    end
    return nothing
end

# It follows user code

mutable struct Lorenz{FT} <: Dynamics{FT}
    x0::Vector{FT}
    p::Vector{FT}
    nsteps::Int
    step::Int
    traj::Matrix{FT}
    tvec::Vector{FT}
    dt::FT
    k::Vector{Vector{FT}}
    x::Vector{Vector{FT}}
    t::Vector{FT}
end

function Lorenz{FT}(x0::Vector{FT}, pvec::Vector{FT}, tspan::Tuple{FT,FT}, dt::FT, nsteps::Int) where FT
    ndim = length(x0)
    traj = zeros(FT, (ndim, nsteps))
    # dt = (tspan[2] - tspan[1])/nsteps
    tvec = FT.(collect(range(start=tspan[1], stop=tspan[2], length=nsteps)))
    ndim = length(x0)
    k = [zeros(FT, ndim), zeros(FT, ndim), zeros(FT, ndim), zeros(FT, ndim)]
    x = [zeros(FT, ndim), zeros(FT, ndim), zeros(FT, ndim), zeros(FT, ndim)]
    t = zeros(FT, 4)
    return Lorenz(x0, pvec, nsteps, 0, traj, tvec, dt, k, x, t)
end

function rhs!(dx, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    dx[1] = σ*(x[2] - x[1])
    dx[2] = x[1]*(ρ - x[3]) - x[2]
    dx[3] = x[1]*x[2] - β*x[3]
    return nothing
end


# TEST PROBLEM
tspan=(0.0, 0.5)
pvec = [2.0, 1.0, 8.0/3.0]
# pvec = [10.0, 8.0, 8.0/3.0]
x0 = [1.0, 1.0, 1.0]
nsteps = 1000
dt = (tspan[2] - tspan[1])/nsteps

idx_var = 1

lorenz = Lorenz{Float64}(x0, pvec, tspan, dt, nsteps)
rk4!(lorenz)

r(x) = (x[1]^ 2) ./ 2.0

function numerical_integration(val::Vector{FT}, lorenz::Lorenz{FT}) where {FT}
    rk4!(lorenz)
    val[1] = 0.0
    for i=1:(lorenz.nsteps-1)
        val[1] += (lorenz.tvec[i+1] - lorenz.tvec[i])*r(lorenz.traj[:, i+1])
    end
    return nothing
end

# Finite difference
h = 1e-6
lorenz = Lorenz{Float64}(x0, pvec, tspan, dt, nsteps)
fdlorenz = Lorenz{Float64}([1.0+h, 1.0, 1.0], pvec, tspan, dt, nsteps)

val = zeros(1)
fdval = zeros(1)
numerical_integration(val, lorenz)
numerical_integration(fdval, fdlorenz)
println("Function evaluation: ", val[1])
println("FD gradient: ", (fdval[1]-val[1])/h)

# Reverse
lorenz = Lorenz{Float64}(x0, pvec, tspan, dt, nsteps)
rlorenz = Lorenz{Float64}(zeros(3), zeros(3), (0.0,0.0), 0.0, nsteps)

val = zeros(1)
rval = ones(1)
Enzyme.autodiff(Reverse, numerical_integration, Const, Duplicated(val, rval), Duplicated(lorenz, rlorenz))

println("Reverse AD gradient: ", rlorenz.x0[1])

# Forward
lorenz = Lorenz{Float64}(x0, pvec, tspan, dt, nsteps)
florenz = Lorenz{Float64}(Float64[1.0, 0.0, 0.0], zeros(3), (0.0,0.0), 0.0, nsteps)

val = zeros(1)
fval = zeros(1)

Enzyme.autodiff(Forward, numerical_integration, Const, Duplicated(val, fval), Duplicated(lorenz, florenz))

println("Forward AD gradient: ", fval[1])

# BROKEN: Forward over reverse
# lorenz = Lorenz(x0, pvec, tspan, dt, nsteps)
# florenz = Lorenz(Float64[1.0, 0.0, 0.0], zeros(3), (0.0,0.0), 0.0, nsteps)
# rlorenz = Lorenz(zeros(3), zeros(3), (0.0,0.0), 0.0, nsteps)
# rflorenz = Lorenz(zeros(3), zeros(3), (0.0,0.0), 0.0, nsteps)

# val = zeros(1)
# fval = zeros(1)

# rval = ones(1)
# rfval = zeros(1)

# Enzyme.autodiff(
#     Forward,
#     (x,y) -> Enzyme.autodiff_deferred(Reverse, numerical_integration, Const, x, y),
#     Const,
#     Duplicated(Duplicated(val, rval), Duplicated(fval, rfval)),
#     Duplicated(Duplicated(lorenz, rlorenz), Duplicated(florenz, rflorenz)),
# )
# println("FoR AD: ", fval[1], " ", rlorenz.x0[1])

# Reverse over Forward using ForwardDiff

dtspan = (ForwardDiff.Dual(tspan[1], 0.0), ForwardDiff.Dual(tspan[2], 0.0))
dpvec = ForwardDiff.Dual.([2.0, 1.0, 8.0/3.0], [0.0, 0.0, 0.0])
dx0 = ForwardDiff.Dual.([1.0, 1.0, 1.0], [1.0, 0.0, 0.0])
nsteps = 1000
ddt = ForwardDiff.Dual((tspan[2] - tspan[1])/nsteps, 0.0)

lorenz = Lorenz{ForwardDiff.Dual{Nothing, Float64, 1}}(dx0, dpvec, dtspan, ddt, nsteps)
rlorenz = Lorenz{ForwardDiff.Dual{Nothing, Float64, 1}}(
    ForwardDiff.Dual.(zeros(3),zeros(3)),
    ForwardDiff.Dual.(zeros(3),zeros(3)),
    (ForwardDiff.Dual(0.0, 0.0), ForwardDiff.Dual(0.0, 0.0)),
    ForwardDiff.Dual(0.0, 0.0),
    nsteps
)

val = zeros(ForwardDiff.Dual{Nothing, Float64,1}, 1)
rval = ForwardDiff.Dual.([0.0], [1.0])
Enzyme.autodiff(Reverse, numerical_integration, Const, Duplicated(val, rval), Duplicated(lorenz, rlorenz))

println("Reverse over Forward AD gradient: ", ForwardDiff.partials.(rlorenz.x0[1])[1])
println("Reverse over Forward AD second-order: ", ForwardDiff.value.(rlorenz.x0[1]))
# Finite difference
h = 1e-2
lorenz = Lorenz{Float64}(x0, pvec, tspan, dt, nsteps)
fdlorenz = Lorenz{Float64}([1.0+h, 1.0, 1.0], pvec, tspan, dt, nsteps)
ffdlorenz = Lorenz{Float64}([1.0+2*h, 1.0, 1.0], pvec, tspan, dt, nsteps)

val = zeros(1)
fdval = zeros(1)
ffdval = zeros(1)
numerical_integration(val, lorenz)
numerical_integration(fdval, fdlorenz)
numerical_integration(ffdval, ffdlorenz)
println("FD second-order: ", (ffdval[1]-2*fdval[1]+val[1])/(h*h))
