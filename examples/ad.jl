using LinearAlgebra
using ForwardDiff
using Plots
using Calculus
using FiniteDifferences
using Enzyme

# Lorenz system
abstract type Dynamics end

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
end

function rk4(dynamics::Dynamics)
    """
        The function f is of the form
        f(dx, x, p, t)
    """
    dynamics.traj[:,1] .= dynamics.x0
    for dynamics.step=1:(dynamics.nsteps - 1)
        rk4_step!(dynamics)
    end
    return (dynamics.traj, dynamics.tvec)
end

function adj_rk4(dynamics::Dynamics)
    """
        The function f is of the form
        f(dx, x, p, t)
    """
    dynamics.traj[:,1] .= dynamics.x0
    for dynamics.step in 1:dynamics.nsteps-1
        rk4_step!(dynamics)
    end
    return (dynamics.traj, dynamics.tvec)
end

# It follows user code

mutable struct Lorenz <: Dynamics
    x0::Vector{Float64}
    p::Vector{Float64}
    tspan::Tuple{Float64, Float64}
    nsteps::Int
    step::Int
    traj::Matrix{Float64}
    tvec::Vector{Float64}
    dt::Float64
    k::Vector{Vector{Float64}}
    x::Vector{Vector{Float64}}
    t::Vector{Float64}
end

function Lorenz(x0, pvec, tspan, nsteps)
    ndim = length(x0)
    traj = zeros((ndim, nsteps))
    dt = (tspan[2] - tspan[1])/nsteps
    tvec = collect(range(start=tspan[1], stop=tspan[2], length=nsteps))
    ndim = length(x0)
    k = [zeros(ndim), zeros(ndim), zeros(ndim), zeros(ndim)]
    x = [zeros(ndim), zeros(ndim), zeros(ndim), zeros(ndim)]
    t = zeros(4)
    return Lorenz(x0, pvec, tspan, nsteps, 0, traj, tvec, dt, k, x, t)
end

function rhs!(dx, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    dx[1] = σ*(x[2] - x[1])
    dx[2] = x[1]*(ρ - x[3]) - x[2]
    dx[3] = x[1]*x[2] - β*x[3]
end


# TEST PROBLEM
tspan=(0.0, 0.1)
pvec = [2.0, 1.0, 8.0/3.0]
pvec = [10.0, 8.0, 8.0/3.0]
x0 = [1.0, 1.0, 1.0]
nsteps = 1000

idx_var = 1

lorenz = Lorenz(x0, pvec, tspan, nsteps)
traj, tvec = rk4(lorenz)

r(x) = (x[1]^ 2) ./ 2.0

function numerical_integration(lorenz::Lorenz)
    traj, tvec = rk4(lorenz)
    val = 0.0
    for i=1:(lorenz.nsteps-1)
        val += (tvec[i+1] - tvec[i])*r(traj[:, i+1])
    end
    return val
end

lorenz = Lorenz(x0, pvec, tspan, nsteps)
dlorenz = Lorenz(zeros(3), zeros(3), tspan, nsteps)
Enzyme.autodiff(Reverse, numerical_integration, Active, Duplicated(lorenz, dlorenz))

println("Gradient: ", dlorenz.x0)

# revolve = Revolve{Lorenz}(nsteps, nsteps; verbose=1)
# Checkpointing.reset(revolve)
# dlorenz = Zygote.gradient(cost, lorenz, revolve)



# ps = plot(tvec, traj[idx_var,:])



