using LinearAlgebra
using ForwardDiff
using Plots

function lorenz_rhs!(dx, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    dx[1] = σ*(x[2] - x[1])
    dx[2] = x[1]*(ρ - x[3]) - x[2]
    dx[3] = x[1]*x[2] - β*x[3]
end

function lorenz_fx!(df, dx, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -σ*dx[1] + σ*dx[2]
    df[2] = (ρ - x[3])*dx[1] - dx[2] - x[1]*dx[3]
    df[3] = x[2]*dx[1] + x[1]*dx[2] - β*dx[3]
end

function lorenz_fp!(df, dp, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] += (x[2] - x[1])*dp[1]
    df[2] += x[1]*dp[2]
    df[3] += -x[3]*dp[3]
end

function rk4_step!(
    xnew::AbstractArray,
    f::Function,
    xold::AbstractArray,
    p::AbstractArray,
    t::Float64,
    dt::Float64
)
    # Note: this is a bit inefficient because I allocate the k's each time
    # i take a step. the best way might be to pass a data structure
    # where the k's are already allocated.
    ndim = size(xold, 1)
    k1 = zeros(ndim)
    k2 = zeros(ndim)
    k3 = zeros(ndim)
    k4 = zeros(ndim)
    f(k1, xold, p, t)
    f(k2, xold + (dt/2)*k1, p, t + 0.5*dt)
    f(k3, xold + (dt/2)*k2, p, t + 0.5*dt)
    f(k4, xold + dt*k3, p, t + dt)
    xnew .= xold .+ (dt/6)*k1 .+ (dt/3)*k2 .+ (dt/3)*k3 .+ (dt/6)*k4
end

function rk4(f::Function, x0::Vector, p::Vector, tspan::Tuple, nsteps::Int)
    """
        The function f is of the form
        f(dx, x, p, t)
    """
    ndim = size(x0, 1)
    traj = zeros((ndim, nsteps))
    traj[:,1] .= x0
    dt = (tspan[2] - tspan[1])/nsteps
    tvec = collect(range(start=tspan[1], stop=tspan[2], length=nsteps))
    for i=1:(nsteps - 1)
        t = tvec[i]
        rk4_step!(@view(traj[:, i + 1]), f, @view(traj[:, i]), p, t, dt)
    end
    return (traj, tvec)
end

"""
        tlm(fx, fp, dx, traj, tvec[, pidx])

Computes the Tangent Linear Model. If pidx = 0, computes the TLM with respect
to given perturbation direction δx0. Otherwise, computes TLM with respect to
perturbation of parameter pidx.

Important: fp must _add_ values to the vector.

"""
function tlm(
    fx::Function,
    fp::Function,
    dx0::Vector,
    traj::AbstractArray,
    tvec::AbstractArray,
    p::AbstractArray;
    dp::Union{AbstractArray, Nothing}=nothing,
)

    nsteps = size(tvec, 1)
    tlm_traj = similar(traj)
    tlm_traj[:,1] .= dx0

    # we create a closure such tat we can use a time stepper that takes a function
    #    f(dx, x, p, t)
    # but instead of changing the parameter, we change "x" at each step, that is given
    # by the solution of the forward model.

    function tlm_rhs!(df, dx, x, t)
        fx(df, dx, x, p, t)
        if dp != nothing
            fp(df, dp, x, p, t)
        end
    end

    nsteps = size(tvec, 1)
    for i = 1:(nsteps - 1)
        t = tvec[i]
        x = @view(traj[:, i])
        dt = tvec[i + 1] - tvec[i]
        rk4_step!(@view(tlm_traj[:, i + 1]), tlm_rhs!, @view(tlm_traj[:, i]), x, t, dt)
    end

    return tlm_traj
end

# TEST PROBLEM
tspan=(0.0, 0.5)
pvec = [10.0, 5.0, 8.0/3.0]
x0 = [1.0, 1.0, 1.0]
nsteps = 500


idx_var = 1
traj, tvec = rk4(lorenz_rhs!, x0, pvec, tspan, nsteps)
ps = plot(tvec, traj[idx_var,:])

# COMPUTE TLM with given dx perturbation
println("Testing TLM")
# 1.- integrate perturbed system and obtain sensitivities
#     via finite differences at t_end
ϵ = 1e-5
x0eps = [1.0 + ϵ, 1.0, 1.0]
traj2, tvec2 = rk4(lorenz_rhs!, x0eps, pvec, tspan, nsteps)
println((traj2[:,end]-traj[:,end])/ϵ)

# 2.- integrate TLM
dx0 = [1.0, 0.0, 0.0]
tlm_traj  = tlm(lorenz_fx!, lorenz_fp!, dx0, traj, tvec, pvec)

println(tlm_traj[:,end])

# COMPUTE TLM with given dx perturbation
println("Testing parametric TLM")

# 1.- integrate perturbed system and compute finite differences
peps = [pvec[1], pvec[2] + ϵ, pvec[3]]
traj3, tvec3 = rk4(lorenz_rhs!, x0, peps, tspan, nsteps)
println((traj3[:,end]-traj[:,end])/ϵ)

# 2.- integrate TLM with parameters
dx0 = [0.0, 0.0, 0.0]
dp = [0.0, 1.0, 0.0]
tlm_traj2  = tlm(lorenz_fx!, lorenz_fp!, dx0, traj, tvec, pvec, dp=dp)
println(tlm_traj2[:,end])
