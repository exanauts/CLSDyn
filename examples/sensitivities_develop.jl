using LinearAlgebra
using ForwardDiff
using Plots
using Calculus
using FiniteDifferences

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

function lorenz_fx_trans!(df, dx, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -σ*dx[1] + (ρ - x[3])*dx[2] + x[2]*dx[3]
    df[2] = σ*dx[1] - dx[2] + x[1]*dx[3]
    df[3] = -x[1]*dx[2] - β*dx[3]
end

function lorenz_fp!(df, dp, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] += (x[2] - x[1])*dp[1]
    df[2] += x[1]*dp[2]
    df[3] += -x[3]*dp[3]
end

function lorenz_fp_trans!(df, dp, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] += (x[2] - x[1])*dp[1]
    df[2] += x[1]*dp[2]
    df[3] += -x[3]*dp[3]
end

function lorenz_fxx!(df, u, z, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = u[2]*z[3] - z[2]*u[3]
    df[2] = u[1]*z[3]
    df[3] = -u[1]*z[2]
end

function lorenz_fxp!(df, u, z, x, p, t)
    # Note: this is the transpose of the above function
    #   u = δθ
    #   z = λ
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -z[1]*u[1] + z[2]*u[2]
    df[2] = z[1]*u[1]
    df[3] = -z[3]*u[3]
end

function lorenz_fpx!(df, u, z, x, p, t)
    # Note: this is the transpose of the above function
    #   u = δx
    #   z = λ
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -z[1]*u[1] + z[1]*u[2]
    df[2] = z[2]*u[1]
    df[3] = -z[3]*u[3]
end

function lorenz_fpp!(df, u, z, x, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = 0.0
    df[2] = 0.0
    df[3] = 0.0
end


function rk4_step!(
    xnew::AbstractArray,
    f::Function,
    xold::AbstractArray,
    p,
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

function rk4(
    f::Function,
    x0::AbstractArray,
    p::AbstractArray,
    tspan::Tuple,
    nsteps::Int
)
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
        tlm(fx, fp, dx, traj, tvec, p[, dp])

Computes the Tangent Linear Model. If dp = nothing, computes the TLM with respect
to given perturbation direction δx0. Otherwise, computes TLM with respect to
perturbation of parameter perturbation dp.

Important: fp must _add_ values to the vector.

"""
function tlm(
    fx::Function,
    fp::Function,
    dx0::AbstractArray,
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

"""
    adjoint_sens

Compute adjoint sensitivities with respect to functional.
"""
function adjoint_sens(
    fx_trans::Function,
    fp_trans::Function,
    rx::Function,
    rp::Function,
    traj::AbstractArray,
    tvec::AbstractArray,
    p::AbstractArray,
    λ0::AbstractArray;
    λ_traj::Union{AbstractArray, Nothing}=nothing,
)

    # obtain dimensions
    xdim = size(traj, 1)
    pdim = size(p, 1)
    @assert xdim == size(λ0, 1)
    nsteps = size(tvec, 1)

    # check if we need to store trajectory of λ
    if λ_traj == nothing
        store_trajectory = false
    else
        store_trajectory = true
        @assert size(λ_traj, 1) == size(traj, 1)
        @assert size(λ_traj, 2) == size(traj, 2)
        λ_traj[:, end] .= λ0
    end

    μ0 = zeros(pdim)
    u0 = vcat(λ0, μ0)
    u = similar(u0)
    u .= u0
    unext = similar(u0)
    
    function adj_sys!(du, u, x, t)
        # this is a preliminary, inneficient implementation.
        λ = @view(u[1:xdim])
        μ = @view(u[xdim + 1:end])
        dλ = @view(du[1:xdim])
        dμ = @view(du[xdim + 1:end])

        drx = zeros(xdim)
        drp = zeros(pdim)
        
        fx_trans(dλ, λ, x, p, t)
        rx(drx, x, p)
        dλ .+= drx
        # it should be negative. but it works when positive.
        # what is going on?
        #dλ .*= -1
        
        fp_trans(dμ, λ, x, p, t)
        rp(drp, x, p)
        dμ .+= drp
    end
    
    nsteps = size(tvec, 1)
    for i = reverse(2:nsteps)
        t = tvec[i]
        x = @view(traj[:, i])
        dt = tvec[i] - tvec[i - 1]
        rk4_step!(unext, adj_sys!, u, x, t, dt)
        u .= unext
        if store_trajectory == true
            λ_traj[:, i - 1] .= unext[1:xdim]
        end
    end
    # this is dg/dx0
    λ = u[1:xdim]
    # this is dg/dp
    μ = u[xdim + 1:end]
    return (λ, μ)
end

"""
    forward_over_adjoint_sens

Compute forward over adjoint sensitivities given a perturbation direction, v.
"""
function forward_over_adjoint_sens(
    v::AbstractVector,
    fx::Function,
    fp::Function,
    fx_trans::Function,
    fp_trans::Function,
    fxx_trans::Function,
    traj::AbstractArray,
    λ_traj::AbstractArray,
    tvec::AbstractArray,
    p::AbstractArray,
)

    # obtain dimensions
    xdim = size(traj, 1)
    pdim = size(p, 1)
    @assert xdim == size(λ0, 1)
    nsteps = size(tvec, 1)

    # we first need to compute the trajectory of the TLM given direction
    # perhaps we can compute this externally such that we do not assume
    # it had not been computed.
    tlm_traj  = tlm(fx, fp, v, traj, tvec, p)

    # preallocate vectors
    σ0 = zeros(xdim)
    σ = similar(σ0)
    σ .= σ0
    σnext = similar(σ0)
    
    # then we form the residual function
    function fwd_adj_sys!(dσ, σ, ctx, t)
        # retrieve information
        x, λ, δx = ctx

        # NOTE: do not preallocate here. will be moved to context
        fx_vec = zeros(xdim)
        fxx_vec = zeros(xdim)
        
        fx_trans(fx_vec, σ, x, p, t)
        fxx_trans(fxx_vec, δx, λ, x, p, t)

        dσ .= fx_vec + fxx_vec
    end
    
    nsteps = size(tvec, 1)
    for i = reverse(2:nsteps)
        t = tvec[i]
        x = @view(traj[:, i])
        λ = @view(λ_traj[:, i])
        δx = @view(tlm_traj[:, i])
        ctx = (x, λ, δx)        
        dt = tvec[i] - tvec[i - 1]
        rk4_step!(σnext, fwd_adj_sys!, σ, ctx, t, dt)
        σ .= σnext
    end
    println(σ)
end

"""
    forward_over_adjoint_sens2

Compute forward over adjoint sensitivities given a perturbation direction, v.
"""
function forward_over_adjoint_sens2(
    v::AbstractVector,
    δθ::AbstractVector,
    fx::Function,
    fp::Function,
    fx_trans::Function,
    fp_trans::Function,
    fxx_trans::Function,
    fxp_trans::Function,
    fpx_trans::Function,
    fpp_trans::Function,
    rxx::Function,
    rxp::Function,
    rpx::Function,
    rpp::Function,
    traj::AbstractArray,
    λ_traj::AbstractArray,
    tvec::AbstractArray,
    p::AbstractArray,
)

    # obtain dimensions
    xdim = size(traj, 1)
    pdim = size(p, 1)
    @assert xdim == size(λ0, 1)
    nsteps = size(tvec, 1)

    # we first need to compute the trajectory of the TLM given direction
    # perhaps we can compute this externally such that we do not assume
    # it had not been computed.
    tlm_traj  = tlm(fx, fp, v, traj, tvec, p, dp=δθ)

    # preallocate vectors
    σ0 = zeros(xdim)
    τ0 = zeros(pdim)
    u0 = vcat(σ0, τ0)
    u = similar(u0)
    u .= u0
    unext = similar(u0)
    
    # then we form the residual function
    function fwd_adj_sys!(du, u, ctx, t)
        # retrieve information
        x, λ, δx = ctx
        σ = @view(u[1:xdim])
        τ = @view(u[xdim + 1:end])
        dσ = @view(du[1:xdim])
        dτ = @view(du[xdim + 1:end])

        # NOTE: do not preallocate here. will be moved to context
        fx_vec = zeros(xdim)
        fxx_vec = zeros(xdim)
        fxp_vec = zeros(xdim)
        rxx_vec = zeros(xdim)
        rxp_vec = zeros(xdim)
        
        # mat-vec products
        fx_trans(fx_vec, σ, x, p, t)
        fxx_trans(fxx_vec, δx, λ, x, p, t)
        fxp_trans(fxp_vec, δθ, λ, x, p, t)
        rxx(rxx_vec, δx, x, p)
        rxp(rxp_vec, δθ, x, p)

        dσ .= fx_vec + fxx_vec + fxp_vec + rxx_vec + rxp_vec

        # NOTE: do not preallocate here. will be moved to context
        fx_vec = zeros(xdim)
        fp_vec = zeros(pdim)
        fpx_vec = zeros(pdim)
        fpp_vec = zeros(pdim)
        rpx_vec = zeros(pdim)
        rpp_vec = zeros(pdim)

        # mat-vec products
        fp_trans(fp_vec, σ, x, p, t)
        fpx_trans(fpx_vec, δx, λ, x, p, t)
        fpp_trans(fpp_vec, δθ, λ, x, p, t)
        rpx(rpx_vec, δx, x, p)
        rpp(rpp_vec, δθ, x, p)

        dτ .= fp_vec + fpx_vec + fpp_vec + rpx_vec + rpp_vec
    end
    
    nsteps = size(tvec, 1)
    for i = reverse(2:nsteps)
        t = tvec[i]
        x = @view(traj[:, i])
        λ = @view(λ_traj[:, i])
        δx = @view(tlm_traj[:, i])
        ctx = (x, λ, δx)        
        dt = tvec[i] - tvec[i - 1]
        rk4_step!(unext, fwd_adj_sys!, u, ctx, t, dt)
        u .= unext
    end
    # this is the sensitivity of the cost function with respect to the
    # parameters
    σ = u[1:xdim]
    # this is the sensitivity of the cost function with respect to the
    # initial condition
    τ = u[xdim + 1:end]
    return (σ, τ)
end



# TEST PROBLEM
tspan=(0.0, 0.5)
pvec = [2.0, 1.0, 8.0/3.0]
#pvec = [10.0, 8.0, 8.0/3.0]
x0 = [1.0, 1.0, 1.0]
nsteps = 1000


idx_var = 1
traj, tvec = rk4(lorenz_rhs!, x0, pvec, tspan, nsteps)
ps = plot(tvec, traj[idx_var,:])

# COMPUTE TLM with given dx perturbation
println("Testing TLM")
# 1.- integrate perturbed system and obtain sensitivities
#     via finite differences at t_end
ϵ = 1e-7
x0eps = [x0[1] + ϵ, x0[2], x0[3]]
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

# COMPUTE sensitivities of cost functional
println("Test first-order adjoint (terminal condition w.r.t x0)")
# r(x, p) -> functional cost (integral from t0 to tf).
# w(x(tf), p) -> terminal cost.

# Functional, gradient of functional, and Hessian-vec products

r(x, p) = (x[1] .^ 2) ./ 2
function rx(dr, x, p)
    dr[1] = x[1]
end
function rp(dr, x, p)
end

function rxx(dr, dx, x, p)
    dr[1] = dx[1]
end

function rxp(dr, dx, x, p)
end

function rpx(dr, dx, x, p)
end

function rpp(dr, dx, x, p)
end

function caca1!(df, dx, x, p, t)
end

function caca2(dr, x, p)
end

# Start adjoint computations

λ0 = [1.0, 0.0, 0.0]
λ, μ = adjoint_sens(lorenz_fx_trans!, caca1!, caca2, caca2, traj, tvec, pvec, λ0)
println(λ)
function numerical_integration(x0, pvec)
    traj, tvec = rk4(lorenz_rhs!, x0, pvec, tspan, nsteps)
    val = 0.0
    for i=1:(nsteps-1)
        val += (tvec[i + 1] - tvec[i])*r(traj[:, i + 1], pvec)
    end
    return val
end

function terminal_condition(x0)
    traj, tvec = rk4(lorenz_rhs!, x0, pvec, tspan, nsteps)
    return traj[1,end]
end

function terminal_condition_scalar(x0)
    v = [1.0, 0.0, 0.0]
    traj, tvec = rk4(lorenz_rhs!, x0, pvec, tspan, nsteps)
    λ, μ = adjoint_sens(lorenz_fx_trans!, caca1!, caca2, caca2, traj, tvec, pvec, v)
    return λ[1]
end

fint_p(pvec) = numerical_integration(x0, pvec)
fint_x(x0) = numerical_integration(x0, pvec)

println(Calculus.gradient(terminal_condition,x0))

println("Test first-order adjoint (functional w.r.t x0, p)")
λ0 = [0.0, 0.0, 0.0]
λ, μ = adjoint_sens(lorenz_fx_trans!, lorenz_fp_trans!, rx, rp, traj, tvec, pvec, λ0)
println(λ)
println(Calculus.gradient(fint_x,x0))
println(μ)
println(Calculus.gradient(fint_p,pvec))


println("Test second-order adjoint (terminal condition w.r.t x0)")
λ0 = [1.0, 0.0, 0.0]
λ_traj = similar(traj)
adjoint_sens(lorenz_fx_trans!, caca1!, caca2, caca2, traj, tvec,
             pvec, λ0, λ_traj=λ_traj)

# compute Hv
v = [1.0, 0.0, 0.0]
forward_over_adjoint_sens(v, lorenz_fx!, lorenz_fp!, lorenz_fx_trans!,
                          lorenz_fp_trans!, lorenz_fxx!, traj, λ_traj, tvec, pvec)

v = [0.0, 1.0, 0.0]
forward_over_adjoint_sens(v, lorenz_fx!, lorenz_fp!, lorenz_fx_trans!,
                          lorenz_fp_trans!, lorenz_fxx!, traj, λ_traj, tvec, pvec)

v = [0.0, 0.0, 1.0]
forward_over_adjoint_sens(v, lorenz_fx!, lorenz_fp!, lorenz_fx_trans!,
                          lorenz_fp_trans!, lorenz_fxx!, traj, λ_traj, tvec, pvec)

# finite differences (just in case...)
H_calc = Calculus.hessian(terminal_condition,x0)
show(stdout, "text/plain", H_calc)
println("")
println("Test second-order adjoint (functional w.r.t x0, p)")

# compute Hv
λ0 = [0.0, 0.0, 0.0]
λ_traj = similar(traj)
adjoint_sens(lorenz_fx_trans!, lorenz_fp_trans!, rx, rp, traj, tvec,
             pvec, λ0, λ_traj=λ_traj)


v = [0.0, 0.0, 0.0]
δθ = [1.0, 0.0, 0.0]
σ1, τ1 = forward_over_adjoint_sens2(v, δθ, lorenz_fx!, lorenz_fp!,
                           lorenz_fx_trans!,
                           lorenz_fp_trans!,
                           lorenz_fxx!, lorenz_fxp!,
                           lorenz_fpx!, lorenz_fpp!,
                           rxx, rxp, rpx, rpp,
                           traj, λ_traj, tvec, pvec)

v = [0.0, 0.0, 0.0]
δθ = [0.0, 1.0, 0.0]
σ2, τ2 = forward_over_adjoint_sens2(v, δθ, lorenz_fx!, lorenz_fp!,
                           lorenz_fx_trans!,
                           lorenz_fp_trans!,
                           lorenz_fxx!, lorenz_fxp!,
                           lorenz_fpx!, lorenz_fpp!,
                           rxx, rxp, rpx, rpp,
                           traj, λ_traj, tvec, pvec)

v = [0.0, 0.0, 0.0]
δθ = [0.0, 0.0, 1.0]
σ3, τ3 = forward_over_adjoint_sens2(v, δθ, lorenz_fx!, lorenz_fp!,
                           lorenz_fx_trans!,
                           lorenz_fp_trans!,
                           lorenz_fxx!, lorenz_fxp!,
                           lorenz_fpx!, lorenz_fpp!,
                           rxx, rxp, rpx, rpp,
                           traj, λ_traj, tvec, pvec)

# finite differences
H_x0 = Calculus.hessian(fint_x,x0)
H_p = Calculus.hessian(fint_p,pvec)

println(σ1)
println(σ2)
println(σ3)
show(stdout, "text/plain", H_x0)
println("")
println(τ1)
println(τ2)
println(τ3)
show(stdout, "text/plain", H_p)
println("")

