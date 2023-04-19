import CLSDyn
using LazyArtifacts


# function definition
function lorenz_rhs!(dx, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    dx[1] = σ*(x[2] - x[1])
    dx[2] = x[1]*(ρ - x[3]) - x[2]
    dx[3] = x[1]*x[2] - β*x[3]
end

function lorenz_fx!(df, dx, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -σ*dx[1] + σ*dx[2]
    df[2] = (ρ - x[3])*dx[1] - dx[2] - x[1]*dx[3]
    df[3] = x[2]*dx[1] + x[1]*dx[2] - β*dx[3]
end

function lorenz_fx_trans!(df, dx, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -σ*dx[1] + (ρ - x[3])*dx[2] + x[2]*dx[3]
    df[2] = σ*dx[1] - dx[2] + x[1]*dx[3]
    df[3] = -x[1]*dx[2] - β*dx[3]
end

function lorenz_fp!(df, dp, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = (x[2] - x[1])*dp[1]
    df[2] = x[1]*dp[2]
    df[3] = -x[3]*dp[3]
end

function lorenz_fp_trans!(df, dp, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = (x[2] - x[1])*dp[1]
    df[2] = x[1]*dp[2]
    df[3] = -x[3]*dp[3]
end

function lorenz_fxx!(df, u, z, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = u[2]*z[3] - z[2]*u[3]
    df[2] = u[1]*z[3]
    df[3] = -u[1]*z[2]
end

function lorenz_fxp!(df, u, z, x, p, t, ctx)
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

function lorenz_fpx!(df, u, z, x, p, t, ctx)
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

function lorenz_fpp!(df, u, z, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = 0.0
    df[2] = 0.0
    df[3] = 0.0
end

# IVP parameters
tspan=(0.0, 0.5)
pvec = [2.0, 1.0, 8.0/3.0]
x0 = [1.0, 1.0, 1.0]
nsteps = 1000
ϵ = 1e-7

struct LorenzContext <: CLSDyn.SystemContext end
lc = LorenzContext()


# system dynamics
sys = CLSDyn.SystemDynamics(lc, pvec, lorenz_rhs!)
problem = CLSDyn.IVP(nsteps, x0, tspan, CLSDyn.RK(), sys)
CLSDyn.set_fx!(sys, lorenz_fx!)
CLSDyn.set_fp!(sys, lorenz_fp!)


println("Forward sensitivity")
traj, tvec = CLSDyn.solve(problem)
x0eps = [x0[1] + ϵ, x0[2], x0[3]]

problem2 = CLSDyn.IVP(nsteps, x0eps, tspan, CLSDyn.RK(), sys)
traj2, tvec2 = CLSDyn.solve(problem2)
println((traj2[:,end]-traj[:,end])/ϵ)

dx0 = [1.0, 0.0, 0.0]
tlm_traj = CLSDyn.tlm(problem, traj, tvec, dx0)
println(tlm_traj[:,end])

# adjoint sensitivity
r(x, p, t, ctx) = (x[1] .^ 2) ./ 2
function rx(dr, x, p, t, ctx)
    dr[1] = x[1]
end
function rp(dr, x, p, t, ctx)
end

function rxx(dr, dx, x, p, t, ctx)
    dr[1] = dx[1]
end

function rpp(dr, dx, x, p, t, ctx)
end

function rxp(dr, dx, x, p, t, ctx)
end

function rpx(dr, dx, x, p, t, ctx)
end

println("Adjoint sensitivity")
cost = CLSDyn.CostFunctional(sys, r, nothing)
CLSDyn.set_rx!(cost, rx)
CLSDyn.set_rp!(cost, rp)
CLSDyn.set_rxx!(cost, rxx)
CLSDyn.set_rpp!(cost, rpp)
CLSDyn.set_rxp!(cost, rxp)
CLSDyn.set_rpx!(cost, rpx)

CLSDyn.set_fx!(sys, lorenz_fx!)
CLSDyn.set_fp!(sys, lorenz_fp!)
CLSDyn.set_fx_trans!(sys, lorenz_fx_trans!)
CLSDyn.set_fp_trans!(sys, lorenz_fp_trans!)
CLSDyn.set_fxx!(sys, lorenz_fxx!)
CLSDyn.set_fxp!(sys, lorenz_fxp!)
CLSDyn.set_fpx!(sys, lorenz_fpx!)
CLSDyn.set_fpp!(sys, lorenz_fpp!)

λ0 = [0.0, 0.0, 0.0]
λ, μ = CLSDyn.adjoint_sens(problem, cost, traj, tvec, λ0)
println(λ)

println("Second-order adjoint sensitivity")
λ0 = [0.0, 0.0, 0.0]
λ_traj = similar(traj)
CLSDyn.adjoint_sens(problem, cost, traj, tvec, λ0, λ_traj=λ_traj)

v = [0.0, 0.0, 0.0]
δθ = [1.0, 0.0, 0.0]

σ, τ = CLSDyn.forward_over_adjoint(
    problem,
    cost,
    traj,
    λ_traj,
    tvec,
    pvec,
    v,
    δθ
)
println(τ)
