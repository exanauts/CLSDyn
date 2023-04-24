
struct LorenzContext <: CLSDyn.SystemContext end

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

@testset "Lorenz system" begin
    tspan=(0.0, 0.5)
    pvec = [2.0, 1.0, 8.0/3.0]
    x0 = [1.0, 1.0, 1.0]
    nsteps = 1000
    nx, np = length(x0), length(pvec)
    ϵ = 1e-7

    lc = LorenzContext()

    # system dynamics
    sys = CLSDyn.SystemDynamics(lc, pvec, lorenz_rhs!)
    problem = CLSDyn.IVP(nsteps, x0, tspan, CLSDyn.RK(), sys)
    CLSDyn.set_fx!(sys, lorenz_fx!)
    CLSDyn.set_fp!(sys, lorenz_fp!)
    CLSDyn.set_fx_trans!(sys, lorenz_fx_trans!)
    CLSDyn.set_fp_trans!(sys, lorenz_fp_trans!)
    CLSDyn.set_fxx!(sys, lorenz_fxx!)
    CLSDyn.set_fxp!(sys, lorenz_fxp!)
    CLSDyn.set_fpx!(sys, lorenz_fpx!)
    CLSDyn.set_fpp!(sys, lorenz_fpp!)

    # costs
    cost = CLSDyn.CostFunctional(sys, r, nothing)
    CLSDyn.set_rx!(cost, rx)
    CLSDyn.set_rp!(cost, rp)
    CLSDyn.set_rxx!(cost, rxx)
    CLSDyn.set_rpp!(cost, rpp)
    CLSDyn.set_rxp!(cost, rxp)
    CLSDyn.set_rpx!(cost, rpx)

    # Utils
    function numerical_integration(x0, pvec)
        traj, tvec = CLSDyn.rk4(sys.rhs!, x0, pvec, tspan, nsteps)
        val = 0.0
        for i=1:(nsteps-1)
            val += (tvec[i + 1] - tvec[i])*r(traj[:, i + 1], pvec, i, lc)
        end
        return val
    end

    function terminal_condition(x0)
        traj, tvec = CLSDyn.rk4(sys.rhs!, x0, pvec, tspan, nsteps)
        return traj[1,end]
    end

    fint_x(x0) = numerical_integration(x0, pvec)
    fint_p(pvec) = numerical_integration(x0, pvec)

    # Trajectory
    traj, tvec = CLSDyn.solve(problem)
    x0eps = [x0[1] + ϵ, x0[2], x0[3]]

    problem2 = CLSDyn.IVP(nsteps, x0eps, tspan, CLSDyn.RK(), sys)
    traj2, tvec2 = CLSDyn.solve(problem2)
    @test traj ≈ traj2 atol=1e-5
    @test size(traj) == (nx, nsteps)

    @testset "Forward sensitivities" begin
        dx0 = [1.0, 0.0, 0.0]
        tlm_traj = CLSDyn.tlm(problem, traj, tvec, dx0)
        @test size(tlm_traj) == (np, nsteps)
    end

    λ0 = [0.0, 0.0, 0.0]
    @testset "Adjoint sensitivities" begin
        λ, μ = CLSDyn.adjoint_sens(problem, cost, traj, tvec, λ0)
        @test size(λ) == (nx, )
        @test size(μ) == (np, )

        λ_fd = FiniteDiff.finite_difference_gradient(fint_x, x0)
        @test λ ≈ λ_fd atol=1e-4

        μ_fd = FiniteDiff.finite_difference_gradient(fint_p, pvec)
        @test μ ≈ μ_fd atol=1e-4
    end

    @testset "Second-order sensitivities" begin
        λ_traj = similar(traj)
        CLSDyn.adjoint_sens(problem, cost, traj, tvec, λ0; λ_traj=λ_traj)
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
        @test size(σ) == (nx, )
        @test size(τ) == (np, )
        H = FiniteDiff.finite_difference_hessian(fint_x, x0)
        @test_broken σ ≈ H * δθ atol=1e-4
        H = FiniteDiff.finite_difference_hessian(fint_p, pvec)
        @test τ ≈ H * δθ atol=1e-4
    end
end

