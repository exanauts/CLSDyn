
@testset "IVP problem" begin
    case = "case9"
    case_file = joinpath(DATA_DIR, "$(case).m")

    x0, ps = CLSDyn.load_matpower(case_file)
    f!(du,u,p,t) = CLSDyn.classic_resfun!(du, u, ps)

    u0 = copy(x0)
    u0[1] = -0.005
    nsteps = 100
    tspan=(0.0,5.0)

    # RK4 from CLSDyn
    pvec = zeros(2) #dummy
    traj, tvec = CLSDyn.rk4(f!, u0, pvec, tspan, nsteps)

    # system dynamics
    sys = CLSDyn.SystemDynamics(ps, CLSDyn.classic_resfun2!)
    traj1, tvec1 = CLSDyn.rk4(sys.rhs!, u0, sys.pvec, tspan, nsteps)

    # IVP
    problem = CLSDyn.IVP(nsteps, u0, tspan, CLSDyn.RK(), sys)
    traj2, tvec2 = CLSDyn.solve(problem)

    @test traj ≈ traj1 ≈ traj2
    @test tvec ≈ tvec1 ≈ tvec2
end
