using CLSDyn
using FiniteDiff
using ForwardDiff
using LazyArtifacts
const DATA_DIR = joinpath(artifact"ExaData", "ExaData")

case = "case3"
case_file = joinpath(DATA_DIR, "$(case).m")

x0, pvec, ps = CLSDyn.load_matpower(case_file)
f!(du,u,p,t) = CLSDyn.classic_resfun!(du, u, ps)

u0 = copy(x0)
u0[1] = -0.005
nsteps = 100
tspan=(0.0,5.0)

# system dynamics
sys = CLSDyn.SystemDynamics(ps, CLSDyn.classic_resfun2!)



# sensitivity of initial conditions w.r.t p
println(CLSDyn.get_x0(ps, pvec))
Jx0 = CLSDyn.full_x0_sens(ps, pvec)
println(Jx0)
f(p) = CLSDyn.get_x0(ps, p)
Jx0_fd = FiniteDiff.finite_difference_jacobian(f, pvec)
