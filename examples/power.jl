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
sys = CLSDyn.SystemDynamics(ps, pvec, CLSDyn.classic_resfun2!)

# sensitivity of initial conditions w.r.t p
Jx0 = CLSDyn.full_x0_sens(ps, pvec)
f(p) = CLSDyn.get_x0(ps, p)
Jx0_fd = FiniteDiff.finite_difference_jacobian(f, pvec)

# sensitivity of r.h.s. w.r.t. x

function rhs_x(x)
    dx = zeros(Float64, length(x))
    sys.rhs!(dx, x, pvec, 0.0)
    return dx
end

Jrhs_x = zeros(Float64, length(x0), length(x0))
CLSDyn.classic_jacobian!(Jrhs_x, x0, pvec, 0.0, ps)
println(Jrhs_x)

Jrhs_x_fd = FiniteDiff.finite_difference_jacobian(rhs_x, x0)
println(Jrhs_x_fd)

# sensitivity of r.h.s. w.r.t. p
function rhs_p(p)
    dx = zeros(Float64, length(x0))
    ps_temp = deepcopy(ps)
    CLSDyn.set_pvec!(ps_temp, p)
    sys_temp = CLSDyn.SystemDynamics(ps_temp, p, CLSDyn.classic_resfun2!)
    sys_temp.rhs!(dx, x0, p, 0.0)
    return dx
end

Jrhs_p_fd = FiniteDiff.finite_difference_jacobian(rhs_p, pvec)
