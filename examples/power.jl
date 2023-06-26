using CLSDyn
using NLPModels
using MadNLP
using FiniteDiff
using ForwardDiff
using LazyArtifacts
const DATA_DIR = joinpath(artifact"ExaData", "ExaData")

case = "case9"
case_file = joinpath(DATA_DIR, "$(case).m")

x0, pvec, ps = CLSDyn.load_matpower(case_file)
f!(du,u,p,t) = CLSDyn.classic_resfun!(du, u, ps)

# system dynamics
sys = CLSDyn.SystemDynamics(ps, pvec, CLSDyn.classic_resfun2!)

# sensitivity of initial conditions w.r.t p
Jx0 = CLSDyn.full_x0_sens(ps, pvec)
f(p) = CLSDyn.get_x0(ps, p)
Jx0_fd = FiniteDiff.finite_difference_jacobian(f, pvec)
println(Jx0)
println(Jx0_fd)

# sensitivity of r.h.s. w.r.t. x

function rhs_x(x)
    dx = zeros(Float64, length(x))
    sys.rhs!(dx, x, pvec, 0.0)
    return dx
end

Jrhs_x = zeros(Float64, length(x0), length(x0))
CLSDyn.classic_jacobian!(Jrhs_x, x0, pvec, 0.0, ps)
Jrhs_x_fd = FiniteDiff.finite_difference_jacobian(rhs_x, x0)
println(Jrhs_x)
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

# build system
u0 = copy(x0)
u0[1] = -0.01
nsteps = 100
tspan=(0.0,5.0)
problem = CLSDyn.IVP(nsteps, u0, tspan, CLSDyn.RK(), sys)

function psys_fx_trans!(df, dx, x, p, t, ctx)
    psys = ctx
    Jrhs_x = zeros(Float64, length(x), length(x))
    CLSDyn.classic_jacobian!(Jrhs_x, x, p, t, psys)
    df .= transpose(Jrhs_x) * dx
end

function psys_fp_trans!(df, dx, x, p, t, ctx)
    psys = ctx
    Jrhs_p = zeros(Float64, length(x), length(p))
    Jrhs_p_fd = FiniteDiff.finite_difference_jacobian(rhs_p, pvec)
    df .= transpose(Jrhs_p) * dx
end

CLSDyn.set_fx_trans!(sys, psys_fx_trans!)
CLSDyn.set_fp_trans!(sys, psys_fp_trans!)

# cost functional
r(x, p, t, ctx) = (x[1] .^ 2) ./ 2
function rx(dr, x, p, t, ctx)
    dr[1] = x[1]
end
function rp(dr, x, p, t, ctx)
end

cost = CLSDyn.CostFunctional(sys, r, nothing)
CLSDyn.set_rx!(cost, rx)
CLSDyn.set_rp!(cost, rp)

function objective_power(p, problem::CLSDyn.IVP, cost::CLSDyn.CostFunctional)
    problem.sys.pvec .= p
    CLSDyn.set_pvec!(problem.sys.ctx, p)
    traj, tvec = CLSDyn.solve(problem)
    val = 0.0
    nsteps = length(tvec)
    robj = cost.r!
    for i=1:(nsteps-1)
        val += (tvec[i + 1] - tvec[i])*robj(traj[:, i + 1], p, tvec[i + 1])
    end
    return val
end

function gradient_power(p, problem::CLSDyn.IVP, cost::CLSDyn.CostFunctional)
    problem.sys.pvec .= p
    ∇Ψ = zeros(length(p))
    
    # need to find a better way to do this
    CLSDyn.set_pvec!(problem.sys.ctx, p)

    traj, tvec = CLSDyn.solve(problem)
    λ0 = zeros(size(traj, 1))
    λ, μ = CLSDyn.adjoint_sens(problem, cost, traj, tvec, λ0)
    ∇Ψ .= μ
    Jx0 = CLSDyn.full_x0_sens(problem.sys.ctx, p)
    ∇Ψ += transpose(Jx0) * λ
    return ∇Ψ
end

obj = objective_power(pvec, problem, cost)
println(obj)

obj_wrapper(p) = objective_power(p, problem, cost)
gx_fd = FiniteDiff.finite_difference_gradient(obj_wrapper, pvec)
println(gx_fd)

gx = zeros(Float64, length(pvec))
gx .= gradient_power(pvec, problem, cost)

# construct NLP struct
lvar = pvec - 0.5*pvec
uvar = pvec + 0.5*pvec
nlp = CLSDyn.DynamicNLP(problem, cost, lvar, uvar)

# NLP function abstractions
function NLPModels.obj(nlp::CLSDyn.DynamicNLP, x::AbstractVector)
    increment!(nlp, :neval_obj)
    return objective_power(x, nlp.ivp, nlp.cost)
end

function NLPModels.grad!(nlp::CLSDyn.DynamicNLP, x::AbstractVector, gx::AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= -gradient_power(x, nlp.ivp, nlp.cost)
  return gx
end

function MadNLP.jac_dense!(nlp::CLSDyn.DynamicNLP, x::AbstractVector, jac::AbstractMatrix)
end


# set initial guess
p0 = pvec

# solve problem
ips = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.INFO,
            kkt_system=MadNLP.DENSE_KKT_SYSTEM,
            hessian_approximation=MadNLP.DENSE_BFGS,
            linear_solver=LapackCPUSolver,
            max_iter=10)
MadNLP.solve!(ips)
