using NLPModels
import CLSDyn
include("lorenz.jl")

# IVP parameters
tspan=(0.0, 0.1)
pvec = [2.0, 1.0, 8.0/3.0]
x0 = [1.0, 1.0, 1.0]
nsteps = 1000

struct LorenzContext <: CLSDyn.SystemContext end
lc = LorenzContext()

# system dynamics
sys = CLSDyn.SystemDynamics(lc, pvec, lorenz_rhs!)
problem = CLSDyn.IVP(nsteps, x0, tspan, CLSDyn.RK(), sys)
CLSDyn.set_fx!(sys, lorenz_fx!)
CLSDyn.set_fp!(sys, lorenz_fp!)

# cost functional
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

function gradient(p, problem::CLSDyn.IVP, cost::CLSDyn.CostFunctional)
    problem.sys.pvec .= p
    traj, tvec = CLSDyn.solve(problem)
    λ0 = [0.0, 0.0, 0.0]
    λ, μ = CLSDyn.adjoint_sens(problem, cost, traj, tvec, λ0)
    return μ
end

function hessian(p, problem::CLSDyn.IVP, cost::CLSDyn.CostFunctional)
    problem.sys.pvec .= p
    pdim = length(p)
    hess = zeros(pdim, pdim)

    λ0 = zeros(pdim)
    λ_traj = similar(traj)
    CLSDyn.adjoint_sens(problem, cost, traj, tvec, λ0, λ_traj=λ_traj)
    δθ = zeros(pdim)
    v = zeros(pdim)

    for i=1:pdim
        δθ .= 0.0
        δθ[i] = 1.0
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
        hess[:,i] = τ
    end
    return hess
end

mutable struct DynamicNLP{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    ivp::CLSDyn.IVP
    cost::CLSDyn.CostFunctional
    counters::Counters
end

function DynamicNLP(
        ivp::CLSDyn.IVP,
        cost::CLSDyn.CostFunctional,
        lvar::AbstractArray,
        uvar::AbstractArray;
        nvar=length(ivp.sys.pvec), #change this to pvec
        ncon=0,
        counters=Counters()
    )
    meta = NLPModelMeta(
        nvar,
        ncon=ncon,
        name="Optimal control problem",
        lvar=lvar,
        uvar=uvar,
        minimize=true,
    )
    return DynamicNLP(meta, ivp, cost, counters)
end

function NLPModels.obj(nlp::DynamicNLP, x::AbstractVector)
    increment!(nlp, :neval_obj)
    # Temporary hack using quadrature
    nlp.ivp.sys.pvec .= x
    traj, tvec = CLSDyn.solve(nlp.ivp)
    val = 0.0
    nsteps = length(tvec)
    robj = nlp.cost.r!
    for i=1:(nsteps-1)
        val += (tvec[i + 1] - tvec[i])*robj(traj[:, i + 1], x, tvec[i + 1])
    end
    return val
end

function NLPModels.grad!(nlp::DynamicNLP, x::AbstractVector, gx::AbstractVector)
  increment!(nlp, :neval_grad)
  gx .= gradient(x, nlp.ivp, nlp.cost)
  return gx
end

using MadNLP
function MadNLP.jac_dense!(nlp::DynamicNLP, x::AbstractVector, jac::AbstractMatrix)
end
lvar = [1.0, 0.5, 1.0]
uvar = [3.0, 1.5, 3.0]
nlp = DynamicNLP(problem, cost, lvar, uvar)
p0 = [2.0, 1.0, 8.0/3.0]
gx = zeros(3)
println(NLPModels.obj(nlp, p0))
NLPModels.grad!(nlp, p0, gx)
println(gx)

# Verify gradient
using FiniteDiff
gx_fd = FiniteDiff.finite_difference_gradient(x -> NLPModels.obj(nlp, x), p0)
println(gx_fd)

ips = MadNLP.MadNLPSolver(nlp; print_level=MadNLP.INFO,
            kkt_system=MadNLP.DENSE_KKT_SYSTEM,
            hessian_approximation=MadNLP.DENSE_BFGS,
            linear_solver=LapackCPUSolver,)
MadNLP.solve!(ips)
