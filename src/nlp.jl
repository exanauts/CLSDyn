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

function gradient(p, problem::CLSDyn.IVP, cost::CLSDyn.CostFunctional)
    problem.sys.pvec .= p
    ∇Ψ = zeros(length(p))

    traj, tvec = CLSDyn.solve(problem)
    λ0 = zeros(size(traj, 1))
    λ, μ = CLSDyn.adjoint_sens(problem, cost, traj, tvec, λ0)
    ∇Ψ .= μ
    return ∇Ψ
end

function hessian(p, problem::CLSDyn.IVP, cost::CLSDyn.CostFunctional)
    problem.sys.pvec .= p
    pdim = length(p)
    hess = zeros(pdim, pdim)

    λ0 = zeros(size(traj, 1))
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
