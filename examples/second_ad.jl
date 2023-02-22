import CLSDyn
using Diff
struct CLSDiffTag end

struct Dynamics
    ps::PSystem
    f!::Function
    g::Function
    dg::Function
    x0::Vector{Float64}
    u0
    seed::Matrix{Float64}
    prob
    sol
end
export Dynamics

function Dynamics(case_file::String)
    x0, ps = CLSDyn.load_matpower(case_file)
    f!(du,u,p,t) = CLSDyn.classic_resfun!(du, u, ps)
    eval(
        :(ForwardDiff.:≺(
            ::Type{ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, ForwardDiff.Dual{CLSDiffTag, Float64, 1}}},
            ::Type{CLSDiffTag}
        ) = true)
    )
    eval(
        :(ForwardDiff.:≺(
            ::Type{CLSDiffTag},
            ::Type{ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, ForwardDiff.Dual{CLSDiffTag, Float64, 1}}}
        ) = false)
    )
    seed = zeros(length(x0), length(x0))
    for i in 1:length(x0)
        seed[i,i] = 1.0
    end
    g(u, p, t) = (u[1] .^ 2) ./ 2

    function dg(out, u, p, t)
        out[1] = u[1]
    end
    u0 = ForwardDiff.Dual{CLSDiffTag}.(copy(x0), seed)
    u0[1] = -0.005

    tspan=(0.0,5.0)
    prob = ODEProblem(f!,u0,tspan)
    return Dynamics(ps, f!, g, dg, x0, u0, seed, prob, nothing)

end

function solve!(prob::Dynamics, saveat)
    return solve(prob.prob; saveat=saveat)
end

function second_order_sensitivities(prob::Dynamics, sol)
    res = adjoint_sensitivities(
        sol,
        Vern9(),
        dgdu_continuous = prob.dg,
        g = prob.g,
        abstol = 1e-8,
        reltol = 1e-8
    )
    return res
end
