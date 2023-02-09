import CLSDyn
using DifferentialEquations
using Plots
using Calculus

x0, ps = CLSDyn.load_matpower("examples/case9.m")
f!(du,u,p,t) = CLSDyn.classic_resfun!(du, u, ps)

u0 = copy(x0)
u0[1] = -0.005

tspan=(0.0,5.0)

prob = ODEProblem(f!,u0,tspan)
sol = solve(prob, saveat=1.0/120)

# define a functional and its partials w.r.t u
g(u, p, t) = (u[1] .^ 2) ./ 2

function dg(out, u, p, t)
    out[1] = u[1]
end

using SciMLSensitivity
du0, dp = adjoint_sensitivities(
    sol,
    Vern9(),
    dgdu_continuous = dg,
    g = g,
    abstol = 1e-8,
    reltol = 1e-8
)

# verify gradient w.r.t initial conditions
using QuadGK
function objective(x)
    temp_prob = ODEProblem(f!,x,tspan)
    sol = solve(temp_prob,Vern9())
    res, err = quadgk((t)-> (sol(t)[1]^2)/2,0.0,5.0,atol=1e-14,rtol=1e-10)
    res
end

res2 = Calculus.gradient(objective,u0)
println(du0)
println(res2)
