import CLSDyn
using DifferentialEquations
using Plots
using ForwardDiff
using LazyArtifacts

# Choose case to run
case = "case9"

# Load case file from Artifact
const DATA_DIR = joinpath(artifact"ExaData", "ExaData")
case_file = joinpath(DATA_DIR, "$(case).m")

x0, ps = CLSDyn.load_matpower(case_file)
f!(du,u,p,t) = CLSDyn.classic_resfun!(du, u, ps)

struct MyTag end
# To compute Hessian H[i,:] = d^2G/du_0^2
seed = zeros(length(x0))
i = 4
seed[i] = 1.0
u0 = ForwardDiff.Dual{MyTag}.(copy(x0), copy(seed))
u0[1] = -0.005

tspan=(0.0,5.0)

ForwardDiff.:≺(::Type{ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, ForwardDiff.Dual{MyTag, Float64, 1}}}, ::Type{MyTag}) = true
ForwardDiff.:≺(::Type{MyTag}, ::Type{ForwardDiff.Tag{DiffEqBase.OrdinaryDiffEqTag, ForwardDiff.Dual{MyTag, Float64, 1}}}) = false
prob = ODEProblem(f!,u0,tspan)
sol = solve(prob, saveat=1.0/120)

# define a functional and its partials w.r.t u
g(u, p, t) = (u[1] .^ 2) ./ 2

function dg(out, u, p, t)
    out[1] = u[1]
end

using SciMLSensitivity
res = adjoint_sensitivities(
    sol,
    Vern9(),
    dgdu_continuous = dg,
    g = g,
    abstol = 1e-8,
    reltol = 1e-8
)
