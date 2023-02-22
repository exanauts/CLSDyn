import CLSDyn
using DifferentialEquations
using Plots
using LazyArtifacts

# Choose case to run
case = "case9"

# Load case file from Artifact
const DATA_DIR = joinpath(artifact"ExaData", "ExaData")
case_file = joinpath(DATA_DIR, "$(case).m")

x0, ps = CLSDyn.load_matpower(case_file)
f!(du,u,p,t) = CLSDyn.classic_resfun!(du, u, ps)

u0 = copy(x0)
u0[1] = -0.005

tspan=(0.0,5.0)

prob = ODEProblem(f!,u0,tspan)
sol = solve(prob, saveat=0.1)

# RK4 from CLSDyn
pvec = zeros(2) #dummy
traj, tvec = CLSDyn.rk4(f!, u0, pvec, tspan, size(sol.t, 1))

idx_var = 1
plt_handle = plot(sol,idxs=(idx_var))
plot!(plt_handle, tvec, traj[idx_var,:])
