module CLSDyn
using DifferentialEquations
using Plots
using ForwardDiff
using SciMLSensitivity

import PowerModels

const _WS = 1.0

struct PSystem
    nbus::Int
    ngen::Int
    nload::Int
    # parameters
    vmag::Vector{Float64}
    yred::Matrix{ComplexF64}
    pmec::Vector{Float64}
    gen_inertia::Vector{Float64}
    gen_damping::Vector{Float64}
end

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

function classic_resfun!(dx, x, ps::PSystem)
    ngen = ps.ngen
    vmag = ps.vmag
    pmec = ps.pmec
    yred = ps.yred
    H = ps.gen_inertia
    D = ps.gen_damping

    pelec = zeros(eltype(x), ngen)
    w = x[1:ngen]
    delta = x[ngen + 1:end]
    compute_pelec(pelec, vmag, delta, yred)
    for i in 1:ngen
        dx[i] = (1.0/(2.0*H[i]))*(pmec[i] - pelec[i] - D[i]*w[i])
        dx[ngen + i] = _WS*w[i]
    end
end

function compute_pelec(pelec, vmag, vang, yred)
    nbus = size(pelec, 1)
    pelec .= 0.0
    for i in 1:nbus
        for j in 1:nbus
            if i == j
                pelec[i] += vmag[i]*vmag[i]*real(yred[i, i])
            else
                pelec[i] += vmag[i]*vmag[j]*(imag(yred[i, j])*sin(vang[i] - vang[j]) +
                        real(yred[i, j])*cos(vang[i] - vang[j]))
            end
        end
    end
end

"""
    load_matpower(casefile::String)

Load system from matpower. Returns initial equilibrium (obtained via
power flow) and system struct.
"""
function load_matpower(casefile)

    PowerModels.silence()
    network_data = PowerModels.parse_file(casefile)
    sol = PowerModels.compute_ac_pf(network_data)
    ybus_data = PowerModels.calc_admittance_matrix(network_data)
    YBB = ybus_data.matrix

    # number of elements
    nbus = length(network_data["bus"])
    ngen = length(network_data["gen"])
    nload = length(network_data["load"])

    # add loads to admittance matrix
    for i=1:nload
        load_bus = network_data["load"][string(i)]["load_bus"]
        pd = network_data["load"][string(i)]["pd"]
        qd = network_data["load"][string(i)]["qd"]
        vm = sol["solution"]["bus"][string(load_bus)]["vm"]

        yload = -pd/vm^2 + im*(qd/vm^2)
        YBB[load_bus, load_bus] -= yload
    end

    vmag = zeros(Float64, ngen)
    vang = zeros(Float64, ngen)

    #     [ YAA   YAB ]
    #     [ YBA   YBB ]
    YAA = zeros(Complex, ngen, ngen)
    YAB = zeros(Complex, ngen, nbus)
    YBA = zeros(Complex, nbus, ngen)

    for i=1:ngen
        gen_bus = network_data["gen"][string(i)]["gen_bus"]
        vm = sol["solution"]["bus"][string(gen_bus)]["vm"]
        va = sol["solution"]["bus"][string(gen_bus)]["va"]
        p_inj = sol["solution"]["gen"][string(i)]["pg"]
        q_inj = sol["solution"]["gen"][string(i)]["qg"]

        # not sure how to get this one from MATPOWER.
        # we need to approximate generator reactance
        xdp = 1.0
        egen = (vm + p_inj*xdp/vm) + im*(q_inj*xdp/vm)

        vmag[i] = abs(egen)
        vang[i] = angle(egen) + va

        yint = 1/(im*xdp)
        YAA[i, i] += yint
        YAB[i, gen_bus] -= yint
        YBA[gen_bus, i] -= yint
        YBB[gen_bus, gen_bus] += yint
    end

    yred = YAA - YAB*(YBB\YBA)

    # here we would read the dynamics. For now we just make up
    # inertia and damping values

    gen_inertia = ones(Float64, ngen)
    gen_damping = zeros(Float64, ngen)
    pmec = zeros(Float64, ngen)
    w = zeros(Float64, ngen) # ws = 1
    delta = vang

    compute_pelec(pmec, vmag, vang, yred)

    for i=1:ngen
        pmec[i] += gen_damping[i]*w[i]
    end

    ps = PSystem(nbus, ngen, nload, vmag, yred, pmec, gen_inertia, gen_damping)
    x0 = vcat(w, delta)
    return x0, ps
end

end # module CLSDyn
