"""
    classic_resfun!(dx::AbstractVector, x::AbstractVector, ps::PSystem)

Compute the residual function for the classic CLS model.
"""
function classic_resfun!(dx::AbstractVector, x::AbstractVector, ps::PSystem)
    ngen = ps.ngen
    emag = ps.emag
    pmec = ps.pmec
    yred = ps.yred
    H = ps.gen_inertia
    D = ps.gen_damping

    pelec = zeros(eltype(x), ngen)
    w = x[1:ngen]
    delta = x[ngen + 1:end]
    compute_pelec(pelec, emag, delta, yred)
    for i in 1:ngen
        dx[i] = (1.0/(2.0*H[i]))*(pmec[i] - pelec[i] - D[i]*w[i])
        dx[ngen + i] = _WS*w[i]
    end
end

function classic_resfun2!(
    dx::AbstractVector,
    x::AbstractVector,
    p::AbstractVector,
    t::Float64,
    ps::PSystem
)

    # NOTE: In our case, p is a dummy vector and parameters are updated at the beginning
    # of the simulation. This is because the way that parameters influence the r.h.s. is
    # not explicit but there are some preprocessing steps which, if implemented in
    # the r.h.s., would make it slower.
    ngen = ps.ngen
    yred = ps.yred
    H = ps.gen_inertia
    D = ps.gen_damping
    emag = ps.emag
    pmec = ps.pmec

    pelec = zeros(eltype(x), ngen)
    w = x[1:ngen]
    delta = x[ngen + 1:end]
    compute_pelec(pelec, emag, delta, yred)
    for i in 1:ngen
        dx[i] = (1.0/(2.0*H[i]))*(pmec[i] - pelec[i] - D[i]*w[i])
        dx[ngen + i] = _WS*w[i]
    end
end

function classic_jacobian!(
    J::AbstractMatrix,
    x::AbstractVector,
    p::AbstractVector,
    t::Float64,
    ps::PSystem
)
    """ Jacobian matrix of the classical model """
    ngen = ps.ngen
    H = ps.gen_inertia
    D = ps.gen_damping
    yred = ps.yred
    emag = ps.emag
    w = x[1:ngen]
    delta = x[ngen+1:end]

    for i in 1:ngen
        J[i, i] = -D[i]/(2*H[i])
        J[ngen + i, i] = _WS
        for j in 1:ngen
            if i != j
                J[i, ngen + j] = -emag[i]*emag[j]*(real(yred[i, j])*sin(delta[i] - delta[j]) -
                            imag(yred[i, j])*cos(delta[i] - delta[j]))
                J[i, ngen + j] = (1/(2*H[i]))*J[i, ngen + j]
                J[i, ngen + i] += emag[i]*emag[j]*(real(yred[i, j])*sin(delta[i] - delta[j]) -
                            imag(yred[i, j])*cos(delta[i] - delta[j]))
            end
        end
        J[i, ngen + i] = (1/(2*H[i]))*J[i, ngen + i]
    end
end

function classic_dfdp(
    dfdp::AbstractMatrix,
    x::AbstractVector,
    p::AbstractVector,
    t::Float64,
    ps::PSystem
)
    ngen = ps.ngen

end

"""
    compute_pelec(pelec::AbstractVector, vmag::AbstractVector, vang::AbstractVector,
                  yred::AbstractMatrix)

Compute the electrical power injected at each generator.
"""
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
        load_bus_tag = network_data["load"][string(i)]["load_bus"]
        load_bus = ybus_data.bus_to_idx[load_bus_tag]
        pd = network_data["load"][string(i)]["pd"]
        qd = network_data["load"][string(i)]["qd"]
        vm = sol["solution"]["bus"][string(load_bus_tag)]["vm"]

        yload = -pd/vm^2 + im*(qd/vm^2)
        YBB[load_bus, load_bus] -= yload
    end

    emag = zeros(Float64, ngen)
    eang = zeros(Float64, ngen)

    #     [ YAA   YAB ]
    #     [ YBA   YBB ]
    YAA = zeros(Complex, ngen, ngen)
    YAB = zeros(Complex, ngen, nbus)
    YBA = zeros(Complex, nbus, ngen)

    pvec = zeros(Float64, 4*ngen)

    for i=1:ngen
        gen_bus_tag = network_data["gen"][string(i)]["gen_bus"]
        gen_bus = ybus_data.bus_to_idx[gen_bus_tag]
        vm = sol["solution"]["bus"][string(gen_bus_tag)]["vm"]
        va = sol["solution"]["bus"][string(gen_bus_tag)]["va"]
        p_inj = sol["solution"]["gen"][string(i)]["pg"]
        q_inj = sol["solution"]["gen"][string(i)]["qg"]

        # store values in parameter vector
        #  [vm_1, vm_2 ... vm_n, va_1, va_2 ... va_n, 
        #       p_inj_1, p_inj_2 ... p_inj_n, q_inj_1, q_inj_2 ... q_inj_n]
        pvec[i] = vm
        pvec[ngen + i] = va
        pvec[2*ngen + i] = p_inj
        pvec[3*ngen + i] = q_inj

        # not sure how to get this one from MATPOWER.
        # we need to approximate generator reactance
        xdp = 1.0
        egen = (vm + p_inj*xdp/vm) + im*(q_inj*xdp/vm)

        emag[i] = abs(egen)
        eang[i] = angle(egen) + va

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
    delta = eang

    compute_pelec(pmec, emag, delta, yred)

    for i=1:ngen
        pmec[i] += gen_damping[i]*w[i]
    end

    ps = PSystem(nbus, ngen, nload, emag, yred, pmec, gen_inertia, gen_damping)
    x0 = vcat(w, delta)
    return x0, pvec, ps
end

function get_x0(ps::PSystem, pvec::AbstractVector)
    w = zeros(Float64, ps.ngen)
    delta = zeros(Float64, ps.ngen)

    for i=1:ps.ngen
        vm = pvec[i]
        va = pvec[ps.ngen + i]
        p_inj = pvec[2*ps.ngen + i]
        q_inj = pvec[3*ps.ngen + i]

        # TODO: figure out how to get xdp from matpower
        xdp = 1.0
        egen = (vm + p_inj*xdp/vm) + im*(q_inj*xdp/vm)
        delta[i] = angle(egen) + va
    end
    x0 = vcat(w, delta)
    return x0
end

function set_pvec!(ps::PSystem, pvec::AbstractVector)
    ngen = ps.ngen
    emag = zeros(Float64, ngen)
    eang = zeros(Float64, ngen)
    pmec = zeros(Float64, ngen)
    for i=1:ngen
        vm = pvec[i]
        va = pvec[ps.ngen + i]
        p_inj = pvec[2*ps.ngen + i]
        q_inj = pvec[3*ps.ngen + i]

        xdp = 1.0
        egen = (vm + p_inj*xdp/vm) + im*(q_inj*xdp/vm)

        emag[i] = abs(egen)
        eang[i] = angle(egen) + va
    end
    x0 = get_x0(ps, pvec)
    w = x0[1:ngen]
    delta = x0[ngen+1:end]
    compute_pelec(pmec, emag, delta, ps.yred)
    for i=1:ngen
        pmec[i] += ps.gen_damping[i]*w[i]
    end
    for i=1:ngen
        ps.emag[i] = emag[i]
        ps.pmec[i] = pmec[i]
    end
end

function full_x0_sens(ps::PSystem, pvec::AbstractVector)
    n = ps.ngen
    J = zeros(Float64, 2*n, 4*n)

    for i=1:ps.ngen
        vm = pvec[i]
        va = pvec[ps.ngen + i]
        p = pvec[2*ps.ngen + i]
        q = pvec[3*ps.ngen + i]
        xdp = 1.0

        # sensitivity of δ w.r.t vm
        res = -2*q*vm*xdp/(q^2*xdp^2 + (p*xdp + vm^2)^2)
        J[ps.ngen + i, i] = res

        # sensitivity of δ w.r.t va
        res = 1.0
        J[ps.ngen + i, ps.ngen + i] = res

        # sensitivity of δ w.r.t p
        res = -q*xdp^2/(q^2*xdp^2 + (p*xdp + vm^2)^2)
        J[ps.ngen + i, 2*ps.ngen + i] = res

        # sensitivity of δ w.r.t q
        res = xdp*(p*xdp + vm^2)/(q^2*xdp^2 + (p*xdp + vm^2)^2)
        J[ps.ngen + i, 3*ps.ngen + i] = res
    end
    return J
end
