function rk4_step!(xnew::AbstractArray,
                   f::Function,
                   xold::AbstractArray,
                   p::Vector,
                   t::Float64,
                   dt::Float64)
    # Note: this is a bit inefficient because I allocate the k's each time
    # i take a step. the best way might be to pass a data structure
    # where the k's are already allocated.
    ndim = size(xold, 1)
    k1 = zeros(ndim)
    k2 = zeros(ndim)
    k3 = zeros(ndim)
    k4 = zeros(ndim)
    f(k1, xold, p, t)
    f(k2, xold + (dt/2)*k1, p, t + 0.5*dt)
    f(k3, xold + (dt/2)*k2, p, t + 0.5*dt)
    f(k4, xold + dt*k3, p, t + dt)
    xnew .= xold .+ (dt/6)*k1 .+ (dt/3)*k2 .+ (dt/3)*k3 .+ (dt/6)*k4
end

function rk4(f::Function, x0::Vector, p::Vector, tspan::Tuple, nsteps::Int)
    """
        The function f is of the form
        f(dx, x, p, t)
    """
    ndim = size(x0, 1)
    traj = zeros((ndim, nsteps))
    traj[:,1] .= x0
    dt = (tspan[2] - tspan[1])/nsteps
    tvec = collect(range(start=tspan[1], stop=tspan[2], length=nsteps))
    for i=1:(nsteps - 1)
        t = tvec[i]
        rk4_step!(@view(traj[:, i + 1]), f, @view(traj[:, i]), p, t, dt)
    end
    return (traj, tvec)
end

function step!(xnew::AbstractArray,
               f::Function,
               xold::AbstractArray,
               p::Vector,
               t::Float64,
               dt::Float64,
               method::RK)
    ndim = size(xold, 1)
    k1 = method.k[1]
    k2 = method.k[2]
    k3 = method.k[3]
    k4 = method.k[4]
    f(k1, xold, p, t)
    f(k2, xold + (dt/2)*k1, p, t + 0.5*dt)
    f(k3, xold + (dt/2)*k2, p, t + 0.5*dt)
    f(k4, xold + dt*k3, p, t + dt)
    xnew .= xold .+ (dt/6)*k1 .+ (dt/3)*k2 .+ (dt/3)*k3 .+ (dt/6)*k4
end

function solve(ivp::IVP)
    # preallocate storage for k vectors
    preallocate!(ivp)
    ndim = size(ivp.x0, 1)
    traj = zeros((ndim, ivp.nsteps))
    traj[:,1] .= ivp.x0
    dt = (ivp.tspan[2] - ivp.tspan[1])/ivp.nsteps
    tvec = collect(range(start=ivp.tspan[1], stop=ivp.tspan[2], length=ivp.nsteps))
    for i=1:(ivp.nsteps - 1)
        t = tvec[i]
        step!(@view(traj[:, i + 1]), ivp.sys.rhs!,
              @view(traj[:, i]), ivp.sys.pvec, t, dt, ivp.method)
    end
    return (traj, tvec)
end
