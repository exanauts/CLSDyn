"""
        tlm(fx, fp, dx, traj, tvec, p[, dp])

Computes the Tangent Linear Model. If dp = nothing, computes the TLM with respect
to given perturbation direction δx0. Otherwise, computes TLM with respect to
perturbation of parameter perturbation dp.

"""
function tlm(
    fx::Function,
    fp::Function,
    δx0::AbstractArray,
    traj::AbstractArray,
    tvec::AbstractArray,
    p::AbstractArray,
    method::ODEMethod;
    dp::Union{AbstractArray, Nothing}=nothing,
)

    nsteps = size(tvec, 1)
    tlm_traj = similar(traj)
    tlm_traj[:,1] .= δx0

    size_x = size(traj, 1)
    size_p = size(p, 1)

    buffer_fx = zeros(size_x)
    buffer_fp = zeros(size_p)

    # we create a closure such tat we can use a time stepper that takes a function
    #    f(dx, x, p, t)
    # but instead of changing the parameter, we change "x" at each step, that is given
    # by the solution of the forward model.

    function tlm_rhs!(df, dx, x, t)
        fx(buffer_fx, dx, x, p, t)
        if dp != nothing
            fp(buffer_fp, dp, x, p, t)
        end
        df .= buffer_fx .+ buffer_fp
    end

    nsteps = size(tvec, 1)
    for i = 1:(nsteps - 1)
        t = tvec[i]
        x = @view(traj[:, i])
        dt = tvec[i + 1] - tvec[i]
        step!(@view(tlm_traj[:, i + 1]), tlm_rhs!, @view(tlm_traj[:, i]),
              x, t, dt, method)
    end

    return tlm_traj
end

"""
    tlm(ivp, traj, tvec, δx0[, dp])

Compute TLM with respect to given perturbation direction δx0. If dp = nothing,

Design Note: I am not sure if passing an IVP object is the best way to do this.
On one hand, the TLM might be related to the IVP and might re-use integration
horizon, initial condition, etc. On the other hand, the TLM can be integrated
with different methods so it might be better to pass a SystemDynamics object
and a ODEMethod object.
"""
function tlm(
    ivp::IVP,
    traj::AbstractArray,
    tvec::AbstractArray,
    δx0::AbstractArray;
    dp::Union{AbstractArray, Nothing}=nothing,
)
    @assert typeof(ivp.sys.fx!) != Nothing
    if dp != nothing
        @assert typeof(ivp.sys.fp!) != Nothing
    end

    return tlm(ivp.sys.fx!, ivp.sys.fp!,
               δx0, traj, tvec, ivp.sys.pvec, ivp.method; dp=dp)
end


"""
    adjoint_sens

Compute adjoint sensitivities with respect to functional.
"""
function adjoint_sens(
    fx_trans::Function,
    fp_trans::Function,
    rx::Function,
    rp::Function,
    traj::AbstractArray,
    tvec::AbstractArray,
    p::AbstractArray,
    λ0::AbstractArray,
    method::ODEMethod;
    λ_traj::Union{AbstractArray, Nothing}=nothing,
)

    # obtain dimensions
    xdim = size(traj, 1)
    pdim = size(p, 1)
    @assert xdim == size(λ0, 1)
    nsteps = size(tvec, 1)

    # check if we need to store trajectory of λ
    if λ_traj == nothing
        store_trajectory = false
    else
        store_trajectory = true
        @assert size(λ_traj, 1) == size(traj, 1)
        @assert size(λ_traj, 2) == size(traj, 2)
        λ_traj[:, end] .= λ0
    end

    μ0 = zeros(pdim)
    u0 = vcat(λ0, μ0)
    u = similar(u0)
    u .= u0
    unext = similar(u0)
    drx = zeros(xdim)
    drp = zeros(pdim)
    
    function adj_sys!(du, u, x, t)
        λ = @view(u[1:xdim])
        μ = @view(u[xdim + 1:end])
        dλ = @view(du[1:xdim])
        dμ = @view(du[xdim + 1:end])
        
        fx_trans(dλ, λ, x, p, t)
        rx(drx, x, p, t)
        dλ .+= drx
        
        fp_trans(dμ, λ, x, p, t)
        rp(drp, x, p, t)
        dμ .+= drp
    end
    
    nsteps = size(tvec, 1)
    for i = reverse(2:nsteps)
        t = tvec[i]
        x = @view(traj[:, i])
        dt = tvec[i] - tvec[i - 1]
        step!(unext, adj_sys!, u, x, t, dt, method)
        u .= unext
        if store_trajectory == true
            λ_traj[:, i - 1] .= unext[1:xdim]
        end
    end
    # this is dg/dx0
    λ = u[1:xdim]
    # this is dg/dp
    μ = u[xdim + 1:end]
    return (λ, μ)
end

function adjoint_sens(
        ivp::IVP,
        cost::CostFunctional,
        traj::AbstractArray,
        tvec::AbstractArray,
        λ0::AbstractArray;
        λ_traj::Union{AbstractArray, Nothing}=nothing,
    )
    sys = ivp.sys
    @assert typeof(ivp.sys.fx_transpose!) != Nothing
    if typeof(ivp.sys.fp_transpose!) != Nothing
        @assert typeof(cost.rp!) != Nothing
    end
    
    # NOTE: this is a bad outcome of the current design. When associating
    # IVP to a SYS, the preallocation of the IVP depends on the underlying SYS
    # However, in the case of the adjoint sensitivities, we need to preallocate
    # the IVP with the size of the trajectory and the size of the parameter vec.
    # This call to preallocate! is a hack to make this work, but need to re-think.
    preallocate!(ivp, size(traj, 1) + size(sys.pvec, 1))

    # NOTE: need to adress case where fp and rp are not defined
    sol = adjoint_sens(
            sys.fx_transpose!, sys.fp_transpose!,
            cost.rx!, cost.rp!,
            traj, tvec, sys.pvec, λ0, ivp.method;
            λ_traj=λ_traj,
        )
    return sol
end
