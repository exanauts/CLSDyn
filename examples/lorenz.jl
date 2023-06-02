# function definition
function lorenz_rhs!(dx, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    dx[1] = σ*(x[2] - x[1])
    dx[2] = x[1]*(ρ - x[3]) - x[2]
    dx[3] = x[1]*x[2] - β*x[3]
end

function lorenz_fx!(df, dx, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -σ*dx[1] + σ*dx[2]
    df[2] = (ρ - x[3])*dx[1] - dx[2] - x[1]*dx[3]
    df[3] = x[2]*dx[1] + x[1]*dx[2] - β*dx[3]
end

function lorenz_fx_trans!(df, dx, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -σ*dx[1] + (ρ - x[3])*dx[2] + x[2]*dx[3]
    df[2] = σ*dx[1] - dx[2] + x[1]*dx[3]
    df[3] = -x[1]*dx[2] - β*dx[3]
end

function lorenz_fp!(df, dp, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = (x[2] - x[1])*dp[1]
    df[2] = x[1]*dp[2]
    df[3] = -x[3]*dp[3]
end

function lorenz_fp_trans!(df, dp, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = (x[2] - x[1])*dp[1]
    df[2] = x[1]*dp[2]
    df[3] = -x[3]*dp[3]
end

function lorenz_fxx!(df, u, z, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = u[2]*z[3] - z[2]*u[3]
    df[2] = u[1]*z[3]
    df[3] = -u[1]*z[2]
end

function lorenz_fxp!(df, u, z, x, p, t, ctx)
    # Note: this is the transpose of the above function
    #   u = δθ
    #   z = λ
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -z[1]*u[1] + z[2]*u[2]
    df[2] = z[1]*u[1]
    df[3] = -z[3]*u[3]
end

function lorenz_fpx!(df, u, z, x, p, t, ctx)
    # Note: this is the transpose of the above function
    #   u = δx
    #   z = λ
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = -z[1]*u[1] + z[1]*u[2]
    df[2] = z[2]*u[1]
    df[3] = -z[3]*u[3]
end

function lorenz_fpp!(df, u, z, x, p, t, ctx)
    σ = p[1]
    ρ = p[2]
    β = p[3]

    df[1] = 0.0
    df[2] = 0.0
    df[3] = 0.0
end
