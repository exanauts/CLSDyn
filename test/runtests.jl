
using Test

using FiniteDiff
using LazyArtifacts

using CLSDyn

const DATA_DIR = joinpath(artifact"ExaData", "ExaData")

@testset "CLSDyn" begin
    include("lorenz.jl")
    include("power_systems.jl")
end

