using Test, SafeTestsets

@time @testset verbose=true "Hydrodynamics.jl" begin
    @time @safetestset "Differentiability" begin
        include("test_differentiability.jl")
    end
    @time @safetestset "Test Hydrodynamics" begin
        include("test_hydrodynamics.jl")
    end
    # @time @safetestset "Doc Tests" begin
    #     include("test_doctest.jl")
    # end
end
