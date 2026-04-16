using Test, SafeTestsets

@time @testset verbose = true "RepoTemplate.jl" begin
    @time @safetestset "Test X" begin
        include("test_X.jl")
    end
    @time @safetestset "Doc Tests" begin
        include("test_doctest.jl")
    end
end
