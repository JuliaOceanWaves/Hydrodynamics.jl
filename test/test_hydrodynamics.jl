using NBInclude
using Test

tol = 1e-6
gradient_fd = @nbinclude(joinpath(@__DIR__, "..", "examples", "power_performance.ipynb"))
# CI resolves different OrdinaryDiffEq major versions on Julia 1.10 and 1.12.
expected_gradient = if VERSION < v"1.12"
    [-1.15711e-17 1.47205e-17 2.54348e-18;
     1.8834778351590034 6.656485501004106 -0.4113564032511996;
     6.34733e-18 -1.41143e-17 -1.63797e-18]
else
    [-1.15711e-17 1.47205e-17 2.54348e-18;
     1.7879871478798326 6.167134816056571 -0.38005269042252054;
     6.34733e-18 -1.41143e-17 -1.63797e-18]
end

@test size(expected_gradient) == size(gradient_fd)
for idx in CartesianIndices(expected_gradient)
    @test isapprox(expected_gradient[idx], gradient_fd[idx]; atol = tol, rtol = tol)
end
