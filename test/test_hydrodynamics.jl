using NBInclude
using Test

tol = 1e-6
gradient_loss_fd = @nbinclude(joinpath(@__DIR__, "..", "examples", "power_performance.ipynb"))
last_result = [-1.15711e-17 1.47205e-17 2.54348e-18;
               -0.17120228103348425 -2.2054039047410607 0.03334019114760055;
               6.34733e-18 -1.41143e-17 -1.63797e-18]

@test size(last_result) == size(gradient_loss_fd)
for idx in CartesianIndices(last_result)
    @test isapprox(last_result[idx], gradient_loss_fd[idx]; atol = tol, rtol = tol)
end
