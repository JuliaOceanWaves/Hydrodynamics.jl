using NBInclude
using Test

tol = 1e-6
@nbinclude(joinpath("..", "examples", "power_performance.ipynb"))
actual_gradient_loss_fd = gradient_loss_fd
expected_gradient_loss_fd = [-1.15711e-17 1.47205e-17 2.54348e-18;
                             -0.17120228103348425 -2.2054039047410607 0.03334019114760055;
                             6.34733e-18 -1.41143e-17 -1.63797e-18]

@test size(expected_gradient_loss_fd) == size(actual_gradient_loss_fd)
for idx in CartesianIndices(expected_gradient_loss_fd)
    @test isapprox(expected_gradient_loss_fd[idx], actual_gradient_loss_fd[idx]; atol = tol, rtol = tol)
end
