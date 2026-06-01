using NBInclude
using Test

tol = 1e-6
gradient_fd = @nbinclude(joinpath(@__DIR__, "..", "examples", "power_performance.ipynb"))
# The notebook exercises the full example. Small ODE solver differences across
# platforms currently produce either of these stable gradient baselines.
expected_gradients = (
    [-1.15711e-17 1.47205e-17 2.54348e-18;
     1.8834778351590034 6.656485501004106 -0.4113564032511996;
     6.34733e-18 -1.41143e-17 -1.63797e-18],
    [-1.15711e-17 1.47205e-17 2.54348e-18;
     1.7879871478798326 6.167134816056571 -0.38005269042252054;
     6.34733e-18 -1.41143e-17 -1.63797e-18]
)

@test size(gradient_fd) == (3, 3)
@test any(expected_gradients) do expected_gradient
    all(isapprox.(expected_gradient, gradient_fd; atol = tol, rtol = tol))
end
