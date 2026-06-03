using Hydrodynamics
using Test

@testset "ramp function" begin
    @test ramp_function(1.0, 3.0, 0.5) == 0.0
    @test ramp_function(1.0, 3.0, 3.0) == 1.0
    @test 0.0 < ramp_function(1.0, 3.0, 2.0) < 1.0
end

@testset "excitation force and solver" begin
    omega = [1.0, 2.0]
    phase = [0.0, pi / 2]
    spectrum = [0.5, 0.2]
    d_frequency = 0.1
    wave = (omega, phase, spectrum, d_frequency, 0.0, 0.0)

    excitation_coeff = zeros(1, 1, 2, 2)
    excitation_coeff[1, 1, :, 1] = [1.0, 0.5]
    excitation_coeff[1, 1, :, 2] = [0.0, 0.25]

    force = calculate_excitation_force(0.25, excitation_coeff, wave)
    @test size(force) == (1,)
    @test isfinite(force[1])

    k = reshape([2.0], 1, 1)
    c = reshape([0.2], 1, 1)
    inverse_mass = reshape([1.0], 1, 1)
    constant_forces = [0.0]
    p = (k, c, inverse_mass, excitation_coeff, constant_forces, wave)

    ts = collect(0.0:0.1:0.3)
    sol = hydrodynamic_solver([0.0], [0.1], ts, p)
    @test sol.t == ts
    @test length(sol.x) == length(ts)
    @test length(sol.dx) == length(ts)
    @test all(v -> length(v) == 1 && isfinite(v[1]), sol.x)
    @test all(v -> length(v) == 1 && isfinite(v[1]), sol.dx)
end

@testset "Capytaine reader" begin
    hydro = Hydrodynamics.Bemio.read_capytaine(joinpath(
        @__DIR__, "..", "examples", "data", "rm3.nc"))
    @test length(hydro.w) > 0
    @test length(hydro.period) == length(hydro.w)
    @test size(hydro.ex, 1) > 0
    @test size(hydro.khs, 1) > 0
end
