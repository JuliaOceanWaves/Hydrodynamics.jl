using Hydrodynamics
using Test
using LinearAlgebra

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

    k = 2.0 * ones(1, 1)
    c = 0.2 * ones(1, 1)
    inverse_mass = 1.0 * ones(1, 1)
    constant_forces = [0.0]

    pto = ([0.0], 0*k, 0*c)
    mooring = ([0.0], 0*k, 0*c)

    hydro = (k, c, excitation_coeff, constant_forces, wave)
    p = (inverse_mass, hydro, pto, mooring)

    ts = collect(0.0:0.1:0.3)
    sol = hydrodynamic_solver([0.0; 0.1], ts, p, method = :point)
    @test sol.t == ts
    @test length(sol[1, :]) == length(ts)
    @test length(sol[2, :]) == length(ts)
    @test all(isfinite.(sol[1, :]))
    @test all(isfinite.(sol[2, :]))
end

@testset "Capytaine reader" begin
    hydro = Hydrodynamics.Bemio.read_capytaine(joinpath(
        @__DIR__, "..", "examples", "data", "rm3.nc"))
    @test length(hydro.w) > 0
    @test length(hydro.period) == length(hydro.w)
    @test size(hydro.ex, 1) > 0
    @test size(hydro.khs, 1) > 0
end
