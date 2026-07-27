using Hydrodynamics
using Test
using NBInclude
using Unitful
using DimensionfulAngles: radᵃ as rad

@testset "ramp function" begin
    @test ramp_function(1.0, 3.0, 0.5) == 0.0
    @test ramp_function(1.0, 3.0, 1.0) == 0.0
    @test 0.0 < ramp_function(1.0, 3.0, 2.0) < 1.0
    @test ramp_function(1.0, 3.0, 3.0) == 1.0
    @test ramp_function(1.0, 3.0, 4.0) == 1.0
end

@testset "ramp function - unitful" begin
    @test ramp_function(1.0u"s", 3.0u"s", 0.5u"s") == 0.0
    @test ramp_function(1.0u"s", 3.0u"s", 1.0u"s") == 0.0
    @test 0.0 < ramp_function(1.0u"s", 3.0u"s", 2.0u"s") < 1.0
    @test ramp_function(1.0u"s", 3.0u"s", 3.0u"s") == 1.0
    @test ramp_function(1.0u"s", 3.0u"s", 4.0u"s") == 1.0
    @test_broken ramp_function(1.0u"s^2", 3.0u"s", 4.0u"s")
    @test_broken ramp_function(1.0u"s", 3.0u"s^2", 4.0u"s")
    @test_broken ramp_function(1.0u"s", 3.0u"s", 4.0u"s^2")
end

function linear_extra_force(t, platform_state, system_state, u; p = nothing)
    n_dof = Int(length(platform_state) / 2)
    x = platform_state[1:n_dof]
    dx = platform_state[(n_dof + 1):(2n_dof)]
    return Hydrodynamics.calculate_linear_force(dx, x, p)
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
    pto_system = Hydrodynamics.ExtraSystem(
        n_state = 0,
        force = linear_extra_force,
        p = pto
    )

    mooring_system = Hydrodynamics.ExtraSystem(
        n_state = 0,
        force = linear_extra_force,
        p = mooring
    )

    p = Hydrodynamics.HydroParams(
        inverse_mass = inverse_mass,
        hydro = hydro,
        extra_systems = [pto_system, mooring_system]
    )

    ts = collect(0.0:0.1:0.3)
    sol = hydrodynamic_solver([0.0; 0.1], ts, p, method = :point)
    @test sol.t == ts
    @test length(sol[1, :]) == length(ts)
    @test length(sol[2, :]) == length(ts)
    @test all(isfinite.(sol[1, :]))
    @test all(isfinite.(sol[2, :]))
end

@testset "test stiffness force" begin
    khs = [1.0 0.0; 0.0 1.0]
    position = [2.0, 3.0]
    force = Hydrodynamics.calculate_stiffness_force(position, khs)
    @test force == -position
    @test typeof(force) <: Vector
    @test size(force) == size(position)

    khs = [1.0 -1.0; -1.0 1.0]
    position = [5.0, 100.0]
    force2 = Hydrodynamics.calculate_stiffness_force(position, khs)
    dp = diff(position)[1]
    @test force2 == [dp, -dp]
    @test typeof(force) <: Vector
    @test size(force2) == size(position)

    # No matter khs, force is zero when the equilibrium_position = position
    khs = [1.23e6 -3e3; 0.0 123.45e9]
    position = [2.0, 3.0]
    force = Hydrodynamics.calculate_stiffness_force(position, khs; equilibrium_position = position)
    @test force == 0.0*position
    @test typeof(force) <: Vector
    @test size(force) == size(position)
end

@testset "test stiffness force (Unitful)" begin
    stiffness_units = Unitful.upreferred.([u"N/m" u"N"/rad; u"N*m/m" u"N*m"/rad])
    position_units = [u"m", rad]
    force_units = Unitful.upreferred.([u"N", u"N*m"])

    # test translation, rotational, and mixed unit dofs
    u = for dof in ([1, 2], [2, 2], [1, 2])
        khs = [1.0 0.0; 0.0 1.0] .* stiffness_units[dof, dof]
        position = [2.0, 3.0] .* position_units[dof]
        force = Hydrodynamics.calculate_stiffness_force(position, khs; equilibrium_position = 0.0*position)
        @test Unitful.ustrip.(force) == Unitful.ustrip.(-position)
        @test typeof(force) <: Vector
        @test size(force) == size(position)

        @test typeof(force[1]) <: Unitful.Quantity
        @test Unitful.unit.(Unitful.upreferred.(force)) == force_units[dof]
    end

    # equilibrium_position has mismatched units from position
    dof = [1, 2]
    khs = [1.0 0.0; 0.0 1.0] .* stiffness_units[dof, dof]
    position = [2.0, 3.0] .* position_units[dof]
    @test_broken Hydrodynamics.calculate_stiffness_force(position, khs; equilibrium_position = 0.0)

    # stiffness and position units don't align
    dof = [1, 2]
    khs = [1.0 0.0; 0.0 1.0] .* stiffness_units[dof, dof]
    dof = [1, 1]
    position = [2.0, 3.0] .* position_units[dof]
    @test_broken Hydrodynamics.calculate_stiffness_force(position, khs)
end

@testset "test damping force" begin
    c = [1.0 0.0; 0.0 1.0]
    velocity = [2.0, 3.0]
    force = Hydrodynamics.calculate_damping_force(velocity, c)
    @test force == -velocity
    @test typeof(force) <: Vector
    @test size(force) == size(velocity)

    c = [1.0 -1.0; -1.0 1.0]
    velocity = [5.0, 100.0]
    force2 = Hydrodynamics.calculate_damping_force(velocity, c)
    dp = diff(velocity)[1]
    @test force2 == [dp, -dp]
    @test typeof(force) <: Vector
    @test size(force2) == size(velocity)
end

@testset "test damping force (Unitful)" begin
    damping_units = Unitful.upreferred.([u"N/(m/s)" u"N*s"/rad; u"N*m/(m/s)" u"N*m*s"/rad])
    velocity_units = [u"m/s", rad*u"s^-1"]
    force_units = Unitful.upreferred.([u"N", u"N*m"])

    # test translation, rotational, and mixed unit dofs
    u = for dof in ([1, 2], [2, 2], [1, 2])
        c = [1.0 0.0; 0.0 1.0] .* damping_units[dof, dof]
        velocity = [2.0, 3.0] .* velocity_units[dof]
        force = Hydrodynamics.calculate_damping_force(velocity, c)
        @test Unitful.ustrip.(force) == Unitful.ustrip.(-velocity)
        @test typeof(force) <: Vector
        @test size(force) == size(velocity)

        @test typeof(force[1]) <: Unitful.Quantity
        @test Unitful.unit.(Unitful.upreferred.(force)) == force_units[dof]
    end

    # damping and velocity units don't align
    dof = [1, 2]
    c = [1.0 0.0; 0.0 1.0] .* damping_units[dof, dof]
    dof = [1, 1]
    velocity = [2.0, 3.0] .* velocity_units[dof]
    @test_broken Hydrodynamics.calculate_damping_force(velocity, c)
end

@testset "init_velocity_history" begin
    Hydrodynamics.init_velocity_history(Float64, 3, 10)
    @test Hydrodynamics.velocity_history == zeros(Float64, 1, 3, 10)

    Hydrodynamics.velocity_history .= 1.0
    @test Hydrodynamics.velocity_history == ones(Float64, 1, 3, 10)

    Hydrodynamics.init_velocity_history(Float64, 3, 10)
    @test Hydrodynamics.velocity_history == zeros(Float64, 1, 3, 10)
end

@testset "Capytaine reader" begin
    hydro = Hydrodynamics.Bemio.read_capytaine(joinpath(
        @__DIR__, "..", "examples", "data", "rm3.nc"))
    @test length(hydro.w) > 0
    @test length(hydro.period) == length(hydro.w)
    @test size(hydro.ex, 1) > 0
    @test size(hydro.khs, 1) > 0
end

@testset "WEC-Sim verification" begin
    file = joinpath(@__DIR__, "..", "examples", "wec-sim_comparison_3dof.ipynb")
    position_error = @nbinclude(file)
    # accuracy of the verification to wec-sim when the notebook was first created. 
    # last_result is the RMSE of position in [surge, heave, pitch] dofs x [cic, ss] methods
    last_result = [0.0777 0.1116; 0.7228 0.7229; 0.0069 0.0073]
    @test size(last_result) == size(position_error)
    for idx in CartesianIndices(last_result)
        @test position_error[idx] < last_result[idx]
    end
end

@testset "AF verification" begin
    # TODO - use power performance notebook to confirm gradients are functional
    last_result = 0.051
    file = joinpath(@__DIR__, "..", "examples", "power_performance.ipynb")
    mae = @nbinclude(file)
    @test mae < last_result
end
