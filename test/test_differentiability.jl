using ForwardDiff
using Hydrodynamics
using LinearAlgebra
using Test

function point_parameters(theta)
    T = eltype(theta)
    inverse_mass = reshape([theta[1]], 1, 1)
    K_hs = reshape([theta[2]], 1, 1)
    B = reshape([theta[3]], 1, 1)
    excitation_coeff = zeros(T, 1, 1, 2, 2)
    excitation_coeff[1, 1, 1, 1] = theta[4]
    excitation_coeff[1, 1, 2, 2] = theta[5]
    F = [theta[6]]
    wave = ([0.7, 1.4], [0.0, 0.3], [0.20, 0.08], 0.1, 0.0, 1.0)
    pto = ([zero(T)], reshape([theta[7]], 1, 1), reshape([theta[8]], 1, 1))
    mooring = ([zero(T)], reshape([theta[9]], 1, 1), reshape([theta[10]], 1, 1))
    hydro = (K_hs, B, excitation_coeff, F, wave)
    return (inverse_mass, hydro, pto, mooring)
end

function state_space_parameters(theta)
    inverse_mass, hydro, pto, mooring = point_parameters(theta[1:10])
    K_hs, _, excitation_coeff, F, wave = hydro
    B = zeros(eltype(theta), 1, 1)
    state_space = (
        reshape([-theta[11]], 1, 1),
        reshape([theta[12]], 1, 1),
        reshape([theta[13]], 1, 1),
        reshape([theta[14]], 1, 1),
        1
    )
    return (inverse_mass, (K_hs, B, excitation_coeff, F, wave, state_space), pto, mooring)
end

@testset "Core force and solver paths are ForwardDiff-compatible" begin
    theta = [0.8, 2.5, 0.2, 0.4, -0.1, 0.1, 1.1, 0.05, 0.3, 0.02]
    u = [0.1, -0.2]
    oscillator_response(theta) = sum(Hydrodynamics.hydrodynamic_oscillator(
        u, point_parameters(theta), 0.4))
    gradient = ForwardDiff.gradient(oscillator_response, theta)
    @test all(isfinite, gradient)
    @test norm(gradient) > 0

    ts = collect(0.0:0.05:0.2)
    point_solution_response(theta) = begin
        solution = Hydrodynamics.hydrodynamic_solver(
            [0.1, 0.0], ts, point_parameters(theta); method = :point)
        solution[1, end] + 0.25 * solution[2, end]
    end
    point_gradient = ForwardDiff.gradient(point_solution_response, theta)
    @test all(isfinite, point_gradient)
    @test norm(point_gradient) > 0

    theta_ss = [theta; 0.9; 0.25; 0.18; 0.02]
    ss_solution_response(theta) = begin
        solution = Hydrodynamics.hydrodynamic_solver(
            [0.1, 0.0, 0.0], ts, state_space_parameters(theta); method = :ss)
        solution[1, end] + 0.25 * solution[2, end] + 0.1 * solution[3, end]
    end
    ss_gradient = ForwardDiff.gradient(ss_solution_response, theta_ss)
    @test all(isfinite, ss_gradient)
    @test norm(ss_gradient) > 0
end

@testset "Convolution integral force is ForwardDiff-compatible" begin
    t = reshape(collect(0.0:0.1:0.3), 1, 1, 4)
    ci_response(theta) = begin
        K = reshape(theta[1:4], 1, 1, 4)
        Hydrodynamics.velocity_history = zeros(eltype(theta), 1, 1, 4)
        sum(Hydrodynamics.calculate_ci_force([theta[5]], (K, t)))
    end
    theta = [0.8, 0.5, 0.2, 0.05, -0.3]
    gradient = ForwardDiff.gradient(ci_response, theta)
    @test all(isfinite, gradient)
    @test norm(gradient) > 0
end

@testset "Radiation preprocessing paths are ForwardDiff-compatible" begin
    w = [0.4, 0.9, 1.3, 1.8]
    t = collect(0.0:0.2:0.8)'

    irf_response(theta) = begin
        rd = reshape(theta, 1, 1, 4)
        K, _ = Hydrodynamics.Bemio.radiation_irf(rd, w; w_max = 1.8, t = t)
        sum(K)
    end
    rd_theta = [0.0, 0.4, 0.25, 0.05]
    irf_gradient = ForwardDiff.gradient(irf_response, rd_theta)
    @test all(isfinite, irf_gradient)
    @test norm(irf_gradient) > 0

    ainf_response(theta) = begin
        K = reshape(theta[1:5], 1, 1, 5)
        A = reshape(theta[6:7], 1, 1, 2)
        sum(Hydrodynamics.Bemio.alternate_Ainf(K, A, [0.8, 1.6], vec(t)))
    end
    ainf_theta = [1.0, 0.8, 0.5, 0.2, 0.1, 2.0, 2.1]
    ainf_gradient = ForwardDiff.gradient(ainf_response, ainf_theta)
    @test all(isfinite, ainf_gradient)
    @test norm(ainf_gradient) > 0
end

@testset "Radiation state-space realization has a provided ForwardDiff rule" begin
    t = reshape(collect(0.0:0.2:1.0), 1, 1, 6)
    theta = exp.(-vec(t))

    _, _,
    _,
    _,
    _,
    _,
    orders = Hydrodynamics.Bemio.radiation_state_space(
        reshape(theta, 1, 1, 6), t, 2, -Inf; verbose = false)
    @test orders == reshape([1], 1, 1)

    state_space_response(theta) = begin
        K = reshape(theta, 1, 1, 6)
        A, B,
        C,
        D,
        Kss,
        R2,
        _ = Hydrodynamics.Bemio.radiation_state_space(
            K, t, 2, 0.95; orders = reshape([1], 1, 1), verbose = false)
        sum(A) + sum(B) + sum(C) + sum(D) + 0.1 * sum(Kss) + sum(R2)
    end

    gradient = ForwardDiff.gradient(state_space_response, theta)
    @test all(isfinite, gradient)
    @test norm(gradient) > 0

    step = 1e-6
    theta_plus = copy(theta)
    theta_minus = copy(theta)
    theta_plus[1] += step
    theta_minus[1] -= step
    finite_difference = (
        state_space_response(theta_plus) - state_space_response(theta_minus)) / (2step)
    @test isapprox(gradient[1], finite_difference; rtol = 1e-4, atol = 1e-4)
end
