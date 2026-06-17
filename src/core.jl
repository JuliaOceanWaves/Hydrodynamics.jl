import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS # required for Zygote's reverse AD

global velocity_history

function ramp_function(start_time, ramp_time, current_time)
    if current_time < start_time
        ramp = 0.0
    elseif current_time >= ramp_time
        ramp = 1.0
    else
        ramp = 0.5 * (1 + cos(pi + pi .* current_time ./ ramp_time))
    end
end

function calculate_excitation_force(current_time, excitation_coeff, wave)
    omega, phase, spectrum, dFrequency, start_time, ramp_time = wave
    ov = reshape(omega, (1, 1, length(omega)))
    p = reshape(phase, (1, 1, length(phase)))
    s = reshape(spectrum, (1, 1, length(spectrum)))

    ramp = ramp_function(start_time, ramp_time, current_time)
    exponential_term = ov .* current_time .+ p
    force = ramp .* (excitation_coeff[:, :, :, 1] .* cos.(exponential_term) -
             excitation_coeff[:, :, :, 2] .* sin.(exponential_term)) .*
            sqrt.(2 * s .* dFrequency)

    # Format required for unitful input. `sum` doesn't play nice with matrices of mixed units and dimensions.
    # So instead multiply by an identity matrix to do the same summation in another way.
    # return sum(force[:,:,:]; dims=[2,3])
    o = ones(size(force, 3))
    return force[:, 1, :] * o
end

function calculate_stiffness_force(x, Kₕₛ)
    return -Kₕₛ * x
end

function calculate_radiation_force(dx, B)
    return -B * dx
end

function init_velocity_history(n_dof, n_time_steps)
    global velocity_history = zeros(1, n_dof, n_time_steps)
end

function calculate_ci_force(dx, cic)
    # Convolution integrals
    Kᵣ, tᵣ = cic
    global velocity_history
    velocity_history .= circshift(velocity_history, (0, 0, 1))
    velocity_history[1, :, 1] = dx
    integrand = sum(Kᵣ .* velocity_history; dims = [2])[:, 1, :] # nDOF, nDOF, nt --> nDOF, nt
    dt = diff(tᵣ; dims = 3)[:, 1, :] # 1, nt-1
    radiation_force = sum(
        (integrand[:, 1:(end - 1)] .+ integrand[:, 2:end]) .* 0.5 .* dt;
        dims = [2])[:, 1] # nDOF
    return -radiation_force
end

function calculate_added_mass_force(ddx, A)
    return -A * ddx
end

function calculate_linear_force(dx, x, coefficients)
    x₀, k, c = coefficients
    return -c * dx - k * (x - x₀)
end

function calculate_total_linear_hydro_forces(dx, x, p, t)
    # NOTE: added mass force is not included and should be lumped with the 
    # body's mass matrix when solving the equations of motion that depend on this calculation
    (inverse_mass, hydro, pto, mooring) = p
    Kₕₛ, B, excitation_coeff, F, wave = hydro[1:5]
    Fₑₓ = calculate_excitation_force(t, excitation_coeff, wave)
    Fₖₕₛ = calculate_stiffness_force(x, Kₕₛ)
    Fᵣ = calculate_radiation_force(dx, B)
    Fₚₜₒ = calculate_linear_force(dx, x, pto)
    Fₘ = calculate_linear_force(dx, x, mooring)
    return Fₑₓ .+ Fₖₕₛ .+ Fᵣ .+ Fₚₜₒ .+ Fₘ .+ F
end

function hydrodynamic_oscillator(u, p, t)
    n_dof = Int64(length(u) / 2)
    x = u[1:n_dof] # position
    dx = u[(n_dof + 1):end] # velocity

    inverse_mass = p[1]
    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(dx, x, p, t)
    ddx = inverse_mass * Fₜₒₜₐₗ

    return [dx; ddx]
end

function hydrodynamic_oscillator_cic(u, p, t)
    n_dof = Int64(length(u) / 2)
    x = u[1:n_dof] # position
    dx = u[(n_dof + 1):end] # velocity

    inverse_mass = p[1]
    cic = p[2][6]
    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(dx, x, p, t) +
             calculate_ci_force(dx, cic)
    ddx = inverse_mass * Fₜₒₜₐₗ

    return [dx; ddx]
end

function hydrodynamic_oscillator_ss(u, p, t)
    # u = [x, dx, states]
    # added mass should utilize infinite frequency added mass only
    # c should not include radiation damping
    # system of equations in u and du should include velocity and the state space vector
    inverse_mass = p[1]
    state_space = p[2][6]
    Aᵣ, Bᵣ, Cᵣ, Dᵣ, nₛₛ = state_space
    n_dof = Int64((size(u)[1] - nₛₛ) / 2)

    x = u[1:n_dof] # position
    dx = u[(n_dof + 1):(n_dof * 2)] # velocity
    ss = u[(end - nₛₛ + 1):end] # state space vector

    # The general state space is defined such that:
    #    dx = Aᵣ * x + Bᵣ * u
    #     y = Cᵣ * x + Dᵣ * u
    # Where:
    #    x is the state vector (ss)
    #    y is the output (radiation force)
    #    u is the input (velocity)
    Fᵣ = Cᵣ * ss + Dᵣ * dx
    dss = Aᵣ * ss + Bᵣ * dx

    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(dx, x, p, t) - Fᵣ
    ddx = inverse_mass * Fₜₒₜₐₗ

    return [dx; ddx; dss]
end

function hydrodynamic_solver(u₀, ts, p; method::Symbol = :point)
    # u₀ = [x₀, dx₀]
    dt = diff(ts[1:2])[1]

    if method == :point
        ode_prob = ODE.ODEProblem(hydrodynamic_oscillator, u₀, ts[[1, end]], p)
        ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)

    elseif method == :cic
        init_velocity_history(size(p[2][6][1], 2), size(p[2][6][1], 3))
        ode_prob = ODE.ODEProblem(hydrodynamic_oscillator_cic, u₀, ts[[1, end]], p)
        ode_sol = ODE.solve(ode_prob, ODE.Euler(), saveat = dt, adaptive = false, dt = dt)

    elseif method == :ss
        ode_prob = ODE.ODEProblem(hydrodynamic_oscillator_ss, u₀, ts[[1, end]], p)
        ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    else
        throw(ArgumentError("method must be a Symbol with value :point, :cic, or :ss"))
    end

    return ode_sol
end

function hydrodynamic_solver_cic(u₀, ts, p)
    # u₀ = [x₀, dx₀]
    dt = diff(ts[1:2])[1]
    global velocity_history = zeros(1, size(p[7][1], 2), size(p[7][1], 3))

    ode_prob = ODE.SecondOrderODEProblem(
        hydrodynamic_oscillator_cic, u₀, ts[[1, end]], p)
    ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    return ode_sol
end

function hydrodynamic_solver_ss(u₀, ts, p)
    # u₀ = [x₀, dx₀, states₀]
    dt = diff(ts[1:2])[1]

    ode_prob = ODE.ODEProblem(
        hydrodynamic_oscillator_ss, u₀, ts[[1, end]], p)
    ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    return ode_sol
end

function hydrodynamic_solver_2nd(dx₀, x₀, ts, p)
    dt = diff(ts[1:2])[1]
    ode_prob = ODE.SecondOrderODEProblem(
        hydrodynamic_oscillator, dx₀, x₀, ts[[1, end]], p)
    ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    return ode_sol
end
