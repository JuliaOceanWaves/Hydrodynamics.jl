import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS # required for Zygote's reverse AD
using DSP

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

function hydrodynamic_oscillator(du, u, p, t)
    (k, c, inverse_mass, excitation_coeff, constant_forces, wave) = p
    excitation_force = calculate_excitation_force(t, excitation_coeff, wave)
    return inverse_mass * (-c * du - k * u + excitation_force + constant_forces) # ddu
end

function hydrodynamic_oscillator_cic(du, u, p, t)
    (k, c, inverse_mass, excitation_coeff, constant_forces, wave, cic) = p

    # Convolution integral
    Kᵣ, tᵣ = cic
    global velocity_history = circshift(velocity_history, 1)
    global velocity_history[1, :, 1] = du
    # velocity_history = zeros(1, size(Kᵣ, 2), size(Kᵣ, 3))
    # integrand = DSP.conv(Kᵣ, velocity_history) # nDOF, nDOF, nt
    integrand = Kᵣ .* velocity_history # nDOF, nDOF, nt
    dt = diff(tᵣ; dims = 3) # 1, 1, nt-1
    rad_force = sum(
        (integrand[:, :, 1:(end - 1)] .+ integrand[:, :, 2:end]) .* 0.5 .* dt;
        dims = [3]) # nDOF, nDOF

    excitation_force = calculate_excitation_force(t, excitation_coeff, wave)
    return inverse_mass * (-c * du - k * u + excitation_force + constant_forces) # ddu
end

function hydrodynamic_oscillator_ss(u, p, t)
    # u = [x, dx, states]
    # added mass should utilize infinite frequency added mass only
    # c should not include radiation damping
    # system of equations in u and du should include velocity and the state space vector
    (k, c, inverse_mass, excitation_coeff, constant_forces, wave, state_space) = p

    Aᵣ, Bᵣ, Cᵣ, Dᵣ, nₛₛ = state_space
    n_dof = Int64((size(u)[1] - nₛₛ) / 2)

    x = u[1:n_dof]
    dx = u[(n_dof + 1):(n_dof * 2)]
    ss = u[(end - nₛₛ + 1):end]

    # d(SS vector) = dx = Ar * x + Br * velocity
    #    CI kernel =  y = Cr * x + Dr * velocity
    # rad_force         = Cr * u[7:12] + Dr * du[1:6]
    rad_force = Cᵣ * ss + Dᵣ * dx
    du_ss = Aᵣ * ss + Bᵣ * dx

    excitation_force = calculate_excitation_force(t, excitation_coeff, wave)
    ddx = inverse_mass * (-c * dx - k * x + excitation_force + constant_forces + rad_force)
    return [u[(n_dof + 1):(n_dof * 2)]; ddx; du_ss] # dx, ddx, du_ss
end

function hydrodynamic_solver(dx₀, x₀, ts, p)
    dt = diff(ts[1:2])[1]
    ode_prob = ODE.SecondOrderODEProblem(
        hydrodynamic_oscillator, dx₀, x₀, ts[[1, end]], p)
    ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    return ode_sol
end

function hydrodynamic_solver_cic(dx₀, x₀, ts, p)
    dt = diff(ts[1:2])[1]

    global velocity_history = zeros(1, size(p[7][1], 2), size(p[7][1], 3))
    ode_prob = ODE.SecondOrderODEProblem(
        hydrodynamic_oscillator_cic, dx₀, x₀, ts[[1, end]], p)
    ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    return ode_sol
end

function hydrodynamic_solver_ss(u₀, ts, p)
    dt = diff(ts[1:2])[1]
    # u₀ = [x₀, dx₀, states₀]
    ode_prob = ODE.ODEProblem(
        hydrodynamic_oscillator_ss, u₀, ts[[1, end]], p)
    ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    return ode_sol
end
