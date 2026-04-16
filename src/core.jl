import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS # required for Zygote's reverse AD

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
    return inverse_mass * (-c * du - k * u + excitation_force + constant_forces)
end

function hydrodynamic_oscillator_convolution(du, u, p, t)
    (k, c, inverse_mass, excitation_coeff, constant_forces, wave) = p
    excitation_force = calculate_excitation_force(t, excitation_coeff, wave)
    return inverse_mass * (-c * du - k * u + excitation_force + constant_forces)
end

function hydrodynamic_solver(dx₀, x₀, ts, p)
    dt = diff(ts[1:2])[1]
    ode_prob = ODE.SecondOrderODEProblem(
        hydrodynamic_oscillator, dx₀, x₀, ts[[1, end]], p)
    ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    return ode_sol
end
