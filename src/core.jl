struct HydrodynamicSolution{TT, TX, TV}
    t::TT
    x::TX
    dx::TV
end

function ramp_function(start_time, ramp_time, current_time)
    if current_time < start_time
        return 0.0
    elseif current_time >= ramp_time
        return 1.0
    end
    return 0.5 * (1 + cos(pi + pi * current_time / ramp_time))
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

    weights = ones(size(force, 3))
    return force[:, 1, :] * weights
end

function hydrodynamic_oscillator(du, u, p, t)
    (k, c, inverse_mass, excitation_coeff, constant_forces, wave) = p
    excitation_force = calculate_excitation_force(t, excitation_coeff, wave)
    return inverse_mass * (-c * du - k * u + excitation_force + constant_forces)
end

function hydrodynamic_oscillator_convolution(du, u, p, t)
    return hydrodynamic_oscillator(du, u, p, t)
end

function hydrodynamic_solver(dx0, x0, ts, p)
    length(ts) >= 2 || throw(ArgumentError("time vector must contain at least two samples"))
    x = [copy(x0) for _ in eachindex(ts)]
    dx = [copy(dx0) for _ in eachindex(ts)]

    for i in 1:(length(ts) - 1)
        dt = ts[i + 1] - ts[i]
        acceleration = hydrodynamic_oscillator(dx[i], x[i], p, ts[i])
        dx[i + 1] = dx[i] + dt * acceleration
        x[i + 1] = x[i] + dt * dx[i + 1]
    end

    return HydrodynamicSolution(ts, x, dx)
end
