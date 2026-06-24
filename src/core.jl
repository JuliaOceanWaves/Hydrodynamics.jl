import OrdinaryDiffEq as ODE
import SimpleDiffEq as SDE

struct HydrodynamicSolution{TT, TX, TV}
    t::TT
    x::TX
    dx::TV
end

global velocity_history

function _collect_real_eltypes!(types, x)
    if x isa Real
        push!(types, typeof(x))
    elseif x isa AbstractArray
        if eltype(x) <: Real
            push!(types, eltype(x))
        else
            foreach(value -> _collect_real_eltypes!(types, value), x)
        end
    elseif x isa Tuple || x isa NamedTuple
        foreach(value -> _collect_real_eltypes!(types, value), x)
    end
    return types
end

function _real_eltype(args...)
    types = DataType[]
    for arg in args
        _collect_real_eltypes!(types, arg)
    end
    if isempty(types)
        return Float64
    end
    return reduce(promote_type, types)
end

function ramp_function(start_time, ramp_time, current_time)
    if current_time <= start_time
        return zero(current_time)
    elseif current_time >= ramp_time
        return one(current_time)
    end
    return 0.5 * (1 .+ cos.(pi .+ pi .* current_time ./ ramp_time))
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
    weights = ones(size(force, 3))
    return force[:, 1, :] * weights
end

function calculate_stiffness_force(x, Kₕₛ)
    return -Kₕₛ * x
end

function calculate_radiation_force(dx, B)
    return -B * dx
end

function init_velocity_history(T, n_dof, n_time_steps)
    global velocity_history = zeros(T, 1, n_dof, n_time_steps)
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

function calculate_total_linear_hydro_forces(x, dx, hydro, t)
    # NOTE: added mass force is not included and should be lumped with the 
    # body's mass matrix when solving the equations of motion that depend on this calculation
    Kₕₛ, B, excitation_coeff, Fgb, wave = hydro[1:5]
    Fₑₓ = calculate_excitation_force(t, excitation_coeff, wave) # excitation force
    Fₖₕₛ = calculate_stiffness_force(x, Kₕₛ) # hydrostatic stiffness force
    Fᵣ = calculate_radiation_force(dx, B) # radiation force
    # Fgb = gravity force + buoyancy force
    return Fₑₓ .+ Fᵣ .+ Fₖₕₛ .+ Fgb
end

function hydrodynamic_oscillator(u, p, t)
    n_dof = Int64(length(u) / 2)
    x = u[1:n_dof] # position
    dx = u[(n_dof + 1):end] # velocity

    inverse_mass = p[1]
    hydro = p[2]
    force_other, u_other, p_other = p[3]
    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(x, dx, hydro, t) +
             force_other(t, [x; dx], u_other; p_other)
    ddx = inverse_mass * Fₜₒₜₐₗ

    return [dx; ddx]
end

function hydrodynamic_oscillator_cic(u, p, t)
    n_dof = Int64(length(u) / 2)
    x = u[1:n_dof] # position
    dx = u[(n_dof + 1):end] # velocity

    inverse_mass = p[1]
    hydro = p[2]
    cic = hydro[6]
    force_other, u_other, p_other = p[3]
    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(x, dx, hydro, t) +
             calculate_ci_force(dx, cic) + force_other(t, [x; dx], u_other; p_other)
    ddx = inverse_mass * Fₜₒₜₐₗ

    return [dx; ddx]
end

function hydrodynamic_oscillator_ss(u, p, t)
    # u = [x, dx, states]
    # added mass should utilize infinite frequency added mass only
    # c should not include radiation damping
    # system of equations in u and du should include velocity and the state space vector
    inverse_mass = p[1]
    hydro = p[2]
    state_space = hydro[6]
    force_other, u_other, p_other = p[3]
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
    Fᵣ = - (Cᵣ * ss + Dᵣ * dx)
    dss = Aᵣ * ss + Bᵣ * dx

    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(x, dx, hydro, t) + Fᵣ +
             force_other(t, [x; dx], u_other; p_other)
    ddx = inverse_mass * Fₜₒₜₐₗ

    return [dx; ddx; dss]
end

function hydrodynamic_stepping(dx0, x0, ts, p)
    length(ts) >= 2 || throw(ArgumentError("time vector must contain at least two samples"))
    x = [copy(x0) for _ in eachindex(ts)]
    dx = [copy(dx0) for _ in eachindex(ts)]

    for i in 1:(length(ts) - 1)
        dt = ts[i + 1] - ts[i]
        acceleration = hydrodynamic_oscillator([x[i]; dx[i]], p, ts[i])
        dx[i + 1] = dx[i] + dt * acceleration
        x[i + 1] = x[i] + dt * dx[i + 1]
    end

    return HydrodynamicSolution(ts, x, dx)
end

function hydrodynamic_solver(u₀, ts, p; method::Symbol = :point)
    # u₀ = [x₀, dx₀]
    T = _real_eltype(u₀, p)
    u₀ = T === eltype(u₀) ? u₀ : convert.(T, u₀)
    dt = diff(ts[1:2])[1]

    if method == :point
        ode_prob = ODE.ODEProblem(hydrodynamic_oscillator, u₀, ts[[1, end]], p)
        ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)

    elseif method == :cic
        init_velocity_history(T, size(p[2][6][1], 2), size(p[2][6][1], 3))
        ode_prob = ODE.ODEProblem(hydrodynamic_oscillator_cic, u₀, ts[[1, end]], p)
        ode_sol = ODE.solve(
            ode_prob, SDE.SimpleEuler(), saveat = dt, adaptive = false, dt = dt)

    elseif method == :ss
        ode_prob = ODE.ODEProblem(hydrodynamic_oscillator_ss, u₀, ts[[1, end]], p)
        ode_sol = ODE.solve(ode_prob, ODE.Vern6(), saveat = dt)
    else
        throw(ArgumentError("method must be a Symbol with value :point, :cic, or :ss"))
    end

    return ode_sol
end
