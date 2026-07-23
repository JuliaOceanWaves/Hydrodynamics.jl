import OrdinaryDiffEq as ODE
import SimpleDiffEq as SDE

Base.@kwdef struct HydroParams
    inverse_mass
    hydro
    force_other
    u_control = nothing
    p_other = nothing

    extra_rhs = nothing
    extra_force = nothing
    p_extra = nothing
end

struct HydrodynamicSolution{TT,TX,TV}
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

function calculate_excitation_force(current_time, excitation_coefficients, wave)
    omega, phase, spectrum, frequency_spacing, start_time, ramp_time = wave
    omega_reshaped = reshape(omega, (1, 1, length(omega)))
    phase_reshaped = reshape(phase, (1, 1, length(phase)))
    spectrum_reshaped = reshape(spectrum, (1, 1, length(spectrum)))

    ramp = ramp_function(start_time, ramp_time, current_time)
    exponential_term = omega_reshaped .* current_time .+ phase_reshaped
    force = ramp .* (excitation_coefficients[:, :, :, 1] .* cos.(exponential_term) -
                     excitation_coefficients[:, :, :, 2] .* sin.(exponential_term)) .*
            sqrt.(2 * spectrum_reshaped .* frequency_spacing)

    # Format required for unitful input. `sum` doesn't play nice with matrices of mixed units and dimensions.
    # So instead multiply by an identity matrix to do the same summation in another way.
    weights = ones(size(force, 3))
    return force[:, 1, :] * weights
end

function calculate_stiffness_force(position, stiffness; equilibrium_position=0.0*position)
    return -stiffness * (position - equilibrium_position)
end

function calculate_damping_force(velocity, damping)
    return -damping * velocity
end

function init_velocity_history(T, n_dof, n_time_steps)
    global velocity_history = zeros(T, 1, n_dof, n_time_steps)
end

function calculate_radiation_force_convolution(velocity, irf)
    # Convolution integral form of the radiation damping force
    global velocity_history
    velocity_history .= circshift(velocity_history, (0, 0, 1))
    velocity_history[1, :, 1] = velocity

    Kᵣ, tᵣ = irf # impulse response function tuple

    integrand = sum(Kᵣ .* velocity_history; dims=[2])[:, 1, :] # nDOF, nDOF, nt --> nDOF, nt
    dt = diff(tᵣ; dims=3)[:, 1, :] # 1, nt-1
    radiation_force = sum(
        (integrand[:, 1:(end-1)] .+ integrand[:, 2:end]) .* 0.5 .* dt;
        dims=[2])[:, 1] # nDOF
    return -radiation_force
end

function calculate_added_mass_force(acceleration, added_mass_coefficient)
    # Note: this function is not used in calculate_total_linear_hydro_forces. 
    # It is recommended to lump added mass with the body mass for simulation stability and accuracy
    return -added_mass_coefficient * acceleration
end

function calculate_linear_force(velocity, position, coefficients)
    equilibrium_position, stiffness, damping = coefficients
    return calculate_damping_force(velocity, damping) +
           calculate_stiffness_force(position, stiffness; equilibrium_position=equilibrium_position)
end

function calculate_total_linear_hydro_forces(position, velocity, hydro, time)
    # NOTE: added mass force is not included and should be lumped with the 
    # body's mass matrix when solving the equations of motion that depend on this calculation
    hydrostatic_stiffness_coefficient,
    radiation_damping_coefficient, excitation_coefficients,
    net_gravity_buoyancy_force, wave = hydro[1:5]

    excitation_force = calculate_excitation_force(time, excitation_coefficients, wave)
    hydrostatic_stiffness_force = calculate_stiffness_force(
        position, hydrostatic_stiffness_coefficient; equilibrium_position=0.0*position)
    radiation_force = calculate_damping_force(velocity, radiation_damping_coefficient)

    return excitation_force .+ radiation_force .+ hydrostatic_stiffness_force .+
           net_gravity_buoyancy_force
end

function hydrodynamic_oscillator(state, p::HydroParams, t)
    n_dof = size(p.inverse_mass, 1)
    n_hydro = 2 * n_dof

    hydro_state = state[1:n_hydro]
    extra_state = state[(n_hydro+1):end]

    x = hydro_state[1:n_dof]
    dx = hydro_state[(n_dof+1):n_hydro]
    platform_state = [x; dx]

    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(x, dx, p.hydro, t) +
             p.force_other(t, platform_state, p.u_control; p=p.p_other)

    if p.extra_force !== nothing
        Fₜₒₜₐₗ += p.extra_force(
            t,
            platform_state,
            extra_state,
            p.u_control;
            p=p.p_extra,
        )
    end

    ddx = p.inverse_mass * Fₜₒₜₐₗ
    du_hydro = [dx; ddx]

    if isempty(extra_state)
        return du_hydro
    end

    p.extra_rhs === nothing &&
        throw(ArgumentError("extra states were provided, but p.extra_rhs is nothing"))

    du_extra = p.extra_rhs(
        t,
        platform_state,
        extra_state,
        p.u_control;
        p=p.p_extra,
    )

    return [du_hydro; du_extra]
end

function hydrodynamic_oscillator_cic(state, p::HydroParams, t)
    n_dof = size(p.inverse_mass, 1)
    n_hydro = 2 * n_dof

    hydro_state = state[1:n_hydro]
    extra_state = state[(n_hydro+1):end]

    x = hydro_state[1:n_dof]
    dx = hydro_state[(n_dof+1):n_hydro]
    platform_state = [x; dx]

    cic = p.hydro[6]

    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(x, dx, p.hydro, t) +
             calculate_radiation_force_convolution(dx, cic) +
             p.force_other(t, platform_state, p.u_control; p=p.p_other)

    if p.extra_force !== nothing
        Fₜₒₜₐₗ += p.extra_force(
            t,
            platform_state,
            extra_state,
            p.u_control;
            p=p.p_extra,
        )
    end

    ddx = p.inverse_mass * Fₜₒₜₐₗ
    du_hydro = [dx; ddx]

    if isempty(extra_state)
        return du_hydro
    end

    p.extra_rhs === nothing &&
        throw(ArgumentError("extra states were provided, but p.extra_rhs is nothing"))

    du_extra = p.extra_rhs(
        t,
        platform_state,
        extra_state,
        p.u_control;
        p=p.p_extra,
    )

    return [du_hydro; du_extra]
end

function hydrodynamic_oscillator_ss(state, p::HydroParams, t)
    # u = [position, velocity, states]
    # added mass should utilize infinite frequency added mass only
    # c should not include radiation damping
    # system of equations in u and du should include velocity and the state space vector
    state_space = p.hydro[6]
    Aᵣ, Bᵣ, Cᵣ, Dᵣ, nₛₛ = state_space

    n_dof = size(p.inverse_mass, 1)
    n_hydro = 2 * n_dof + nₛₛ

    hydro_state = state[1:n_hydro]
    extra_state = state[(n_hydro+1):end]

    x = hydro_state[1:n_dof]
    dx = hydro_state[(n_dof+1):(2n_dof)]
    ss = hydro_state[(2n_dof+1):(2n_dof+nₛₛ)]
    platform_state = [x; dx]
    # The general state space is defined such that:
    #    dx = Aᵣ * x + Bᵣ * hydro_state
    #     y = Cᵣ * x + Dᵣ * hydro_state
    # Where:
    #    x is the state vector (ss)
    #    y is the output (radiation force)
    #    hydro_state is the input (velocity)
    Fᵣ = -(Cᵣ * ss + Dᵣ * dx)
    dss = Aᵣ * ss + Bᵣ * dx

    Fₜₒₜₐₗ = calculate_total_linear_hydro_forces(x, dx, p.hydro, t) + Fᵣ +
             p.force_other(t, platform_state, p.u_control; p=p.p_other)

    if p.extra_force !== nothing
        Fₜₒₜₐₗ += p.extra_force(
            t,
            platform_state,
            extra_state,
            p.u_control;
            p=p.p_extra,
        )
    end

    ddx = p.inverse_mass * Fₜₒₜₐₗ
    du_hydro = [dx; ddx; dss]

    if isempty(extra_state)
        return du_hydro
    end

    p.extra_rhs === nothing &&
        throw(ArgumentError("extra states were provided, but p.extra_rhs is nothing"))

    du_extra = p.extra_rhs(
        t,
        platform_state,
        extra_state,
        p.u_control;
        p=p.p_extra,
    )

    return [du_hydro; du_extra]
end

function hydrodynamic_stepping(dx0, x0, ts, p)
    length(ts) >= 2 || throw(ArgumentError("time vector must contain at least two samples"))
    x = [copy(x0) for _ in eachindex(ts)]
    dx = [copy(dx0) for _ in eachindex(ts)]

    n_dof = size(p.inverse_mass, 1)

    for i in 1:(length(ts)-1)
        dt = ts[i+1] - ts[i]
        du = hydrodynamic_oscillator([x[i]; dx[i]], p, ts[i])
        ddx = du[(n_dof+1):(2n_dof)]

        dx[i+1] = dx[i] + dt * ddx
        x[i+1] = x[i] + dt * dx[i+1]
    end

    return HydrodynamicSolution(ts, x, dx)
end

function hydrodynamic_solver(hydro_state₀, ts, p::HydroParams; method::Symbol=:point)
    # hydro_state₀ = [x₀, dx₀]
    T = _real_eltype(hydro_state₀, p)
    hydro_state₀ = T === eltype(hydro_state₀) ? hydro_state₀ : convert.(T, hydro_state₀)
    dt = diff(ts[1:2])[1]

    if method == :point
        problem = ODE.ODEProblem(hydrodynamic_oscillator, hydro_state₀, ts[[1, end]], p)
        solution = ODE.solve(problem, ODE.Vern6(), saveat=dt)

    elseif method == :cic
        init_velocity_history(T, size(p.hydro[6][1], 2), size(p.hydro[6][1], 3))
        problem = ODE.ODEProblem(hydrodynamic_oscillator_cic, hydro_state₀, ts[[1, end]], p)
        solution = ODE.solve(
            problem, SDE.SimpleEuler(), saveat=dt, adaptive=false, dt=dt)

    elseif method == :ss
        problem = ODE.ODEProblem(hydrodynamic_oscillator_ss, hydro_state₀, ts[[1, end]], p)
        solution = ODE.solve(problem, ODE.Vern6(), saveat=dt)
    else
        throw(ArgumentError("method must be a Symbol with value :point, :cic, or :ss"))
    end

    return solution
end
