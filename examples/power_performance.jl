# First import all necessary julia packages:
using Unitful
using DimensionfulAngles: radᵃ as rad, θ₀, 𝐀, Dispersion
using LinearAlgebra
using Plots
import Random
import NetCDF
import ForwardDiff
import WaveSpectra
import FiniteDiff
include("..\\src\\Hydrodynamics.jl")

dof = 1:2:5
n_dof = length(dof)
dof_names = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"][dof]

length_units = u"m"
time_units = u"s"

mass_units = 1.0 .* [repeat([u"kg"], 3, 3) repeat([0u"kg*m"/rad], 3, 3);
              repeat([0u"kg*m"], 3, 3) repeat([0u"kg*m^2"/rad], 3, 3)][dof, dof]

damping_units = 1.0 *
                [repeat([u"kg*m/s^2 / (m/s)"], 3, 3) repeat([u"kg*m/s^2 / (1/s)"/rad], 3, 3);
                 repeat([u"kg*m/s^2 * m / (m/s)"], 3, 3) repeat([u"kg*m/s^2 * m / (1/s)"/rad], 3, 3)][
    dof, dof]
stiffness_units = 1.0 .* [repeat([u"kg*m/s^2 / m"], 3, 3) repeat([u"kg*m/s^2"/rad], 3, 3);
                   repeat([u"kg*m/s^2 * m / m"], 3, 3) repeat([u"kg*m/s^2 * m"/rad], 3, 3)][
    dof, dof]
excitation_units = [repeat([u"kg*m/s^2 / m"], 3, 1); repeat([u"kg*m/s^2 * m / m"], 3, 1)][dof]

position_units = [u"m", u"m", u"m", rad, rad, rad]
velocity_units = position_units / time_units
acceleration_units = position_units / time_units^2

# Wave conditions
Hₛ₀ = 1.0*length_units
Tₑ = 6.0*time_units

# Time-domain set-up
t0 = 0.0*time_units
tr = Tₑ * 1.0
tf = Tₑ * 10.0
dt = 1.0e-2*time_units
ts = collect(t0:dt:tf)
nt = length(ts)
i_ramp = Int64(tr / dt + 1)

# Hydrodynamics
bem_file = joinpath(@__DIR__, "data", "rm3.nc")

A = NetCDF.ncread(bem_file, "added_mass")[dof, dof, :] .* mass_units # Dimensions: influenced_dof radiating_dof omega
B = NetCDF.ncread(bem_file, "radiation_damping")[dof, dof, :] .* damping_units # Dimensions: influenced_dof radiating_dof omega
Kₕₛ = NetCDF.ncread(bem_file, "hydrostatic_stiffness")[dof, dof]' .* stiffness_units # Dimensions: radiating_dof influenced_dof --> influenced_dof radiating_dof
fₑₓ = NetCDF.ncread(bem_file, "excitation_force")[dof, :, :, :] .* excitation_units # Dimensions: influenced_dof wave_dir omega complex
fₑₓ[:, :, :, 2] = -1 .* fₑₓ[:, :, :, 2] # Convert Capytaine's convention to one in which the wave propagates in +X

ω = NetCDF.ncread(bem_file, "omega")*rad/time_units
dω = ω[2] - ω[1]
nω = length(ω)

f = uconvert.(time_units^-1, ω, Dispersion())
df = uconvert.(time_units^-1, dω, Dispersion())

# Hydrostatics parameters - these should also come from BEM
cg = [0.0, 0.0, -0.72]*u"m"
volume = 725.8330*u"m^3"
cb = [0.0, 0.0, -1.2927]*u"m"

# Convolution integral calculation
Kᵣ,
tᵣ = Hydrodynamics.Bemio.radiation_irf(Unitful.ustrip.(B), Unitful.ustrip.(ω), w_max = 5.0,
    t = collect(0:Unitful.ustrip.(dt):30)')
tᵣ = tᵣ .* u"s"
Kᵣ = Kᵣ .* damping_units .* u"s^-1"
Ainf = A[:, :, end]
# Ainf = Hydrodynamics.Bemio.alternate_Ainf(Unitful.ustrip.(Kᵣ), hydro.am, hydro.w, Unitful.ustrip.(tᵣ)) .* mass_units
cic = (Unitful.ustrip.(Kᵣ), Unitful.ustrip.(tᵣ))

# State space formation
Aᵣ, Bᵣ, Cᵣ, Dᵣ, Kₛₛ, R²ₛₛ,
orderₛₛ = Hydrodynamics.Bemio.radiation_state_space(Kᵣ, tᵣ, 8, 0.95)
nₛₛ = sum(orderₛₛ)
s₀ = zeros(nₛₛ)
ss = (Aᵣ, Bᵣ, Cᵣ, Dᵣ, nₛₛ)

S = WaveSpectra.ParametricSpectra.spectrum_pierson_moskowitz(f, Hₛ₀, Tₑ)
ϕ = rand(Random.Xoshiro(0), Float64, size(S)) * 2 * pi * rad
maxS_index = argmax(S)

# Calculate and visualize the wave elevation
ramp_timeseries = zeros(length(ts))
for i in 1:length(ts)
    ramp_timeseries[i] = Hydrodynamics.ramp_function(t0, tr, ts[i])
end

elevation = sum(sqrt.(2*S*df) .* cos.(ω .* ts' .+ ϕ), dims = 1)'
ramped_elevation = elevation .* ramp_timeseries
Plots.plot(
    ts, elevation, xlabel = "Time (s)", ylabel = "Wave elevation (m)", label = "No ramp")
Plots.plot!(ts, ramped_elevation, xlabel = "Time (s)",
    ylabel = "Wave elevation (m)", label = "Ramped", ls = :dash)

# Body properties
m = 725833u"kg"
Ixx = [20907301, 21306090.66, 37085481.11]u"kg*m^2"/rad
Ixy = [1e6, 1e6, 1e5]u"kg*m^2"/rad
I = diagm(Ixx)
I[1, 2:3] = I[2:3, 1] = Ixy[1:2]
I[2, 3] = I[3, 2] = Ixy[3]
body_mass = [diagm(repeat([m], 3)) repeat([0u"kg*m"/rad], 3, 3); repeat([0u"kg*m"], 3, 3) I]

# Coefficients for the form: ẍ + c * ẋ + k * x = 0
mass = body_mass[dof, dof] .+ A[:, :, maxS_index]
inv_mass_units = 1 ./ Unitful.unit.(mass')
inv_mass = inv(Unitful.ustrip.(mass)) .* inv_mass_units

# Adjusted mass calculations
mass_inf = body_mass[dof, dof] .+ Ainf
inv_mass_inf = inv(Unitful.ustrip.(mass_inf)) .* inv_mass_units

g = zeros(6) .* acceleration_units
g[3] = -NetCDF.ncread(bem_file, "g")[1] * u"m/s^2"
force_gravity = diag(body_mass) .* g
rho = NetCDF.ncread(bem_file, "rho")[1] * u"kg/m^3"
force_buoyancy = - rho * g * volume
CGCB = cb - cg
force_buoyancy[4:6] = cross(CGCB, force_buoyancy[1:3])
force_hydrostatic = force_gravity[dof] + force_buoyancy[dof]

# Initial conditions
x₀ = zeros(size(dof)) .* position_units[dof]
dx₀ = zeros(size(dof)) .* velocity_units[dof]
u₀ = [x₀; dx₀]
u₀_ss = [x₀; dx₀; s₀]

# PTO parameters
kₚₜₒ = diagm(zeros(size(dof))) .* stiffness_units
cₚₜₒ = diagm(1.0e5 .* ones(size(dof))) .* damping_units
cₚₜₒ_unitless = Unitful.ustrip.(cₚₜₒ)
pto = (x₀, kₚₜₒ, cₚₜₒ)

# Mooring parameters
kₘ = diagm(zeros(size(dof))) .* stiffness_units
cₘ = diagm(zeros(size(dof))) .* damping_units
mooring = (x₀, kₘ, cₘ)

function linear_extra_force(t, platform_state, system_state, u; p = nothing)
    n_dof = Int(length(platform_state) / 2)

    x = platform_state[1:n_dof]
    dx = platform_state[(n_dof + 1):(2n_dof)]

    return Hydrodynamics.calculate_linear_force(dx, x, p)
end

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

extra_systems = [pto_system, mooring_system]
extra_systems_unitless = Hydrodynamics.ustrip.(extra_systems)

# Unitful and unitless parameter groups
wave = (ω, ϕ, S, df, t0, tr)
hydro = (Kₕₛ, B[:, :, maxS_index], fₑₓ, force_hydrostatic, wave) # optionally cic or ss tuples on the end
p_unitful = Hydrodynamics.HydroParams(
    inverse_mass = inv_mass,
    hydro = hydro,
    extra_systems = extra_systems
)

wave_unitless = (Unitful.ustrip.(ω), Unitful.ustrip.(ϕ), Unitful.ustrip.(S),
    Unitful.ustrip.(df), Unitful.ustrip.(t0), Unitful.ustrip.(tr))
hydro_unitless = Unitful.ustrip.(Kₕₛ),
Unitful.ustrip.(B[:, :, maxS_index]), Unitful.ustrip.(fₑₓ),
Unitful.ustrip.(force_hydrostatic), wave_unitless
p_unitless = Hydrodynamics.HydroParams(
    inverse_mass = Unitful.ustrip.(inv_mass_inf),
    hydro = hydro_unitless,
    extra_systems = extra_systems_unitless
)

# CIC specific parameter set
hydro_unitless_cic = Unitful.ustrip.(Kₕₛ),
Unitful.ustrip.(0.0 * B[:, :, maxS_index]), Unitful.ustrip.(fₑₓ),
Unitful.ustrip.(force_hydrostatic), wave_unitless, cic
p_unitless_cic = Hydrodynamics.HydroParams(
    inverse_mass = Unitful.ustrip.(inv_mass_inf),
    hydro = hydro_unitless_cic,
    extra_systems = extra_systems_unitless
)

# SS specific parameter set
hydro_unitless_ss = Unitful.ustrip.(Kₕₛ),
Unitful.ustrip.(0.0 * B[:, :, maxS_index]), Unitful.ustrip.(fₑₓ),
Unitful.ustrip.(force_hydrostatic), wave_unitless, ss
p_unitless_ss = Hydrodynamics.HydroParams(
    inverse_mass = Unitful.ustrip.(inv_mass_inf),
    hydro = hydro_unitless_ss,
    extra_systems = extra_systems_unitless
)

# Solve the hydrodynamic system
solution_point = Hydrodynamics.hydrodynamic_solver(
    Unitful.ustrip.(u₀), Unitful.ustrip.(ts), p_unitless, method = :point)
solution_cic = Hydrodynamics.hydrodynamic_solver(
    Unitful.ustrip.(u₀), Unitful.ustrip.(ts), p_unitless_cic, method = :cic)
solution_ss = Hydrodynamics.hydrodynamic_solver(
    Unitful.ustrip.(u₀_ss), Unitful.ustrip.(ts), p_unitless_ss, method = :ss)

# Visualize response
p1 = Plots.plot(solution_point, idxs = [1], label = ["Point"],
    title = "Excited hydrodynamic oscillator: Surge",
    xaxis = "Time", yaxis = "Position", lw = 2)
p1 = Plots.plot!(solution_cic, idxs = [1], label = ["CIC"], lw = 2, ls = :dash)
p1 = Plots.plot!(solution_ss, idxs = [1], label = ["SS"], lw = 2, ls = :dashdot)

p2 = Plots.plot(solution_point, idxs = [2], label = ["Point"],
    title = "Excited hydrodynamic oscillator: Heave",
    xaxis = "Time", yaxis = "Position", lw = 2)
p2 = Plots.plot!(solution_cic, idxs = [2], label = ["CIC"], lw = 2, ls = :dash)
p2 = Plots.plot!(solution_ss, idxs = [2], label = ["SS"], lw = 2, ls = :dashdot)

p3 = Plots.plot(solution_point, idxs = [3], label = ["Point"],
    title = "Excited hydrodynamic oscillator: Pitch",
    xaxis = "Time", yaxis = "Position", lw = 2)
p3 = Plots.plot!(solution_cic, idxs = [3], label = ["CIC"], lw = 2, ls = :dash)
p3 = Plots.plot!(solution_ss, idxs = [3], label = ["SS"], lw = 2, ls = :dashdot)

Plots.plot(p1, p2, p3, layout = (3, 1))

# Calculation of net energy over the simulation
function power_performance(pto_damping, p)
    T = eltype(pto_damping)

    (params, u₀, ts, dt, i_ramp, method) = p

    xpto, kpto, cpto = params.extra_systems[1].p
    cpto = diagm(pto_damping*ones(3))
    pto = Hydrodynamics.ExtraSystem{T}(
        n_state = params.extra_systems[1].n_state,
        force = params.extra_systems[1].force,
        p = (xpto, kpto, cpto)
    )

    mooring = Hydrodynamics.ExtraSystem{T}(
        n_state = params.extra_systems[2].n_state,
        force = params.extra_systems[2].force,
        rhs = params.extra_systems[2].rhs,
        p = params.extra_systems[2].p
    )

    params2 = Hydrodynamics.HydroParams{T}(
        inverse_mass = params.inverse_mass,
        hydro = params.hydro,
        extra_systems = [pto, mooring],
        u_control = params.u_control,
        method = params.method
    )

    u0_typed = convert.(T, u₀) # make initial state match Dual element type

    diff_eq_solution = Hydrodynamics.hydrodynamic_solver(u0_typed, ts, params2; method = method)

    # only absorb power in heave
    heave_ind = findall(x->x==3, Vector(dof))[1]
    heave_damping = pto_damping[heave_ind, heave_ind]
    heave_vel = diff_eq_solution[n_dof + heave_ind, :]
    power = heave_vel .^ 2 .* heave_damping
    energy = sum(power[i_ramp:end]) * dt
    return energy
end

function power_performance(pto_damping)
    power_performance(pto_damping, (
        p, u₀_unitless, ts_unitless, dt_unitless, i_ramp, method))
end

function power_performance_ss(pto_damping)
    power_performance(pto_damping, (
        p, u₀_ss_unitless, ts_unitless, dt_unitless, i_ramp, method))
end

### Single (point) frequency method - energy and gradient
### Calculate energy 
p = p_unitless
method = :point
u₀_unitless = Unitful.ustrip.(u₀)
u₀_ss_unitless = Unitful.ustrip.(u₀_ss)
ts_unitless = Unitful.ustrip.(ts)
dt_unitless = Unitful.ustrip.(dt)
energy = power_performance(cₚₜₒ_unitless) # lump appropriate unitful/unitless u₀, ts, dt into the power_performance parameter set, p 

### Calculate gradient of energy with respect to PTO damping 
# PTO_damping is not a scalar input here so there is no derivative call.
# derivative_fd = FD.derivative(power_performance, cₚₜₒ_unitless[1,1])

# Use gradient for multiple dofs. Also works with a single dof if pto_damping is a 1x1 array
gradient_fd = FiniteDiff.finite_difference_gradient(power_performance, cₚₜₒ_unitless)

gradient_ad_fd = ForwardDiff.gradient(power_performance, cₚₜₒ_unitless)
[gradient_fd gradient_ad_fd]

### CIC method - energy and gradient
### Calculate energy 
p = p_unitless_cic
method = :cic
energy_cic = power_performance(cₚₜₒ_unitless) # lump appropriate unitful/unitless u₀, ts, dt into the power_performance parameter set, p 

### Calculate gradient of energy with respect to PTO damping 
gradient_fd_cic = FiniteDiff.finite_difference_gradient(power_performance, cₚₜₒ_unitless)
# gradient_ad_fd_cic = ForwardDiff.gradient(power_performance, cₚₜₒ_unitless)

### SS method - energy and gradient
### Calculate energy 
p = p_unitless_ss
method = :ss
energy_ss = power_performance(cₚₜₒ_unitless,
    (p, Unitful.ustrip.(u₀_ss), Unitful.ustrip.(ts), Unitful.ustrip.(dt), i_ramp, method)) # lump appropriate unitful/unitless u₀, ts, dt into the power_performance parameter set, p 

### Calculate gradient of energy with respect to PTO damping
gradient_fd_ss = FiniteDiff.finite_difference_gradient(power_performance_ss, cₚₜₒ_unitless)
gradient_ad_fd_ss = ForwardDiff.gradient(power_performance_ss, cₚₜₒ_unitless)
[gradient_fd_ss gradient_ad_fd_ss]

[energy, energy_cic, energy_ss]

gradient_err = gradient_ad_fd - gradient_fd
gradient_err_ss = gradient_ad_fd_ss - gradient_fd_ss
grad_assessment = [gradient_err gradient_err_ss]

mae = sum(abs.(grad_assessment))/18

