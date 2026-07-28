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

# Wave conditions
Hₛ₀ = 1.0
Tₑ = 6.0

# Time-domain set-up
t0 = 0.0
tr = Tₑ * 1.0
tf = Tₑ * 10.0
dt = 1.0e-2
ts = collect(t0:dt:tf)
nt = length(ts)
i_ramp = Int64(tr / dt + 1)

# Hydrodynamics
bem_file = joinpath(@__DIR__, "data", "rm3.nc")

A = NetCDF.ncread(bem_file, "added_mass")[dof, dof, :]      # influenced_dof radiating_dof omega
B = NetCDF.ncread(bem_file, "radiation_damping")[dof, dof, :] # influenced_dof radiating_dof omega
Kₕₛ = NetCDF.ncread(bem_file, "hydrostatic_stiffness")[dof, dof]' # radiating_dof influenced_dof --> influenced_dof radiating_dof
fₑₓ = NetCDF.ncread(bem_file, "excitation_force")[dof, :, :, :]  # influenced_dof wave_dir omega complex
fₑₓ[:, :, :, 2] = -1 .* fₑₓ[:, :, :, 2] # Convert Capytaine's convention to one in which the wave propagates in +X

ω = NetCDF.ncread(bem_file, "omega")
dω = ω[2] - ω[1]
nω = length(ω)

f = ω/(2*pi)
df = dω/(2*pi)

# Hydrostatics parameters (simplified / example values)
cg = [0.0, 0.0, -0.72]
volume = 725.8330
cb = [0.0, 0.0, -1.2927]

# Convolution integral calculation
Kᵣ, tᵣ = Hydrodynamics.Bemio.radiation_irf(B, ω; w_max = 5.0, t = collect(0:dt:30)')
Ainf = A[:, :, end]
# Ainf = Hydrodynamics.Bemio.alternate_Ainf(Kᵣ, hydro.am, hydro.w, tᵣ)
cic = (Kᵣ, tᵣ)

# State space formation
Aᵣ, Bᵣ, Cᵣ, Dᵣ, Kₛₛ, R²ₛₛ,
orderₛₛ = Hydrodynamics.Bemio.radiation_state_space(Kᵣ, tᵣ, 8, 0.95)
nₛₛ = sum(orderₛₛ)
s₀ = zeros(nₛₛ)
ss = (Aᵣ, Bᵣ, Cᵣ, Dᵣ, nₛₛ)

S = WaveSpectra.ParametricSpectra.spectrum_pierson_moskowitz(f*u"s^-1", Hₛ₀*u"m", Tₑ*u"s")
S = Unitful.ustrip.(S)
ϕ = rand(Random.Xoshiro(0), Float64, size(S)) * 2 * pi
maxS_index = argmax(S)

# Calculate and visualize the wave elevation
ramp_timeseries = zeros(length(ts))
for i in 1:length(ts)
    ramp_timeseries[i] = Hydrodynamics.ramp_function(t0, tr, ts[i])
end

elevation = sum(sqrt.(2 .* S .* df) .* cos.(ω .* ts' .+ ϕ), dims = 1)'
ramped_elevation = elevation .* ramp_timeseries
Plots.plot(
    ts, elevation, xlabel = "Time (s)", ylabel = "Wave elevation (m)", label = "No ramp")
Plots.plot!(ts, ramped_elevation, xlabel = "Time (s)",
    ylabel = "Wave elevation (m)", label = "Ramped", ls = :dash)

# Body properties
m = 725833.0
Ixx = [20907301.0, 21306090.66, 37085481.11]
Ixy = [1e6, 1e6, 1e5]
I = diagm(Ixx)
I[1, 2:3] = I[2:3, 1] = Ixy[1:2]
I[2, 3] = I[3, 2] = Ixy[3]
body_mass = [diagm(repeat([m], 3)) zeros(3, 3); zeros(3, 3) I]

# Coefficients for the form: ẍ + c * ẋ + k * x = 0
mass = body_mass[dof, dof] .+ A[:, :, maxS_index]
inv_mass = inv(mass)

# Adjusted mass calculations
mass_inf = body_mass[dof, dof] .+ Ainf
inv_mass_inf = inv(mass_inf)

g = zeros(6)
g[3] = -NetCDF.ncread(bem_file, "g")[1]
force_gravity = diag(body_mass) .* g
rho = NetCDF.ncread(bem_file, "rho")[1]
force_buoyancy = - rho * g * volume
CGCB = cb - cg
force_buoyancy[4:6] = cross(CGCB, force_buoyancy[1:3])
force_hydrostatic = force_gravity[dof] + force_buoyancy[dof]

# Initial conditions
x₀ = zeros(size(dof))
dx₀ = zeros(size(dof))
u₀ = [x₀; dx₀]
u₀_ss = [x₀; dx₀; s₀]

# PTO parameters
kₚₜₒ = diagm(zeros(size(dof)))
cₚₜₒ = diagm(1.0e5 .* ones(size(dof)))
pto = (x₀, kₚₜₒ, cₚₜₒ)

# Mooring parameters
kₘ = diagm(zeros(size(dof)))
cₘ = diagm(zeros(size(dof)))
mooring = (x₀, kₘ, cₘ)

function linear_extra_force(t, platform_state, system_state, u; p = nothing)
    n_dof = Int(length(platform_state) / 2)
    x = platform_state[1:n_dof]
    dx = platform_state[(n_dof + 1):(2n_dof)]
    return Hydrodynamics.calculate_linear_force(dx, x, p)
end

pto_system = Hydrodynamics.ExtraSystem{Float64}(
    n_state = 0,
    force = linear_extra_force,
    p = pto
)

mooring_system = Hydrodynamics.ExtraSystem{Float64}(
    n_state = 0,
    force = linear_extra_force,
    p = mooring
)

extra_systems = [pto_system, mooring_system]

cic_system = Hydrodynamics.ExtraSystem(
    n_state = 0,
    rhs = nothing,
    force = Hydrodynamics.calculate_radiation_force_convolution,
    p = (Kᵣ, tᵣ)
)

ss_system = Hydrodynamics.ExtraSystem(
    n_state = ss[5],
    rhs = Hydrodynamics.radiation_ss_rhs,
    force = Hydrodynamics.calculate_radiation_force_ss,
    p = ss
)

# Parameter groups
wave = (ω, ϕ, S, df, t0, tr)
hydro = (Kₕₛ, B[:, :, maxS_index], fₑₓ, force_hydrostatic, wave) # optionally cic or ss tuples on the end
p_point = Hydrodynamics.HydroParams(
    inverse_mass = inv_mass,
    hydro = hydro,
    extra_systems = extra_systems
)

hydro_cic = (Kₕₛ, 0.0 .* B[:, :, maxS_index], fₑₓ, force_hydrostatic, wave, cic)
p_cic = Hydrodynamics.HydroParams(
    inverse_mass = inv_mass_inf,
    hydro = hydro_cic,
    extra_systems = [extra_systems, cic_system]
)

hydro_ss = (Kₕₛ, 0.0 .* B[:, :, maxS_index], fₑₓ, force_hydrostatic, wave, ss)
p_ss = Hydrodynamics.HydroParams(
    inverse_mass = inv_mass_inf,
    hydro = hydro_ss,
    extra_systems = [extra_systems, ss_system]
)

# Solve the hydrodynamic system (unitless inputs)
solution_point = Hydrodynamics.hydrodynamic_solver(u₀, ts, p_point; method = :point)
solution_cic = Hydrodynamics.hydrodynamic_solver(u₀, ts, p_cic; method = :cic)
solution_ss = Hydrodynamics.hydrodynamic_solver(u₀_ss, ts, p_ss; method = :ss)

# Visualize response
p1 = Plots.plot(solution_point, idxs = [1], label = ["Point"],
    title = "Excited hydrodynamic oscillator: Surge",
    xaxis = "Time", yaxis = "Position", lw = 2)
Plots.plot!(p1, solution_cic, idxs = [1], label = ["CIC"], lw = 2, ls = :dash)
Plots.plot!(p1, solution_ss, idxs = [1], label = ["SS"], lw = 2, ls = :dashdot)

p2 = Plots.plot(solution_point, idxs = [2], label = ["Point"],
    title = "Excited hydrodynamic oscillator: Heave",
    xaxis = "Time", yaxis = "Position", lw = 2)
Plots.plot!(p2, solution_cic, idxs = [2], label = ["CIC"], lw = 2, ls = :dash)
Plots.plot!(p2, solution_ss, idxs = [2], label = ["SS"], lw = 2, ls = :dashdot)

p3 = Plots.plot(solution_point, idxs = [3], label = ["Point"],
    title = "Excited hydrodynamic oscillator: Pitch",
    xaxis = "Time", yaxis = "Position", lw = 2)
Plots.plot!(p3, solution_cic, idxs = [3], label = ["CIC"], lw = 2, ls = :dash)
Plots.plot!(p3, solution_ss, idxs = [3], label = ["SS"], lw = 2, ls = :dashdot)

Plots.plot(p1, p2, p3, layout = (3, 1))

# Calculation of net energy over the simulation
function power_performance(pto_damping, p)
    (params, u₀_local, ts_local, dt_local, i_ramp_local, method) = p

    # ensure element type matches PTO damping so Duals propagate
    T = eltype(pto_damping)

    # construct diagonal PTO matrix with element type T
    xpto, kpto, _ = params.extra_systems[1].p
    cpto = Diagonal([pto_damping[i, i] for i in 1:3])

    pto = Hydrodynamics.ExtraSystem{T}(
        n_state = params.extra_systems[1].n_state,
        force = params.extra_systems[1].force,
        rhs = params.extra_systems[1].rhs,
        p = (xpto, kpto, cpto)
    )

    # rebuild mooring ExtraSystem with same parametric element type
    m_old = params.extra_systems[2]
    mooring = Hydrodynamics.ExtraSystem{T}(
        n_state = m_old.n_state,
        force = m_old.force,
        rhs = m_old.rhs,
        p = m_old.p
    )

    if params.method == :point
        new_systems = [pto, mooring]
    else
        es_3_old = params.extra_systems[3]
        es_3 = Hydrodynamics.ExtraSystem{T}(
            n_state = es_3_old.n_state,
            force = es_3_old.force,
            rhs = es_3_old.rhs,
            p = es_3_old.p
        )
        new_systems = [pto, mooring, es_3]
    end

    # convert initial state and inverse mass to element type T
    u₀_local = convert.(T, u₀_local)
    inv_mass_T = convert.(T, params.inverse_mass)

    params2 = Hydrodynamics.HydroParams{T}(
        inverse_mass = inv_mass_T,
        hydro = params.hydro,
        extra_systems = new_systems,
        u_control = params.u_control,
        method = params.method
    )

    diff_eq_solution = Hydrodynamics.hydrodynamic_solver(u₀_local, ts_local, params2; method = method)

    # only absorb power in heave
    heave_ind = findall(x->x==3, Vector(dof))[1]
    heave_damping = pto_damping[heave_ind, heave_ind]
    heave_vel = diff_eq_solution[n_dof + heave_ind, :]
    power = heave_vel .^ 2 .* heave_damping
    energy = sum(power[i_ramp_local:end]) * dt_local
    return energy
end

function power_performance(pto_damping)
    power_performance(pto_damping, (
        p, u₀, ts, dt, i_ramp, method))
end

### Single (point) frequency method - energy and gradient
### Calculate energy 
p = p_point
method = :point
# energy = power_performance(cₚₜₒ, (p, u₀, ts, dt, i_ramp, method))
energy = power_performance(cₚₜₒ)

### Calculate gradient of energy with respect to PTO damping 
gradient_fd = FiniteDiff.finite_difference_gradient(power_performance, cₚₜₒ)
gradient_ad_fd = ForwardDiff.gradient(power_performance, cₚₜₒ)
[gradient_fd gradient_ad_fd]

### CIC method - energy and gradient
p = p_cic
method = :cic
energy_cic = power_performance(cₚₜₒ)

gradient_fd_cic = FiniteDiff.finite_difference_gradient(power_performance, cₚₜₒ)
# gradient_ad_fd_cic = ForwardDiff.gradient(power_performance, cₚₜₒ)
# [gradient_fd_cic gradient_ad_fd_cic]

### SS method - energy and gradient
p = p_ss
method = :ss
energy_ss = power_performance(cₚₜₒ)

gradient_fd_ss = FiniteDiff.finite_difference_gradient(power_performance, cₚₜₒ)
gradient_ad_fd_ss = ForwardDiff.gradient(power_performance, cₚₜₒ)

[energy, energy_cic, energy_ss]

gradient_err = gradient_ad_fd - gradient_fd
gradient_err_ss = gradient_ad_fd_ss - gradient_fd_ss
grad_assessment = [gradient_err gradient_err_ss]

mae = sum(abs.(grad_assessment)) / length(vec(grad_assessment))
