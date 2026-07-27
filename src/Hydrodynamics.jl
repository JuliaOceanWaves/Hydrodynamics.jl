module Hydrodynamics

include("Bemio.jl")
include("core.jl")

export Bemio
export HydrodynamicSolution, HydroParams, ExtraSystem
export ramp_function, calculate_excitation_force, calculate_stiffness_force,
       calculate_damping_force, calculate_linear_force,
       calculate_radiation_force_convolution,
       calculate_radiation_force_ss, calculate_total_linear_hydro_forces
export hydrodynamic_oscillator, hydrodynamic_stepping, hydrodynamic_solver

end
