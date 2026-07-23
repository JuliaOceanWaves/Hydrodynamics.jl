module Bemio

using ExponentialUtilities
import ForwardDiff
import ImplicitAD
import LinearAlgebra
import NetCDF
using Statistics
import ToeplitzMatrices
import Unitful

abstract type AbstractBEMData end

# BEMData{T, TAX <: AbstractVector, ...} defines a family of BEMData types parameterized by T, TAX, etc. 
# One can have the type operate differently depending on what is parameterized by
# TODO - consider removing the frequency-independent parameters
# TODO - store dimensional or nondimensional data in BEMData?
# TODO - later on can define the frequency x direction parameters as matrices of WaveRealization type (even if length(direction)=1) across dofs

struct BEMIrf <: AbstractBEMData
    Kr::AbstractArray{Float64, 3}
    end_time::Float64
    dt::Float64

    function BEMIrf(
            Kr::AbstractArray{Float64, 3},
            end_time::Float64,
            dt::Float64
    )
        n_time_steps = end_time/dt + 1
        @assert n_time_steps == floor(n_time_steps)
        @assert size(Kr, 3) == Int64(n_time_steps)

        new(Kr, end_time, dt)
    end
end

struct BEMStateSpace <: AbstractBEMData
    A::AbstractArray{Float64, 2}
    B::AbstractArray{Float64, 2}
    C::AbstractArray{Float64, 2}
    D::AbstractArray{Float64, 2}
    irf::BEMIrf
    R2_fit::AbstractArray{Float64, 2}
    order::AbstractArray{Int64, 2}
    n_orders::Int64

    function BEMStateSpace(
            A::AbstractArray{Float64, 2},
            B::AbstractArray{Float64, 2},
            C::AbstractArray{Float64, 2},
            D::AbstractArray{Float64, 2},
            irf::BEMIrf,
            R2_fit::AbstractArray{Float64, 2},
            order::AbstractArray{Int64, 2}
    )
        n_dof = size(D, 1)
        total_order = size(A, 1)
        @assert ndims(A) == ndims(B) == ndims(C) == ndims(D) == ndims(R2_fit) ==
                ndims(order) == 2 "A, B, C, D, R2_fit, order do not all have two dimensions."
        @assert size(A) == (total_order, total_order) "State space variable A must be of size: total_order x total_order"
        @assert size(B) == (total_order, n_dof) "State space variable B must be of size: total_order x n_dof"
        @assert size(C) == (n_dof, total_order) "State space variable C must be of size: n_dof x total_order"
        @assert size(D) == (n_dof, n_dof) "State space variable D must be of size: n_dof x n_dof"
        @assert size(R2_fit) == (n_dof, n_dof) "State space variable R2_fit must be of size: n_dof x n_dof"
        @assert size(order) == (n_dof, n_dof) "State space variable order must be of size: n_dof x n_dof"

        return new(A, B, C, D, irf, R2_fit, order, sum(order))
    end
end

struct BEMData <: AbstractBEMData
    # Below is parameter validation. Can be based on the parametric types or not
    radiating_dof::AbstractVector
    influenced_dof::AbstractVector
    gravitational_constant::Float64
    density::Float64
    water_depth::Float64
    location::AbstractVector{Float64}                               # influenced_dof x (x, y, z)
    center_of_buoyancy::AbstractVector{Float64}                     # influenced_dof x (x, y, z)
    displaced_volume::Float64                                       # influenced_dof
    frequency::AbstractVector                                       # omega
    wave_direction::AbstractVector                                  # wave_dir
    excitation_coefficients::AbstractArray{Complex{Float64}, 3}     # influenced_dof x wave_dir x omega
    Froude_Krylov_coefficients::AbstractArray{Complex{Float64}, 3}  # influenced_dof x wave_dir x omega
    diffraction_coefficients::AbstractArray{Complex{Float64}, 3}    # influenced_dof x wave_dir x omega
    added_mass_coefficients::AbstractArray{Float64, 3}              # influenced_dof x radiating_dof x omega
    infinite_added_mass_coefficients::AbstractArray{Float64, 2}        # influenced_dof x radiating_dof
    radiation_damping_coefficients::AbstractArray{Float64, 3}       # influenced_dof x radiating_dof x omega
    hydrostatic_stiffness::AbstractArray{Float64, 2}                # influenced_dof x radiating_dof

    function BEMData(
            influenced_dof::AbstractVector,
            gravitational_constant,
            density,
            water_depth,
            location::AbstractVector{Float64},
            center_of_buoyancy::AbstractVector{Float64},
            displaced_volume::Float64,
            frequency::AbstractVector,
            wave_direction::AbstractVector,
            excitation_coefficients::AbstractArray{Complex{Float64}, 3},
            Froude_Krylov_coefficients::AbstractArray{Complex{Float64}, 3},
            diffraction_coefficients::AbstractArray{Complex{Float64}, 3},
            added_mass_coefficients::AbstractArray{Float64, 3},
            radiation_damping_coefficients::AbstractArray{Float64, 3},
            hydrostatic_stiffness::AbstractArray{Float64, 2},
            infinite_added_mass_coefficients::AbstractArray{Float64, 2} = added_mass_coefficients[:, :, end]
    )
        # @assert size(rotational_dof_flag) == size(influenced_dof) "Rotational DOF flags not the same size as influenced_dof!"
        @assert length(center_of_buoyancy) == 3 "Center of buoyancy not a vector of x, y, z coordinates!"
        @assert length(location) == 3 "Location not a vector of x, y, z coordinates!"

        n_dof = length(influenced_dof)
        n_freq = length(frequency)
        n_dir = length(wave_direction)
        @assert size(excitation_coefficients) == (n_dof, n_dir, n_freq) "Excitation coefficients not of size: n_dof x n_freq x n_dir!"
        @assert size(Froude_Krylov_coefficients) == (n_dof, n_dir, n_freq) "Froude Krylov coefficients not of size: n_dof x n_freq x n_dir!"
        @assert size(diffraction_coefficients) == (n_dof, n_dir, n_freq) "Diffraction coefficients not of size: n_dof x n_freq x n_dir!"
        @assert size(added_mass_coefficients) == (n_dof, n_dof, n_freq) "Added mass coefficients not of size: n_dof x n_dof x n_freq"
        @assert size(radiation_damping_coefficients) == (n_dof, n_dof, n_freq) "Radiation damping coefficients not of size: n_dof x n_dof x n_freq"
        @assert size(hydrostatic_stiffness) == (n_dof, n_dof) "Hydrostatic stiffness coefficients not of size: n_dof x n_dof!"

        return new(
            influenced_dof,
            influenced_dof,
            gravitational_constant,
            density,
            water_depth,
            location,
            center_of_buoyancy,
            displaced_volume,
            frequency,
            wave_direction,
            excitation_coefficients,
            Froude_Krylov_coefficients,
            diffraction_coefficients,
            added_mass_coefficients,
            infinite_added_mass_coefficients,
            radiation_damping_coefficients,
            hydrostatic_stiffness
        )
    end
end

function select_BEMData_dofs(bem::BEMData,
        dof::Union{AbstractVector{<:Integer}, AbstractRange{<:Integer}})::BEMData
    BEMData(
        bem.influenced_dof[dof],
        bem.gravitational_constant,
        bem.density,
        bem.water_depth,
        bem.location,
        bem.center_of_buoyancy,
        bem.displaced_volume,
        bem.frequency,
        bem.wave_direction,
        bem.excitation_coefficients[dof, :, :],
        bem.Froude_Krylov_coefficients[dof, :, :],
        bem.diffraction_coefficients[dof, :, :],
        bem.added_mass_coefficients[dof, dof, :],
        bem.radiation_damping_coefficients[dof, dof, :],
        bem.hydrostatic_stiffness[dof, dof],
        bem.infinite_added_mass_coefficients[dof, dof]
    )
end

function _vector_to_complex(data)
    return data[:, :, :, 1] + 1im * data[:, :, :, 2]
end

function read_capytaine(
        filename::String; dofs = nothing, center_of_gravity::AbstractVector = [
            0.0, 0.0, 0.0],
        center_of_buoyancy::AbstractVector = center_of_gravity,
        volume::AbstractFloat = 0.0)::BEMData
    temp = NetCDF.ncread(filename, "radiating_dof")
    radiating_dof = [String(temp[:, j]) for j in axes(temp, 2)]

    temp = NetCDF.ncread(filename, "influenced_dof")
    influenced_dof = [String(temp[:, j]) for j in axes(temp, 2)]

    gravity = NetCDF.ncread(filename, "g")[1]
    density = NetCDF.ncread(filename, "rho")[1]

    frequency = NetCDF.ncread(filename, "omega")
    frequency = round.(frequency, digits = 14) # round off errors from reading the nc file
    wave_direction = NetCDF.ncread(filename, "wave_direction")
    depth = NetCDF.ncread(filename, "water_depth")[1]

    try
        center_of_gravity = NetCDF.ncread(filename, "center_of_gravity")
    catch
    end
    try
        center_of_buoyancy = NetCDF.ncread(filename, "center_of_buoyancy")
    catch
    end
    try
        volume = NetCDF.ncread(filename, "volume")
    catch
    end

    excitation_coefficients = _vector_to_complex(NetCDF.ncread(filename, "excitation_force")) # Dimensions: influenced_dof wave_dir omega complex
    Froude_Krylov_coefficients = _vector_to_complex(NetCDF.ncread(filename, "Froude_Krylov_force")) # Dimensions: influenced_dof wave_dir omega complex
    diffraction_coefficients = _vector_to_complex(NetCDF.ncread(filename, "diffraction_force")) # Dimensions: influenced_dof wave_dir omega complex

    # Capytaine uses an inverted x coordinate: +phase velocity in -x. Hydrodynamics.jl expects +phase velocity in +x
    excitation_coefficients = conj.(excitation_coefficients)
    Froude_Krylov_coefficients = conj.(Froude_Krylov_coefficients)
    diffraction_coefficients = conj.(diffraction_coefficients)

    added_mass_coefficients = NetCDF.ncread(filename, "added_mass") # Dimensions: influenced_dof radiating_dof omega
    infinite_added_mass_coefficients = added_mass_coefficients[:, :, end]
    radiation_damping_coefficients = NetCDF.ncread(filename, "radiation_damping") # Dimensions: influenced_dof radiating_dof omega
    hydrostatic_stiffness = NetCDF.ncread(filename, "hydrostatic_stiffness")' # Dimensions: radiating_dof influenced_dof --> influenced_dof radiating_dof

    if isnothing(dofs)
        dofs = 1:1:length(influenced_dof)
    end

    return BEMData(influenced_dof[dofs],
        gravity,
        density,
        depth,
        center_of_gravity,
        center_of_buoyancy,
        volume,
        frequency,
        wave_direction,
        excitation_coefficients[dofs, :, :],
        Froude_Krylov_coefficients[dofs, :, :],
        diffraction_coefficients[dofs, :, :],
        added_mass_coefficients[dofs, dofs, :],
        radiation_damping_coefficients[dofs, dofs, :],
        hydrostatic_stiffness[dofs, dofs],
        infinite_added_mass_coefficients[dofs, dofs]
    )
end

function radiation_impulse_response_function(
        rd_raw::Array, w_raw::Vector; w_max = 20.0, t = collect(0:0.1:60)')::BEMIrf
    @assert length(size(t[:])) == 1
    t = reshape(t, 1, length(t))

    dt = diff(t; dims = 2)
    @assert all(abs.(dt[1] .- dt) .< 1e-12)

    nt = length(t)

    # cut off at the frequency limit
    i_w_end = argmin(abs.(w_raw .- w_max))
    w = w_raw[1:i_w_end]
    rd = rd_raw[:, :, 1:i_w_end]
    nw = length(w)

    # Reshape arrays to enable element-wise multiplication without loops and overwriting initialized arrays
    c = reshape(cos.(w * t), 1, 1, nw, nt)
    dw = reshape(diff(w), 1, 1, nw - 1)
    integrand = rd .* c
    integral = sum(
        (integrand[:, :, 1:(end - 1), :] .+ integrand[:, :, 2:end, :]) .* 0.5 .* dw;
        dims = [3])[:, :, 1, :]
    Kᵣ = 2 / pi .* integral

    BEMIrf(Kᵣ, t[end], dt[1])
end

function _radiation_state_space_realization(Kᵣ, dt, max_order, R2t;
        orders = nothing, verbose = true)
    if max_order < 1
        throw(ArgumentError("max_order must be at least 1"))
    end

    n_time = size(Kᵣ, 3)
    if n_time < max_order + 2
        throw(ArgumentError("radiation_state_space requires at least " *
                            string(max_order+2) * " IRF time samples"))
    end

    T = eltype(Kᵣ)
    ss_A_by_dof = zeros(T, size(Kᵣ, 1), size(Kᵣ, 2), max_order, max_order)
    ss_B_by_dof = zeros(T, size(Kᵣ, 1), size(Kᵣ, 2), max_order, 1)
    ss_C_by_dof = zeros(T, size(Kᵣ, 1), size(Kᵣ, 2), 1, max_order)
    ss_D_by_dof = zeros(T, size(Kᵣ)[1:2])
    ss_K_by_dof = zeros(T, size(Kᵣ)[1:3])
    ss_R2_by_dof = zeros(T, size(Kᵣ)[1:2])
    ss_order_by_dof = isnothing(orders) ? zeros(Int64, size(Kᵣ)[1:2]) : Int64.(orders)
    if size(ss_order_by_dof) != size(Kᵣ)[1:2]
        throw(DimensionMismatch("orders must match the first two dimensions of Kᵣ"))
    end

    if !isnothing(orders) && (any(ss_order_by_dof .< 1) ||
        any(ss_order_by_dof .> max_order))
        throw(ArgumentError("orders must be between 1 and min(max_order, length(t)-2)"))
    end

    if verbose
        print("State space calculation: \n")
    end
    for i in axes(Kᵣ, 1), j in axes(Kᵣ, 2)

        irf_K = Kᵣ[i, j, :]
        R2i = LinearAlgebra.norm(irf_K .- mean(irf_K))
        y = dt .* irf_K
        n = length(y)
        h = ToeplitzMatrices.Hankel([y[2:end]; zeros(T, n - 1)], (n - 1, n - 1))
        u, svh, v = LinearAlgebra.svd(h)

        # If dof specific orders are input, use those. 
        # If not, run up to the max_order
        order_range = isnothing(orders) ? (1:max_order) :
                      (ss_order_by_dof[i, j]:ss_order_by_dof[i, j])
        order = last(order_range)

        # Define variables so they pass outside of the loop 
        R2 = zero(T)
        ac = zeros(T, order, order)
        bc = zeros(T, order)
        cc = zeros(T, 1, order)
        dc = zero(T)
        ss_K_each_dof = zeros(T, n)
        for m in order_range
            order = m
            u1 = u[1:(n - 2), 1:order]
            v1 = v[1:(n - 2), 1:order]
            u2 = u[2:(n - 1), 1:order]
            sqs = sqrt.(svh[1:order])
            ubar = u1' * u2

            a = ubar .* ((1 ./ sqs) * sqs')
            b = v1[1, :] .* sqs
            c = (u1[1, :] .* sqs)'
            d = y[1]

            eye = LinearAlgebra.I(order)
            iidd = dt / 2 * (eye + a)
            ac_transpose = similar(a)
            for col in 1:order
                ac_transpose[
                    :, col] = ImplicitAD.implicit_linear(
                    transpose(iidd), collect(transpose(a - eye)[:, col]))
            end
            ac = transpose(ac_transpose)
            bc = dt .* ImplicitAD.implicit_linear(iidd, b)
            cc = reshape(ImplicitAD.implicit_linear(transpose(iidd), vec(c)), 1, order)
            dc = d - dt / 2 * (cc * b)[1]

            ss_K_each_dof = zeros(T, n)
            for k in 1:n
                term = ac * dt * (k - 1)
                exponential!(term)
                ss_K_each_dof[k] = ((cc * term) * bc)[1]
            end

            # Calculate R2 for the state space fit of Kᵣ. Check if above the R2 threshold
            R2 = 1 - (LinearAlgebra.norm(irf_K - ss_K_each_dof) / R2i)^2
            if R2 >= R2t
                break
            end
        end

        ss_A_by_dof[i, j, 1:order, 1:order] = ac
        ss_B_by_dof[i, j, 1:order, 1] = bc
        ss_C_by_dof[i, j, 1, 1:order] = cc
        ss_D_by_dof[i, j] = dc
        ss_K_by_dof[i, j, :] = ss_K_each_dof
        ss_R2_by_dof[i, j] = R2
        ss_order_by_dof[i, j] = Int64(order)
        if verbose
            print("dof: ", i, " ", j, "; order: ", order, "\n")
        end
    end

    total_order = sum(ss_order_by_dof)
    ss_A = zeros(T, total_order, total_order)
    ss_B = zeros(T, total_order, size(Kᵣ, 2))
    ss_C = zeros(T, size(Kᵣ, 1), total_order)

    order_count = 1
    for i in axes(ss_A_by_dof, 1), j in axes(ss_A_by_dof, 2)

        order = ss_order_by_dof[i, j]
        o1 = order_count
        o2 = order_count + order - 1
        ss_A[o1:o2, o1:o2] = ss_A_by_dof[i, j, 1:order, 1:order]
        ss_B[o1:o2, j] = ss_B_by_dof[i, j, 1:order, 1]
        ss_C[i, o1:o2] = ss_C_by_dof[i, j, 1, 1:order]
        order_count = o2 + 1
    end

    return ss_A, ss_B, ss_C, ss_D_by_dof, ss_K_by_dof, ss_R2_by_dof, ss_order_by_dof
end

function radiation_state_space(irf::BEMIrf, max_order = 10, R2t = 0.95;
        orders = nothing, ad_mode = "cfd", verbose = true)::BEMStateSpace
    if ndims(irf.Kr) == 1
        K_values = Unitful.ustrip.(reshape(irf.Kr, 1, 1, length(irf.Kr)))
    elseif ndims(irf.Kr) == 3
        K_values = Unitful.ustrip.(irf.Kr)
    else
        throw(ArgumentError("radiation_state_space expects a vector or a 3D IRF array"))
    end
    K_values_primal = similar(K_values, Float64)
    for idx in eachindex(K_values)
        value = K_values[idx]
        while value isa ForwardDiff.Dual
            value = ForwardDiff.value(value)
        end
        K_values_primal[idx] = Float64(value)
    end

    if isnothing(orders)
        _, _,
        _,
        _,
        _,
        _,
        ss_order_by_dof = _radiation_state_space_realization(
            K_values_primal, irf.dt, max_order, R2t; verbose)
    else
        ss_order_by_dof = Int64.(orders)
    end

    # The SVD-based realization is differentiated piecewise: model orders are
    # selected from primal values, then held fixed for the provided AD rule.
    p = (size(K_values_primal), irf.dt, max_order, R2t, ss_order_by_dof)
    flat = ImplicitAD.provide_rule(
        function (x, p)
            dims, dt_values, max_order, R2t, fixed_orders = p
            ss_A, ss_B,
            ss_C,
            ss_D,
            ss_K,
            ss_R2,
            _ = _radiation_state_space_realization(
                reshape(x, dims), dt_values, max_order, R2t;
                orders = fixed_orders, verbose = false)
            return vcat(vec(ss_A), vec(ss_B), vec(ss_C), vec(ss_D), vec(ss_K), vec(ss_R2))
        end,
        collect(vec(K_values)), p; mode = ad_mode)

    n_influenced, n_radiating, n_time = size(K_values_primal)
    total_order = sum(ss_order_by_dof)
    next_index = 1

    ss_A_len = total_order * total_order
    ss_A = reshape(flat[next_index:(next_index + ss_A_len - 1)], total_order, total_order)
    next_index += ss_A_len

    ss_B_len = total_order * n_radiating
    ss_B = reshape(flat[next_index:(next_index + ss_B_len - 1)], total_order, n_radiating)
    next_index += ss_B_len

    ss_C_len = n_influenced * total_order
    ss_C = reshape(flat[next_index:(next_index + ss_C_len - 1)], n_influenced, total_order)
    next_index += ss_C_len

    ss_D_len = n_influenced * n_radiating
    ss_D = reshape(flat[next_index:(next_index + ss_D_len - 1)], n_influenced, n_radiating)
    next_index += ss_D_len

    ss_K_len = n_influenced * n_radiating * n_time
    ss_K = reshape(flat[next_index:(next_index + ss_K_len - 1)], size(K_values_primal))
    next_index += ss_K_len
    ss_K = BEMIrf(ss_K, irf.end_time, irf.dt)

    ss_R2_len = n_influenced * n_radiating
    ss_R2 = reshape(flat[next_index:(next_index + ss_R2_len - 1)], n_influenced, n_radiating)

    return BEMStateSpace(ss_A, ss_B, ss_C, ss_D, ss_K, ss_R2, ss_order_by_dof)
end

function alternate_Ainf(bem::BEMData, irf::BEMIrf)
    nw = length(bem.frequency)
    nt = Int64(irf.end_time/irf.dt+1)
    w = reshape(bem.frequency, 1, 1, 1, nw)

    tCIC = reshape(Vector(0:irf.dt:irf.end_time), 1, 1, nt)

    integrand = irf.Kr .* sin.(w .* tCIC) # nDOF, nDOF, nt, nw
    integral = sum(
        (integrand[:, :, 1:(end - 1), :] .+ integrand[:, :, 2:end, :]) .* 0.5 .* irf.dt;
        dims = [3]) # nDOF, nDOF, 1, nw
    ainf_temp = bem.added_mass_coefficients + (integral ./ w)[:, :, 1, :] # nDOF, nDOF, nw
    ainf = mean(ainf_temp; dims = [3])[:, :, 1] # nDOF, nDOF
end

function alternate_Ainf!(bem::BEMData, irf::BEMIrf)
    bem.infinite_frequency_added_mass .= alternate_Ainf(bem, irf)
    return nothing
end

end
