module Bemio

using ExponentialUtilities
import ForwardDiff
import ImplicitAD
import LinearAlgebra
import NetCDF
using Statistics
import ToeplitzMatrices
import Unitful

struct Hydro
    radiating_dof::Any
    influenced_dof::Any
    g::Any
    rho::Any
    w::Any
    period::Any
    wave_direction::Any
    wavelength::Any
    depth::Any
    volume::Any
    cb::Any
    ex::Any
    fk::Any
    di::Any
    am::Any
    rd::Any
    khs::Any
end

function read_capytaine(filename::String)::Hydro
    radiating_dof = NetCDF.ncread(filename, "radiating_dof")
    influenced_dof = NetCDF.ncread(filename, "influenced_dof")
    g = NetCDF.ncread(filename, "g")
    rho = NetCDF.ncread(filename, "rho")

    w = NetCDF.ncread(filename, "omega")
    period = NetCDF.ncread(filename, "period")
    wave_direction = NetCDF.ncread(filename, "wave_direction")
    wavelength = NetCDF.ncread(filename, "wavelength")
    depth = NetCDF.ncread(filename, "water_depth")
    # volume = NetCDF.ncread(filename, "volume")
    volume = 725.8330 # TODO - hard coded for the RM3 float bc my sample file doesn't have this parameter

    # cb = NetCDF.ncread(filename, "center_of_buoyancy")
    cb = [0.0, 0.0, -1.2927] # TODO - hard coded for the RM3 float bc my sample file doesn't have this parameter

    ex = NetCDF.ncread(filename, "excitation_force") # Dimensions: influenced_dof wave_dir omega complex
    fk = NetCDF.ncread(filename, "Froude_Krylov_force") # Dimensions: influenced_dof wave_dir omega complex
    di = NetCDF.ncread(filename, "diffraction_force") # Dimensions: influenced_dof wave_dir omega complex
    am = NetCDF.ncread(filename, "added_mass") # Dimensions: influenced_dof radiating_dof omega
    # ainf = am[:, :, end]
    rd = NetCDF.ncread(filename, "radiation_damping") # Dimensions: influenced_dof radiating_dof omega
    khs = NetCDF.ncread(filename, "hydrostatic_stiffness")' # Dimensions: radiating_dof influenced_dof --> influenced_dof radiating_dof

    # ainf, ra_t, ra_w = radiationIRF!()
    return Hydro(radiating_dof, influenced_dof, g, rho, w, period, wave_direction,
        wavelength, depth, volume, cb, ex, fk, di, am, rd, khs)
end

function radiation_irf(
        rd_raw::Array, w_raw::Vector; w_max = 20.0, t = collect(0:0.1:60)')
    # cut off at the frequency limit
    i_w_end = argmin(abs.(w_raw .- w_max))
    w = w_raw[1:i_w_end]
    rd = rd_raw[:, :, 1:i_w_end]
    nw = length(w)

    # timeseries array
    nt = length(t)

    # Reshape arrays to enable element-wise multiplication without loops and overwriting initialized arrays
    c = reshape(cos.(w * t), 1, 1, nw, nt)
    dw = reshape(diff(w), 1, 1, nw - 1)
    integrand = rd .* c
    integral = sum(
        (integrand[:, :, 1:(end - 1), :] .+ integrand[:, :, 2:end, :]) .* 0.5 .* dw;
        dims = [3])[:, :, 1, :]
    Kᵣ = 2 / pi .* integral
    tᵣ = reshape(t, 1, 1, size(Kᵣ, 3))
    Kᵣ, tᵣ
end

function _radiation_state_space_realization(Kᵣ, tᵣ, max_order, R2t;
        orders = nothing, verbose = true)
    if max_order < 1
        throw(ArgumentError("max_order must be at least 1"))
    end

    dt = tᵣ[2] - tᵣ[1]
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

    print("State space calculation:")
    for i in axes(Kᵣ, 1), j in axes(Kᵣ, 2)

        print("dof: ", i, ", ", j, " complete.")
        irf_K = Kᵣ[i, j, :]
        R2i = LinearAlgebra.norm(irf_K .- mean(irf_K))
        y = dt .* irf_K
        n = length(y)
        h = ToeplitzMatrices.Hankel([y[2:end]; zeros(T, n - 1)], (n - 1, n - 1))
        u, svh, v = LinearAlgebra.svd(h)

        order_range = isnothing(orders) ? (1:max_order) :
                      (ss_order_by_dof[i, j]:ss_order_by_dof[i, j])
        order = last(order_range)
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
            solve_matrix = dt / 2 * (eye + a)
            ac_transpose = similar(a)
            for col in 1:order
                ac_transpose[
                    :, col] = ImplicitAD.implicit_linear(
                    transpose(solve_matrix), collect(transpose(a - eye)[:, col]))
            end
            ac = transpose(ac_transpose)
            bc = dt .* ImplicitAD.implicit_linear(solve_matrix, b)
            cc = reshape(ImplicitAD.implicit_linear(transpose(solve_matrix), vec(c)), 1, order)
            dc = d - dt / 2 * (cc * b)[1]

            ss_K_each_dof = zeros(T, n)
            for k in 1:n
                term = ac * dt * (k - 1)
                exponential!(term)
                ss_K_each_dof[k] = ((cc * term) * bc)[1]
            end

            R2 = R2i > 0 ? 1 - (LinearAlgebra.norm(irf_K - ss_K_each_dof) / R2i)^2 : one(T)
            if isnothing(orders) && R2 >= R2t
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

function radiation_state_space(Kᵣ, tᵣ, max_order = 10, R2t = 0.95;
        orders = nothing, ad_mode = "cfd", verbose = true)
    if ndims(Kᵣ) == 1
        K_values = Unitful.ustrip.(reshape(Kᵣ, 1, 1, length(Kᵣ)))
    elseif ndims(Kᵣ) == 3
        K_values = Unitful.ustrip.(Kᵣ)
    else
        throw(ArgumentError("radiation_state_space expects a vector or a 3D IRF array"))
    end
    t_values = vec(Unitful.ustrip.(tᵣ))
    K_values_primal = similar(K_values, Float64)
    for idx in eachindex(K_values)
        value = K_values[idx]
        while value isa ForwardDiff.Dual
            value = ForwardDiff.value(value)
        end
        K_values_primal[idx] = Float64(value)
    end
    t_values_primal = similar(t_values, Float64)
    for idx in eachindex(t_values)
        value = t_values[idx]
        while value isa ForwardDiff.Dual
            value = ForwardDiff.value(value)
        end
        t_values_primal[idx] = Float64(value)
    end
    if length(t_values_primal) != size(K_values_primal, 3)
        throw(DimensionMismatch("tᵣ length must match the IRF time dimension"))
    end

    if isnothing(orders)
        _, _,
        _,
        _,
        _,
        _,
        ss_order_by_dof = _radiation_state_space_realization(
            K_values_primal, t_values_primal, max_order, R2t; verbose)
    else
        ss_order_by_dof = Int64.(orders)
    end

    # The SVD-based realization is differentiated piecewise: model orders are
    # selected from primal values, then held fixed for the provided AD rule.
    p = (size(K_values_primal), t_values_primal, max_order, R2t, ss_order_by_dof)
    flat = ImplicitAD.provide_rule(
        function (x, p)
            dims, t_values, max_order, R2t, fixed_orders = p
            ss_A, ss_B,
            ss_C,
            ss_D,
            ss_K,
            ss_R2,
            _ = _radiation_state_space_realization(
                reshape(x, dims), t_values, max_order, R2t;
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

    ss_R2_len = n_influenced * n_radiating
    ss_R2 = reshape(flat[next_index:(next_index + ss_R2_len - 1)], n_influenced, n_radiating)

    return ss_A, ss_B, ss_C, ss_D, ss_K, ss_R2, ss_order_by_dof
end

function alternate_Ainf(Kᵣ, A, w_raw, tCIC_raw)
    nw = length(w_raw)
    nt = length(tCIC_raw)
    w = reshape(w_raw, 1, 1, 1, nw)
    tCIC = reshape(tCIC_raw, 1, 1, nt)

    integrand = Kᵣ .* sin.(w .* tCIC) # nDOF, nDOF, nt, nw
    dt = diff(tCIC; dims = 3)
    integral = sum(
        (integrand[:, :, 1:(end - 1), :] .+ integrand[:, :, 2:end, :]) .* 0.5 .* dt;
        dims = [3]) # nDOF, nDOF, 1, nw
    ainf_temp = A + (integral ./ w)[:, :, 1, :] # nDOF, nDOF, nw
    ainf = mean(ainf_temp; dims = [3])[:, :, 1] # nDOF, nDOF
end

end
