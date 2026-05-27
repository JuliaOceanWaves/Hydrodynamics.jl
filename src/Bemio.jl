module Bemio

import NetCDF
using Statistics
import LinearAlgebra
import ToeplitzMatrices
using ExponentialUtilities
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

    # info = NetCDF.ncinfo(filename);
    # ds = NCDataset(filename,"r")

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
    # t = collect(0:dt:t_f)'
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

function radiation_state_space(Kᵣ, tᵣ, max_order = 10, R2t = 0.95)
    dt = Unitful.ustrip.(tᵣ[2] - tᵣ[1])

    ss_A_by_dof = zeros(size(Kᵣ, 1), size(Kᵣ, 2), max_order, max_order)
    ss_B_by_dof = zeros(size(Kᵣ, 1), size(Kᵣ, 2), max_order, 1)
    ss_C_by_dof = zeros(size(Kᵣ, 1), size(Kᵣ, 2), 1, max_order)
    ss_D_by_dof = zeros(size(Kᵣ)[1:2])
    ss_K_by_dof = zeros(size(Kᵣ)[1:3])
    ss_R2_by_dof = zeros(size(Kᵣ)[1:2])
    ss_order_by_dof = Int64.(zeros(size(Kᵣ)[1:2]))

    for i in axes(Kᵣ)[1], j in axes(Kᵣ)[2]
        print("dof: ", i, " ", j)
        irf_K = Unitful.ustrip.(Kᵣ[i, j, :])
        R2i = LinearAlgebra.norm(irf_K .- mean(irf_K))
        R2 = R2i

        order = 0 # Initial state space order
        y = dt .* irf_K
        n = length(y)
        h = ToeplitzMatrices.Hankel([y[2:end]; zeros(n - 1)], (n - 1, n - 1))
        u, svh, v = LinearAlgebra.svd(h)

        # Define variables so they pass outside of the loop 
        ac = zeros(1, 1)
        bc = zeros(1)
        cc = zeros(1)
        dc = 0.0
        ss_K_each_dof = zeros(length(tᵣ))
        # while R2 > R2t && order <= max_order
        # order += 1
        for m in collect(1:max_order)
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

            iidd = inv(dt / 2 * (LinearAlgebra.I(order) + a))  # (T/2*I+T/2*A)^{-1} = 2/T(I+A)^{-1}
            ac = (a - LinearAlgebra.I(order)) * iidd           # (A-I)2/T(I+A)^{-1} = 2/T(A-I)(I+A)^{-1}
            bc = dt * (iidd * b)                               # (T/2+T/2)*2/T(I+A)^{-1}B = 2(I+A)^{-1}B
            cc = c * iidd                                      # C*2/T(I+A)^{-1} = 2/T(I+A)^{-1}
            dc = d .- dt / 2 * ((c * iidd) * b)                # D-T/2C (2/T(I+A)^{-1})B = D-C(I+A)^{-1})B

            ss_K_each_dof = zeros(length(tᵣ))
            method = ExpMethodNative()
            term = ac * dt * 0
            cache = ExponentialUtilities.alloc_mem(term, method) # Main allocation done here
            for k in 1:length(tᵣ)
                term = ac * dt * (k - 1)
                exponential!(term) # Very little allocation here
                ss_K_each_dof[k] = ((cc * term) * bc)[1] # IRF approximation using SS result
            end

            # Calculate R2 for the state space fit of Kᵣ. Check if above 0.95 threshold
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
        print("; order: ", order, "\n")
    end

    total_order = sum(ss_order_by_dof)
    ss_A = zeros(total_order, total_order)
    ss_B = zeros(total_order, size(Kᵣ)[2])
    ss_C = zeros(size(Kᵣ)[1], total_order)
    order_count = 1
    for i in axes(ss_A_by_dof)[1]
        for j in axes(ss_A_by_dof)[2]
            o1 = order_count
            o2 = ss_order_by_dof[i, j] + order_count - 1
            ss_A[o1:o2, o1:o2] = ss_A_by_dof[
                i, j, 1:ss_order_by_dof[i, j], 1:ss_order_by_dof[i, j]]
            ss_B[o1:o2, j] = ss_B_by_dof[i, j, 1:ss_order_by_dof[i, j], 1]
            ss_C[i, o1:o2] = ss_C_by_dof[i, j, 1, 1:ss_order_by_dof[i, j]]
            order_count = o2 + 1
        end
    end
    return ss_A, ss_B, ss_C, ss_D_by_dof, ss_K_by_dof, ss_R2_by_dof, ss_order_by_dof
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
