module Bemio

import NCDatasets
import NetCDF
import Integrals

struct Hydro
    radiating_dof
    influenced_dof
    g
    rho
    w
    period
    wave_direction
    wavelength
    depth
    volume
    cb
    ex
    fk
    di
    am
    rd
    khs
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
    rd = NetCDF.ncread(filename, "radiation_damping") # Dimensions: influenced_dof radiating_dof omega
    khs = NetCDF.ncread(filename, "hydrostatic_stiffness")' # Dimensions: radiating_dof influenced_dof --> influenced_dof radiating_dof

    # ainf, ra_t, ra_w = radiationIRF!()
    return Hydro(radiating_dof, influenced_dof, g, rho, w, period, wave_direction, wavelength, depth, volume, cb, ex, fk, di, am, rd, khs)
end

function radiationIRF!(myHydro::Hydro, timeEnd=60, nDt::Int=1001, nDw::Int=1001, wMin=minimum(myHydro.w), wMax=maximum(myHydro.w))
    time = collect(0:timeEnd/(nDt-1):timeEnd)
end

end