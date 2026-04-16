import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS # required for Zygote's reverse AD

function rampFunction(start_time, ramp_time, current_time)
    if current_time < start_time
        ramp = 0.0
    elseif current_time >= ramp_time
        ramp = 1.0
    else
        ramp = 0.5 * (1 + cos(pi + pi .* current_time ./ ramp_time))
    end
end

function calcExcitationForce(current_time, exCoeff, wave)
    omegaVector, phase, spectrum, dFrequency, start_time, ramp_time = wave
    ov = reshape(omegaVector, (1, 1, length(omegaVector)))
    p = reshape(phase, (1, 1, length(phase)))
    s = reshape(spectrum, (1, 1, length(spectrum)))

    ramp = rampFunction(start_time, ramp_time, current_time)
    expTerm = ov .* current_time .+ p
    forceMatrix = ramp .* (exCoeff[:, :, :, 1] .* cos.(expTerm) -
                   exCoeff[:, :, :, 2] .* sin.(expTerm)) .* sqrt.(2 * s .* dFrequency)

    # Format required for unitful input. `sum`` doesn't play nice with matrices of mixed units and dimensions.
    # So instead multiply by an identity matrix to do the same summation in another way.
    # return sum(forceMatrix[:,:,:]; dims=[2,3])
    o = ones(size(forceMatrix, 3))
    return forceMatrix[:, 1, :] * o
end

function excitedHydroOscillator(du, u, p, t)
    (k, c, invMass, exCoeff, wave) = p
    excitationForce = calcExcitationForce(t, exCoeff, wave)
    return invMass * (-c * du - k * u - excitationForce)
end

function hydrodynamicSolver(dx₀, x₀, ts, p)
    dt = diff(ts[1:2])[1]
    diffEqProb = ODE.SecondOrderODEProblem(excitedHydroOscillator, dx₀, x₀, ts[[1, end]], p)
    diffEqSol = ODE.solve(diffEqProb, ODE.Vern6(), saveat = dt)
    return diffEqSol
end
