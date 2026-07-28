using Hydrodynamics
using NBInclude
using Test

@testset "WEC-Sim verification" begin
    file = joinpath(@__DIR__, "..", "examples", "wec-sim_comparison_3dof.ipynb")
    position_error = @nbinclude(file)
    # accuracy of the verification to wec-sim when the notebook was first created. 
    # last_result is the RMSE of position in [surge, heave, pitch] dofs x [cic, ss] methods
    last_result = [0.0778 0.1116; 0.7228 0.7229; 0.0070 0.0074]
    @test size(last_result) == size(position_error)
    for idx in CartesianIndices(last_result)
        @test position_error[idx] < last_result[idx]
    end
end

@testset "AF verification" begin
    # TODO - use power performance notebook to confirm gradients are functional
    last_result = 0.051
    file = joinpath(@__DIR__, "..", "examples", "power_performance.ipynb")
    mae = @nbinclude(file)
    @test mae < last_result
end