using Test
using ProbNumDiffEq
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary:
    prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo

@testset "Test the different diffusion models" begin
    prob = prob_ode_fitzhughnagumo
    true_sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)

    @testset "Time-Varying Diffusion" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=DynamicDiffusion(), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-4,
        )
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Fixed Diffusion" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedDiffusion(), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-4,
        )
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Fixed Diffusion - uncalibrated and with custom initial value" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedDiffusion(1e3, false), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-4,
        )
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Varying Diagonal Diffusion" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=DynamicMVDiffusion(), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-4,
        )
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Fixed Diagonal Diffusion" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedMVDiffusion(), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-4,
        )
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Fixed Diagonal Diffusion - uncalibrated and with custom values" begin
        d = length(prob.u0)
        initial_diffusion = 1 .+ rand(d)
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedMVDiffusion(initial_diffusion, false), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-4,
        )
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end
end
