# Goal: Make sure some combinations that raise errors do so
using Test
using OrdinaryDiffEq
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary:
    prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo

@testset "Fixed-timestep requires dt" begin
    prob = prob_ode_lotkavoltera
    @test_throws ErrorException solve(prob, EK0(), adaptive=false)
    @test solve(prob, EK0(), adaptive=false, dt=0.05) isa ProbNumDiffEq.ProbODESolution
end

@testset "`dense=true` requires `smooth=true`" begin
    prob = prob_ode_lotkavoltera
    @test_throws ErrorException solve(prob, EK0(smooth=false))
end

@testset "`save_everystep=false` requires `smooth=false`" begin
    prob = prob_ode_lotkavoltera
    @test_throws ErrorException solve(prob, EK0(smooth=true), save_everystep=false)
end
