########################################################################################
# Solution
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end
struct ProbODESolution{T,N,uType,xType,tType,P,A,IType} <: AbstractProbODESolution{T,N,uType}
    u::uType
    x::xType
    t::tType
    prob::P
    alg::A
    dense::Bool
    interp::IType
    retcode::Symbol
end

function DiffEqBase.build_solution(
    prob::DiffEqBase.AbstractODEProblem,
    alg::ODEFilter,
    t,x;
    dense=false,
    retcode = :Default,
    kwargs...)

    d = length(prob.u0)
    function make_Measurement(state)
        # @assert isdiag(state.Σ[1:d,1:d]) state.Σ[1:d,1:d]
        return state.μ[1:d] .± sqrt.(diag(state.Σ)[1:d])
    end
    u = map(make_Measurement, x)
    # u = map(s -> s.μ[1:d], x)

    interp = DiffEqBase.LinearInterpolation(t,u)

    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    return ProbODESolution{T,N,typeof(u),typeof(x),typeof(t),typeof(prob),typeof(alg),typeof(interp)}(
        u,x,t,prob,alg,dense,interp,retcode)
end


########################################################################################
# Plotting
########################################################################################
@recipe function f(sol::AbstractProbODESolution; c=1.96)
    stack(x) = collect(reduce(hcat, x)')
    values = map(u -> Measurements.value.(u), sol.u)
    uncertainties = map(u -> Measurements.uncertainty.(u), sol.u)
    ribbon := stack(uncertainties) * c
    return sol.t, stack(values)
end
