abstract type InitializationScheme end
struct TaylorModeInit <: InitializationScheme end
Base.@kwdef struct ClassicSolverInit{ALG} <: InitializationScheme
    alg::ALG = Tsit5()
    init_on_du::Bool = false
end

function initial_update!(integ, cache)
    return initial_update!(integ, cache, integ.alg.initialization)
end

########################################################################
# Some utilities below
"""Quick and dirty wrapper to make OOP functions IIP"""
function oop_to_iip(f)
    function f!(du, u, p, t)
        du .= f(u, p, t)
        return nothing
    end
    return f!
end

"""Basically an Kalman update"""
function condition_on!(
    x::SRGaussian,
    H::AbstractMatrix,
    data::AbstractVector,
    meascache,
    Kcache,
    covcache,
    Mcache,
)
    z, S = meascache

    _matmul!(z, H, x.μ)
    X_A_Xt!(S, x.Σ, H)
    @assert isdiag(S)
    S_diag = diag(S)
    if any(iszero.(S_diag)) # could happen with a singular mass-matrix
        S_diag .+= 1e-20
    end

    _matmul!(Kcache, x.Σ.mat, H')
    K = Kcache ./= S_diag'

    _matmul!(x.μ, K, data - z, 1.0, 1.0)
    # x.μ .+= K*(data - z)

    D = length(x.μ)
    mul!(Mcache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        Mcache[i, i] += 1
    end
    X_A_Xt!(covcache, x.Σ, Mcache)
    copy!(x.Σ, covcache)
    return nothing
end