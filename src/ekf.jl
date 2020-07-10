########################################################################################
# Structure following https://github.com/SciML/SimpleDiffEq.jl/tree/master/src/rk4
########################################################################################


########################################################################################
# Algorithm
########################################################################################
abstract type AbstractODEFilter <: DiffEqBase.AbstractODEAlgorithm end
mutable struct ODEFilter <: AbstractODEFilter end
mutable struct EKF0 <: AbstractODEFilter end
mutable struct EKF1 <: AbstractODEFilter end


########################################################################################
# Integrator
########################################################################################
mutable struct ODEFilterIntegrator{IIP, S, X, T, P, F} <: DiffEqBase.AbstractODEIntegrator{ODEFilter, IIP, S, T}
    f::F                  # eom
    u::S                  # current functionvalue
    x::X                  # current state
    xprev::X              # previous state
    tmp::X                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    x_modified::Bool
    # ks::Vector{S}         # interpolants of the algorithm
    # cs::SVector{6, T}     # ci factors cache: time coefficients
    # as::SVector{21, T}    # aij factors cache: solution coefficients
    # rs::SVector{22, T}    # rij factors cache: interpolation coefficients

    # My additions
    d::Int
    q::Int
    dm
    mm
    sigma_estimator
    precondition::Bool
    preconditioner
    steprule
end
DiffEqBase.isinplace(::ODEFilterIntegrator{IIP}) where {IIP} = IIP


########################################################################################
# Initialization
########################################################################################
function odefilter_init(f::F, IIP::Bool, u0::S, t0::T, dt::T, p::P, q::Int, method, sigmarule, steprule, abstol, reltol, ρ, prob_kwargs, precondition=true) where {F, P, T, S}
    # if isinstance(u0, )

    d = length(u0)
    dm = ibm(q, d)
    mm = measurement_model(method, d, q, f, p)

    initialize_derivatives = false
    initialize_derivatives = initialize_derivatives == :auto ? q <= 3 : false
    if initialize_derivatives
        derivatives = get_derivatives((x, t) -> f(x, p, t), d, q)
        m0 = vcat(u0, [_f(u0, t0) for _f in derivatives]...)
    else
        m0 = [u0; f(u0, p, t0); zeros(d*(q-1))]
    end

    if eltype(m0) <: Measurement
        P0 = diagm(0 => Measurements.uncertainty.(m0) .^ 2)
        m0 = Measurements.value.(m0)
    else
        P0 = diagm(0 => [zeros(d); ones(d*q)] .+ 1e-16)
    end
    x0 = Gaussian(m0, P0)
    X = typeof(x0)

    precond = preconditioner(dt, d, q)
    precondition && apply_preconditioner!(precond, x0)

    steprules = Dict(
        :constant => constant_steprule(),
        :pvalue => pvalue_steprule(0.05),
        :baseline => classic_steprule(abstol, reltol; ρ=ρ),
        :measurement_error => measurement_error_steprule(;abstol=abstol, reltol=reltol, ρ=ρ),
        :measurement_scaling => measurement_scaling_steprule(),
        :schober16 => schober16_steprule(;ρ=ρ, abstol=abstol, reltol=reltol),
    )
    steprule = steprules[steprule]

    return ODEFilterIntegrator{IIP, S, X, T, P, F}(
        f, u0, _copy(x0), _copy(x0), _copy(x0), t0, t0, t0, dt, sign(dt), p, true,
        d, q, dm, mm, sigmarule, precondition, precond, steprule
    )
end


########################################################################################
# Solve
########################################################################################
function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::ODEFilter;
                            dt=0.1,
                            q=1,
                            saveat=nothing,
                            save_everystep=true,
                            abstol=1e-6, reltol=1e-3, ρ=0.95,
                            method=:ekf1,
                            sigmarule=Schober16Sigma(),
                            steprule=:baseline,
                            progressbar=false,
                            maxiters=1e5,
                            smoothed=true,
                            kwargs...)
    # Init
    IIP = DiffEqBase.isinplace(prob)
    f = IIP ? IIP_to_OOP(prob.f) : prob.f
    integ = odefilter_init(f, false, prob.u0, prob.tspan[1], dt, prob.p, q, method, sigmarule, steprule, abstol, reltol, ρ, prob.kwargs)

    # More Initialization
    t_0, T = prob.tspan
    sol = StructArray([(t=integ.t, x=integ.x)])
    proposals = []
    retcode = :Success

    # Solve
    if progressbar pbar_update, pbar_close = make_progressbar(0.1) end
    iter = 0
    while integ.t < T
        if progressbar pbar_update(fraction=(integ.t-t_0)/(T-t_0)) end

        # Here happens the main "work"
        proposal = predict_update(integ)

        accept, dt_proposal = integ.steprule(integ, proposal, proposals)
        push!(proposals, (proposal..., accept=accept, dt=integ.dt))
        integ.dt = min(dt_proposal, accept ? T-proposal.t : T-integ.t)

        if accept
            push!(sol, (t=proposal.t, x=proposal.filter_estimate))
            integ.x = proposal.filter_estimate
            integ.t = proposal.t
        end

        iter += 1
        if iter >= maxiters
            break
            retcode = :MaxIters
        end
    end
    if progressbar pbar_close() end

    smoothed && smooth!(sol, proposals, integ)
    calibrate!(sol, proposals, integ)
    integ.precondition && undo_preconditioner!(sol, proposals, integ)

    # Format Solution
    sol = DiffEqBase.build_solution(prob, alg, sol.t, StructArray(sol.x),
                                    proposals, integ,
                                    retcode=retcode)

    return sol
end
DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::EKF0; kwargs...) =
    DiffEqBase.__solve(prob, ODEFilter(); method=:ekf0, kwargs...)
DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::EKF1; kwargs...) =
    DiffEqBase.__solve(prob, ODEFilter(); method=:ekf1, kwargs...)



########################################################################################
# Step
########################################################################################
function DiffEqBase.step!(integ::ODEFilterIntegrator{false, S, X, T}) where {S, X, T}
end