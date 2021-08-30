# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    if integ.opts.dense && !integ.alg.smooth
        error("To use `dense=true` you need to set `smooth=true`!")
    elseif !integ.opts.dense && integ.alg.smooth
        @warn "If you set dense=false for efficiency, you might also want to set smooth=false."
    end
    if !integ.opts.save_everystep && integ.alg.smooth
        error("If you do not save all values, you do not need to smooth!")
    end
    @assert integ.saveiter == 1

    # Update the initial state to the known (given or computed with AD) initial values
    initial_update!(integ)

    # These are necessary since the solution object is not 100% initialized by default
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter,
                                   mul!(cache.pu_tmp, cache.SolProj, cache.x))
end



function iekf_refine!(integ, cache)
    tnew = integ.t + integ.dt
    @unpack x_pred, x_filt, SolProj = cache
    u_pred = copy(cache.u_pred)


    # Do the IEKF stuff here!
    # ϵ₁, ϵ₂ = 1e-25, 1e-15
    ϵ₁ = ϵ₂ = integ.opts.abstol

    m_i = x_pred.μ
    z_i = cache.measurement.μ
    m_i_new = x_filt.μ
    K = cache.K2
    @assert !iszero(m_i_new .- m_i)

    # @info "norms after the first normal update" norm(m_i .- m_i_new) norm(z_i)
    i = 0
    maxiters = 10

    if norm(m_i_new .- m_i) < ϵ₁ && norm(z_i) < ϵ₂
        # @info "directly accepted!!!"
        return
    end

    while i < maxiters
        i += 1
        # @info "IEKF iteration $i" m_i m_i_new
        # @info "IEKF iteration $i" norm(m_i_new .- m_i) norm(z_i)

        # Re-evaluate the measurement function to compute z_i, H_i
        evaluate_ode!(integ, u_pred, m_i, tnew) # overwrites cache.du, cache.H, cache.measurement.μ
        compute_measurement_covariance!(cache) # overwrites cache.measurement.Σ

        x_filt = update!(x_filt, x_pred, cache.measurement, m_i,
                         cache.H, cache.R, cache.K1, cache.K2, cache.x_tmp2.Σ.mat)

        m_i_new = x_filt.μ

        if norm(m_i_new .- m_i) < ϵ₁ && norm(z_i) < ϵ₂
            m_i = m_i_new
            _matmul!(view(u_pred, :), SolProj, m_i)
            break
        end

        # if i > 1 && m_i_new != m_i
        #     @info "they are not the same now!"
        # end
        m_i = m_i_new
        _matmul!(view(u_pred, :), SolProj, m_i)
        # @info "?" m_i_new norm(m_i_new .- m_i) norm(z_i)
    end

    if !(norm(m_i_new .- m_i) < ϵ₁ && norm(z_i) < ϵ₂)
        # @info "IEKF did not converge!" norm(m_i_new .- m_i) norm(z_i) < ϵ₂
    else
        # @info "IEKF converged after $i iterations!"
    end

end


"""Perform a step

Not necessarily successful! For that, see `step!(integ)`.

Basically consists of the following steps
- Coordinate change / Predonditioning
- Prediction step
- Measurement: Evaluate f and Jf; Build z, S, H
- Calibration; Adjust prediction / measurement covs if the diffusion model "dynamic"
- Update step
- Error estimation
- Undo the coordinate change / Predonditioning
"""
function OrdinaryDiffEq.perform_step!(integ, cache::GaussianODEFilterCache, repeat_step=false)
    @unpack t, dt = integ
    @unpack d, SolProj = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack x_tmp, x_tmp2 = integ.cache
    @unpack A, Q, Ah, Qh = integ.cache

    make_preconditioners!(integ, dt)
    @unpack P, PI = integ.cache

    tnew = t + dt

    # Build the correct matrices
    @. Ah = PI.diag .* A .* P.diag'
    X_A_Xt!(Qh, Q, PI)

    # Predict mean
    predict_mean!(x_pred, x, Ah)
    _matmul!(view(u_pred, :), SolProj, x_pred.μ)

    # Measure
    evaluate_ode!(integ, u_pred, x_pred.μ, tnew)

    if isdynamic(cache.diffusionmodel)
        # Estimate diffusion
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)

        # Compute predicted covariance, considering the diffusion
        predict_cov!(x_pred, x, Ah, Qh, cache.C1, cache.global_diffusion)

        # Compute measurement covariance only now
        compute_measurement_covariance!(cache)
    else
        # Compute predicted covariance with diffusion 1
        predict_cov!(x_pred, x, Ah, Qh, cache.C1)

        # Compute measurement covariance
        compute_measurement_covariance!(cache)

        # Estimate diffusion - this requires an up-to-date measurement covariance!
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
    end

    x_filt = update!(integ, x_pred)

    # Perform IEKF iterations to refine the solution
    # iekf_refine!(integ, cache)

    # Likelihood
    # cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Estimate error for adaptive steps - can already be done before filtering
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(cache)
        if integ.f isa DynamicalODEFunction # second-order ODE
            DiffEqBase.calculate_residuals!(
                err_tmp, dt * err_est_unscaled,
                integ.u[1, :], u_pred[1, :],
                integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        else # regular first-order ODE
            DiffEqBase.calculate_residuals!(
                err_tmp, dt * err_est_unscaled,
                integ.u, u_pred,
                integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        end
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar
    end


    # If the step gets rejected, we don't even need to perform an update!
    reject = integ.opts.adaptive && integ.EEst >= one(integ.EEst)
    if !reject
        # Update
        # x_filt = update!(integ, x_pred)

        # Save into u_filt and integ.u
        mul!(view(u_filt, :), SolProj, x_filt.μ)
        if integ.u isa Number
            integ.u = u_filt[1]
        else
            integ.u .= u_filt
        end

        # Advance the state here
        copy!(integ.cache.x, integ.cache.x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
end

function evaluate_ode!(integ, u_pred, x_pred_mean, t, second_order::Val{false})
    @unpack f, p, dt, alg = integ
    @unpack du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)

    @unpack E0, E1 = integ.cache

    z, S = measurement.μ, measurement.Σ

    # Mean
    _eval_f!(du, u_pred, p, t, f)
    integ.destats.nf += 1
    # z .= E1*x_pred_mean .- du
    _matmul!(z, E1, x_pred_mean)
    z .-= du[:]

    # Cov
    if alg isa EK1 || alg isa IEKS
        linearize_at = (alg isa IEKS && !isnothing(alg.linearize_at)) ?
            alg.linearize_at(t).μ : u_pred

        # Jacobian is now computed either with the given jac, or ForwardDiff
        if !isnothing(f.jac)
            _eval_f_jac!(ddu, linearize_at, p, t, f)
        elseif isinplace(f)
            ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u_pred)
        else
            ddu .= ForwardDiff.jacobian(u -> f(u, p, t), u_pred)
        end

        integ.destats.njacs += 1
        _matmul!(H, ddu, -E0)
        H .= E1
        _matmul!(H, ddu, E0, -1, 1)
    else
        # H .= E1 # This is already the case!
    end

    return measurement
end

function evaluate_ode!(integ, u_pred, x_pred_mean, t, second_order::Val{true})
    @unpack f, p, dt, alg = integ
    @unpack d, u_pred, du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)
    du2 = du

    @unpack E0, E1, E2 = integ.cache

    z, S = measurement.μ, measurement.Σ

    # Mean
    # _u_pred = E0 * x_pred_mean
    # _du_pred = E1 * x_pred_mean
    if isinplace(f)
        f.f1(du2, view(u_pred, 1:d), view(u_pred, d+1:2d), p, t)
    else
        du2 .= f.f1(view(u_pred, 1:d), view(u_pred, d+1:2d), p, t)
    end
    integ.destats.nf += 1
    z .= E2*x_pred_mean .- du2[:]

    # Cov
    if alg isa EK1
        @assert !(alg isa IEKS)

        if isinplace(f)
            J0 = copy(ddu)
            ForwardDiff.jacobian!(J0, (du2, u) -> f.f1(du2, view(u_pred, 1:d), u, p, t), du2,
                                  u_pred[d+1:2d])

            J1 = copy(ddu)
            ForwardDiff.jacobian!(J1, (du2, du) -> f.f1(du2, du, view(u_pred, d+1:2d),
                                                        p, t), du2,
                                  u_pred[1:d])

            integ.destats.njacs += 2

            H .= E2 .- J0 * E0 .- J1 * E1
        else
            J0 = ForwardDiff.jacobian((u) -> f.f1(view(u_pred, 1:d), u, p, t), u_pred[d+1:2d])
            J1 = ForwardDiff.jacobian((du) -> f.f1(du, view(u_pred, d+1:2d), p, t), u_pred[1:d])
            integ.destats.njacs += 2
            H .= E2 .- J0 * E0 .- J1 * E1
        end
    else
        # H .= E2 # This is already the case!
    end

    return measurement
end
evaluate_ode!(integ, u_pred, x_pred_mean, t) = evaluate_ode!(
    integ, u_pred, x_pred_mean, t, Val(integ.f isa DynamicalODEFunction))

# The following functions are just there to handle both IIP and OOP easily
_eval_f!(du, u, p, t, f::AbstractODEFunction{true}) = f(du, u, p, t)
_eval_f!(du, u, p, t, f::AbstractODEFunction{false}) = (du .= f(u, p, t))
_eval_f_jac!(ddu, u, p, t, f::AbstractODEFunction{true}) = f.jac(ddu, u, p, t)
_eval_f_jac!(ddu, u, p, t, f::AbstractODEFunction{false}) = (ddu .= f.jac(u, p, t))

compute_measurement_covariance!(cache) =
    X_A_Xt!(cache.measurement.Σ, cache.x_pred.Σ, cache.H)

function update!(integ, prediction)
    @unpack measurement, H, R, x_filt = integ.cache
    @unpack K1, K2, x_tmp2 = integ.cache
    update!(x_filt, prediction, measurement, integ.cache.x_pred.μ, H, R, K1, K2, x_tmp2.Σ.mat)
    # assert_nonnegative_diagonal(x_filt.Σ)
    return x_filt
end


function estimate_errors(cache::GaussianODEFilterCache)
    @unpack local_diffusion, Qh, H = cache

    if local_diffusion isa Real && isinf(local_diffusion)
        return Inf
    end

    L = cache.m_tmp.Σ.squareroot

    if local_diffusion isa Diagonal

        mul!(L, H, sqrt.(local_diffusion) * Qh.squareroot)
        error_estimate = sqrt.(diag(L*L'))
        return error_estimate

    elseif local_diffusion isa Number

        mul!(L, H, Qh.squareroot)
        # error_estimate = local_diffusion .* diag(L*L')
        @tullio error_estimate[i] := L[i,j]*L[i,j]
        error_estimate .*= local_diffusion
        error_estimate .= sqrt.(error_estimate)
        return error_estimate

    end
end
