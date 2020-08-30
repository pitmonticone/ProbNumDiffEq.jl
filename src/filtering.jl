########################################################################################
# (Extended) Kalman Filtering and Smoothing
########################################################################################
function kf_predict(m::Vector, P::AbstractMatrix, A::AbstractMatrix, Q::AbstractMatrix)
    return (m=(A*m), P=(A*P*A' + Q))
end

function kf_update(m::Vector, P::AbstractMatrix, A::AbstractMatrix, Q::AbstractMatrix, H::AbstractMatrix, R::AbstractMatrix, y::Vector)
    v = y - H*m
    S = H * P * H' + R
    K = P * H' * inv(S)
    return (m=(m + K*v), P=(P - K*S*K'))
end

function kf_smooth(m_f_t::Vector, P_f_t::AbstractMatrix,
                   m_p_t1::Vector, P_p_t1::AbstractMatrix,
                   m_s_t1::Vector, P_s_t1::AbstractMatrix,
                   A::AbstractMatrix, Q::AbstractMatrix,
                   Precond::AbstractMatrix,
                   jitter=1e-12)
    G = P_f_t * A' * inv(P_p_t1 + jitter*(Precond*Precond'))
    m = m_f_t + G * (m_s_t1 - m_p_t1)
    P = P_f_t + G * (P_s_t1 - P_p_t1) * G'

    # Sanity: Make sure that the diagonal of P is non-negative
    _min = minimum(diag(P))
    if _min < 0
        try
            @assert abs(_min) < 1e-16
            P += - _min*I
        catch e
            @info "Error while smoothing: negative variances!" P_f_t P_s_t1 P_p_t1 P_s_t1-P_p_t1 G P
            display(P_f_t)
            display(P_s_t1 - P_p_t1)
            display(G)
            display(P)
            throw(e)
        end
    end
    @assert all(diag(P) .>= 0) "The covariance `P` might be NaN! Make sure that the covariances during the solve make sense."

    return m, P
end


function ekf_predict(m::Vector, P::AbstractMatrix, f::Function, F::Function, Q::AbstractMatrix)
    return (m=f(m), P=(F(m)*P*F(m)' + Q))
end

function ekf_update(m::Vector, P::AbstractMatrix, h::Function, H::Function, R::AbstractMatrix, y::Vector)
    v = y - h(m)
    S = H(m) * P * H(m)' + R
    K = P * H(m)' * inv(S)
    @show K*S*K'
    return (m=(m + K*v), P=(P - K*S*K'))
end

function ekf_smooth(m_f_t::Vector, P_f_t::AbstractMatrix,
                    m_p_t1::Vector, P_p_t1::AbstractMatrix,
                    m_s_t1::Vector, P_s_t1::AbstractMatrix,
                    F::Function, Q::AbstractMatrix)
    G = P_f_t * F(m_f_t)' * inv(P_p_t1)
    m = m_f_t + G * (m_s_t1 - m_p_t1)
    P = P_f_t + G * (P_s_t1 - P_p_t1) * G'
    return m, P
end
