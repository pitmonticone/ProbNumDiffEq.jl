########################################################################################
# Caches
########################################################################################
abstract type ODEFiltersCache <: OrdinaryDiffEq.OrdinaryDiffEqCache end
mutable struct GaussianODEFilterCache{
    RType,
    ProjType,
    SolProjType,
    PType,
    PIType,
    uType,
    duType,
    xType,
    AType,
    QType,
    matType,
    HMatType,
    diffusionType,
    diffModelType,
    measType,
    puType,
    llType,
    CType,
    rateType,
    UF,
    JC,
    uNoUnitsType,
} <: ODEFiltersCache
    # Constants
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    A::AType
    Q::QType
    Ah::AType
    Qh::QType
    diffusionmodel::diffModelType
    R::RType
    Proj::ProjType
    SolProj::SolProjType
    # Also mutable
    P::PType
    PI::PIType
    # Mutable stuff
    u::uType
    u_pred::uType
    u_filt::uType
    tmp::uType
    atmp::uNoUnitsType
    x::xType
    x_pred::xType
    x_filt::xType
    x_tmp::xType
    x_tmp2::xType
    measurement::measType
    m_tmp::measType
    pu_tmp::puType
    H::HMatType
    du::duType
    ddu::matType
    K1::matType
    K2::matType
    G1::matType
    G2::matType
    covmatcache::matType
    default_diffusion::diffusionType
    local_diffusion::diffusionType
    global_diffusion::diffusionType
    err_tmp::duType
    log_likelihood::llType
    C1::CType
    C2::CType
    du1::rateType
    uf::UF
    jac_config::JC
end

function OrdinaryDiffEq.alg_cache(
    alg::GaussianODEFilter,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{IIP},
) where {IIP}
    initialize_derivatives = true

    if u isa Number
        error("We currently don't support scalar-valued problems")
    end

    is_secondorder_ode = f isa DynamicalODEFunction
    if is_secondorder_ode
        @warn "Assuming that the given ODE is a SecondOrderODE. If this is not the case, e.g. because it is some other dynamical ODE, the solver will probably run into errors!"
    end

    q = alg.order
    d = is_secondorder_ode ? length(u[1, :]) : length(u)
    D = d * (q + 1)

    u_vec = u[:]
    t0 = t

    uType = typeof(u)
    uElType = eltype(u_vec)
    matType = Matrix{uElType}

    # Projections
    Proj = ProjMatGenerator(d, q)
    @assert f isa AbstractODEFunction
    SolProj = f isa DynamicalODEFunction ? [Proj(1); Proj(0)] : Proj(0)

    # Prior dynamics
    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    P, PI = init_preconditioner(d, q, uElType)

    A, Q = ibm(d, q, uElType)

    initial_variance = 1.0 * ones(uElType, D)
    x0 = Gaussian(zeros(uElType, D), SRMatrix(Matrix(Diagonal(sqrt.(initial_variance)))))

    # Measurement model
    R = zeros(uElType, d, d)

    # Pre-allocate a bunch of matrices
    h = zeros(uElType, d)
    # H = f isa DynamicalODEFunction ? copy(E2) : copy(E1)
    du = f isa DynamicalODEFunction ? similar(u[2, :]) : similar(u)
    ddu = f isa DynamicalODEFunction ? zeros(uElType, d, 2d) : zeros(uElType, d, d)
    @assert f.mass_matrix == I "Mass matrices not supported right now"
    H = ODEHMat(Proj(0), Proj(1), ddu)
    v, S = similar(h), similar(ddu)
    v = similar(h)
    S =
        alg isa EK0 ? SRMatrix(zeros(uElType, d, D), Diagonal(zeros(uElType, d, d))) :
        SRMatrix(zeros(uElType, d, D), zeros(uElType, d, d))
    measurement = Gaussian(v, S)
    pu_tmp =
        f isa DynamicalODEFunction ?
        Gaussian(zeros(uElType, 2d), SRMatrix(zeros(uElType, 2d, D))) : similar(measurement)
    K = zeros(uElType, D, d)
    G = zeros(uElType, D, D)
    C1 = SRMatrix(zeros(uElType, D, 2D), zeros(uElType, D, D))
    C2 = SRMatrix(zeros(uElType, D, 3D), zeros(uElType, D, D))
    covmatcache = similar(G)

    if alg isa EK1FDB
        H = [E1; E2]
        v = [v; v]
        S = SRMatrix(zeros(uElType, 2d, D), zeros(uElType, 2d, 2d))
        measurement = Gaussian(v, S)
        K = zeros(uElType, D, 2d)
    end

    diffmodel = alg.diffusionmodel
    initdiff = initial_diffusion(diffmodel, d, q, uEltypeNoUnits)
    copy!(x0.Σ, apply_diffusion(x0.Σ, initdiff))

    Ah, Qh = copy(A), copy(Q)
    u_pred = similar(u)
    u_filt = similar(u)
    tmp = similar(u)
    x_pred = copy(x0)
    x_filt = similar(x0)
    x_tmp = similar(x0)
    x_tmp2 = similar(x0)
    m_tmp = similar(measurement)
    K2 = similar(K)
    G2 = similar(G)
    err_tmp = similar(du)

    # Things for calc_J
    uf = get_uf(f, t, p, Val(IIP))
    du1 = similar(rate_prototype)
    dw1 = zero(u)
    jac_config = OrdinaryDiffEq.build_jac_config(alg, f, uf, du1, uprev, u, tmp, dw1)
    atmp = similar(u, uEltypeNoUnits)

    return GaussianODEFilterCache{
        typeof(R),
        typeof(Proj),
        typeof(SolProj),
        typeof(P),
        typeof(PI),
        uType,
        typeof(du),
        typeof(x0),
        typeof(A),
        typeof(Q),
        matType,
        typeof(H),
        typeof(initdiff),
        typeof(diffmodel),
        typeof(measurement),
        typeof(pu_tmp),
        uEltypeNoUnits,
        typeof(C1),
        typeof(du1),
        typeof(uf),
        typeof(jac_config),
        typeof(atmp),
    }(
        # Constants
        d,
        q,
        A,
        Q,
        Ah,
        Qh,
        diffmodel,
        R,
        Proj,
        SolProj,
        P,
        PI,
        # Mutable stuff
        u,
        u_pred,
        u_filt,
        tmp,
        atmp,
        x0,
        x_pred,
        x_filt,
        x_tmp,
        x_tmp2,
        measurement,
        m_tmp,
        pu_tmp,
        H,
        du,
        ddu,
        K,
        K2,
        G,
        G2,
        covmatcache,
        initdiff,
        initdiff * NaN,
        initdiff * NaN,
        err_tmp,
        zero(uEltypeNoUnits),
        C1,
        C2,
        du1,
        uf,
        jac_config,
    )
end

get_uf(f, t, p, ::Val{true}) = OrdinaryDiffEq.UJacobianWrapper(f, t, p)
get_uf(f, t, p, ::Val{false}) = OrdinaryDiffEq.UDerivativeWrapper(f, t, p)
