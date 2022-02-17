var documenterSearchIndex = {"docs":
[{"location":"getting_started/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"If you are unfamiliar with DifferentialEquations.jl, check out the official tutorial on how to solve ordinary differential equations.","category":"page"},{"location":"getting_started/#Step-1:-Defining-a-problem","page":"Getting Started","title":"Step 1: Defining a problem","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"To solve the Fitzhugh-Nagumo model we first set up an ODEProblem.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"using ProbNumDiffEq\n\nfunction fitz(u, p, t)\n    a, b, c = p\n    return [c*(u[1] - u[1]^3/3 + u[2])\n            -(1/c)*(u[1] -  a - b*u[2])]\nend\n\nu0 = [-1.0; 1.0]\ntspan = (0., 20.)\np = (0.2, 0.2, 3.0)\nprob = ODEProblem(fitz, u0, tspan, p)\nnothing # hide","category":"page"},{"location":"getting_started/#Step-2:-Solving-a-problem","page":"Getting Started","title":"Step 2: Solving a problem","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"To solve the ODEProblem we can use the solve interface that DifferentialEquations.jl defines. All we have to do is to select one of the PN algorithms: EK0 or EK1. In this example we solve the ODE with the default EK0 and high tolerance levels to visualize the resulting uncertainty","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"sol = solve(prob, EK0(), abstol=1e-1, reltol=1e-1)\nnothing # hide","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Note that ProbNumDiffEq.jl supports many of DifferentialEquations.jl's common solver options.","category":"page"},{"location":"getting_started/#Step-3:-Analyzing-the-solution","page":"Getting Started","title":"Step 3: Analyzing the solution","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Just as in DifferentialEquations.jl, the result of solve is a solution object, and we can access the (mean) values and timesteps as usual","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"sol[end]\nsol.u[5]\nsol.t[8]","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"However, the solver returns a probabilistic solution, here a Gaussian distribution over solution values:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"sol.pu[end]","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"It is often convenient to look at means, covariances, and standard deviations via Statistics.jl:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"using Statistics\nmean(sol.pu[5])\ncov(sol.pu[5])\nstd(sol.pu[5])","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"By default, the posterior distribution can be evaluated for arbitrary points in time t by treating sol as a function:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"mean(sol(0.45))","category":"page"},{"location":"getting_started/#Plotting","page":"Getting Started","title":"Plotting","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"The result can be conveniently visualized through Plots.jl:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"using Plots\nplot(sol, color=[\"#107D79\" \"#FF9933\"])\nsavefig(\"./figures/fitzhugh_nagumo.svg\"); nothing # hide","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"(Image: Fitzhugh-Nagumo Solution)","category":"page"},{"location":"solvers/#Solvers-and-Options","page":"Solvers and Options","title":"Solvers and Options","text":"","category":"section"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"ProbNumDiffEq.jl provides mainly the following two solvers, both based on extended Kalman filtering and smoothing. For the best results we suggest using EK1, but note that it relies on the Jacobian of the vector field.","category":"page"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"EK1\nEK0","category":"page"},{"location":"solvers/#ProbNumDiffEq.EK1","page":"Solvers and Options","title":"ProbNumDiffEq.EK1","text":"EK1(; prior=:ibm, order=3, diffusionmodel=DynamicDiffusion(), smooth=true)\n\nGaussian ODE filtering with first order extended Kalman filtering.\n\nCurrently, only the integrated Brownian motion prior :ibm is supported. For the diffusionmodel, chose one of [DynamicDiffusion(), DynamicMVDiffusion(), FixedDiffusion(), FixedMVDiffusion()].\n\nSee also: EK0\n\nReferences:\n\nN. Bosch, P. Hennig, F. Tronarp: Calibrated Adaptive Probabilistic ODE Solvers (2021)\nF. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective (2019)\n\n\n\n\n\n","category":"type"},{"location":"solvers/#ProbNumDiffEq.EK0","page":"Solvers and Options","title":"ProbNumDiffEq.EK0","text":"EK0(; prior=:ibm, order=3, diffusionmodel=DynamicDiffusion(), smooth=true)\n\nGaussian ODE filtering with zeroth order extended Kalman filtering.\n\nCurrently, only the integrated Brownian motion prior :ibm is supported. For the diffusionmodel, chose one of [DynamicDiffusion(), DynamicMVDiffusion(), FixedDiffusion(), FixedMVDiffusion()].\n\nSee also: EK1\n\nReferences:\n\nN. Bosch, P. Hennig, F. Tronarp: Calibrated Adaptive Probabilistic ODE Solvers (2021)\nF. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective (2019)\nM. Schober, S. Särkkä, and P. Hennig: A Probabilistic Model for the Numerical Solution of Initial Value Problems (2018)\n\n\n\n\n\n","category":"type"},{"location":"#ProbNumDiffEq.jl:-Probabilistic-Numerical-Solvers-for-Differential-Equations","page":"Home","title":"ProbNumDiffEq.jl: Probabilistic Numerical Solvers for Differential Equations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Banner)","category":"page"},{"location":"","page":"Home","title":"Home","text":"ProbNumDiffEq.jl provides probabilistic numerical ODE solvers to the DifferentialEquations.jl ecosystem. The implemented ODE filters solve differential equations via Bayesian filtering and smoothing and compute not just a single point estimate of the true solution, but a posterior distribution that contains an estimate of its numerical approximation error.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"For more probabilistic numerics check out the ProbNum Python package. It implements probabilistic ODE solvers, but also probabilistic linear solvers, Bayesian quadrature, and many filtering and smoothing implementations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package can be installed directly with the Julia package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add ProbNumDiffEq","category":"page"},{"location":"#[Getting-Started](@ref)","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To quickly try out ProbNumDiffEq.jl check out the \"Getting Started\" tutorial.","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Two extended Kalman filtering-based probabilistic solvers: the explicit EK0 and semi-implicit EK1.\nAdaptive step-size selection (PI control)\nOn-line uncertainty calibration, for multiple different measurement models\nDense output\nSampling from the solution\nCallback support\nConvenient plotting through a Plots.jl recipe\nAutomatic differentiation via ForwardDiff.jl\nSupports arbitrary precision numbers via BigFloats.jl\nSpecialized solvers for second-order ODEs","category":"page"},{"location":"#Benchmarks","page":"Home","title":"Benchmarks","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Multi-Language Wrapper Benchmark: ProbNumDiffEq.jl vs. OrdinaryDiffEq.jl, Hairer's FORTRAN solvers, Sundials, LSODA, MATLAB, and SciPy.","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"N. Bosch, P. Hennig, F. Tronarp: Calibrated Adaptive Probabilistic ODE Solvers (2021)\nF. Tronarp, S. Särkkä, and P. Hennig: Bayesian ODE Solvers: The Maximum A Posteriori Estimate (2021)\nN. Krämer, P. Hennig: Stable Implementation of Probabilistic ODE Solvers (2020)\nH. Kersting, T. J. Sullivan, and P. Hennig: Convergence Rates of Gaussian Ode Filters (2020)\nF. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective (2019)\nC. J. Oates and T. J. Sullivan: A modern retrospective on probabilistic numerics (2019)\nM. Schober, S. Särkkä, and P. Hennig: A Probabilistic Model for the Numerical Solution of Initial Value Problems (2018)\nP. Hennig, M. A. Osborne, and M. Girolami: Probabilistic numerics and uncertainty in computations (2015)","category":"page"},{"location":"","page":"Home","title":"Home","text":"A more detailed list of references can be found on the probabilistic-numerics.org homepage.","category":"page"}]
}
