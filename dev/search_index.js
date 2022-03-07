var documenterSearchIndex = {"docs":
[{"location":"getting_started/#Solving-ODEs-with-Probabilistic-Numerics","page":"Introduction to ODE Filters","title":"Solving ODEs with Probabilistic Numerics","text":"","category":"section"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"In this tutorial we solve a simple non-linear ordinary differential equation (ODE) with the probabilistic numerical ODE solvers implemented in this package.","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"note: Note\nIf you never used DifferentialEquations.jl, check out their \"Ordinary Differential Equations\" tutorial on how to solve ordinary differential equations with classic numerical solvers.","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"In this tutorial, we consider a Fitzhugh-Nagumo model described by an ODE of the form","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"beginaligned\ndoty_1(t) = c (y_1 - fracy_1^33 + y_2) \ndoty_2(t) = -frac1c (y_1 - a - b y_2)\nendaligned","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"on a time span t in 0 T, with initial value y(0) = y_0. In the following, we","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"define the problem with explicit choices of initial values, integration domains, and parameters,\nsolve the problem with our ODE filters, and\nvisualize the results and the corresponding uncertainties.","category":"page"},{"location":"getting_started/#TL;DR:","page":"Introduction to ODE Filters","title":"TL;DR:","text":"","category":"section"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"using ProbNumDiffEq, Plots\n\nfunction fitz(du, u, p, t)\n    a, b, c = p\n    du[1] = c*(u[1] - u[1]^3/3 + u[2])\n    du[2] = -(1/c)*(u[1] -  a - b*u[2])\nend\nu0 = [-1.0; 1.0]\ntspan = (0., 20.)\np = (0.2, 0.2, 3.0)\nprob = ODEProblem(fitz, u0, tspan, p)\n\nusing Logging; Logging.disable_logging(Logging.Warn) # hide\nsol = solve(prob, EK1())\nLogging.disable_logging(Logging.Debug) # hide\nplot(sol)","category":"page"},{"location":"getting_started/#Step-1:-Defining-the-problem","page":"Introduction to ODE Filters","title":"Step 1: Defining the problem","text":"","category":"section"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"We first import ProbNumDiffEq.jl","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"using ProbNumDiffEq","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"and then set up an ODEProblem exactly as we're used to with DifferentialEquations.jl, by defining the vector field","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"function fitz(du, u, p, t)\n    a, b, c = p\n    du[1] = c*(u[1] - u[1]^3/3 + u[2])\n    du[2] = -(1/c)*(u[1] -  a - b*u[2])\nend\nnothing # hide","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"and then an ODEProblem, with initial value u0, time span tspan, and parameters p","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"u0 = [-1.0; 1.0]\ntspan = (0., 20.)\np = (0.2, 0.2, 3.0)\nprob = ODEProblem(fitz, u0, tspan, p)\nnothing # hide","category":"page"},{"location":"getting_started/#Step-2:-Solving-the-problem","page":"Introduction to ODE Filters","title":"Step 2: Solving the problem","text":"","category":"section"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"To solve the ODE we just use DifferentialEquations.jl's solve interface, together with one of the algorithms implemented in this package. For now, let's use EK1:","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"using Logging; Logging.disable_logging(Logging.Warn) # hide\nsol = solve(prob, EK1())\nLogging.disable_logging(Logging.Debug) # hide\nnothing # hide","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"That's it! we just computed a probabilistic numerical ODE solution!","category":"page"},{"location":"getting_started/#Step-3:-Analyzing-the-solution","page":"Introduction to ODE Filters","title":"Step 3: Analyzing the solution","text":"","category":"section"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"The result of solve is a solution object which can be handled just as in DifferentialEquations.jl. We can access mean values by indexing sol","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"sol[end]","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"or directly via sol.u","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"sol.u[end]","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"and similarly the time steps","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"sol.t[end]","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"But we didn't use probabilstic numerics to just compute means. In fact, sol is a probabilistic numerical ODE solution and it provides Gaussian distributions over solution values. These are stored in sol.pu:","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"sol.pu[end]","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"You can compute means, covariances, and standard deviations via Statistics.jl:","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"using Statistics\nmean(sol.pu[5])\ncov(sol.pu[5])\nstd(sol.pu[5])","category":"page"},{"location":"getting_started/#Dense-output","page":"Introduction to ODE Filters","title":"Dense output","text":"","category":"section"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"Probabilistic numerical ODE solvers approximate the posterior distribution","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"p Big( y(t) mid  doty(t_i) = f_theta(y(t_i) t_i)  Big)","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"which describes a posterior not just for the discrete steps but for any t in the continuous space t in 0 T; in classic ODE solvers, this is also known as \"interpolation\" or \"dense output\". The probabilistic solutions returned by our solvers can be interpolated as usual by treating them as functions, but they return Gaussian distributions","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"sol(0.45)\nmean(sol(0.45))","category":"page"},{"location":"getting_started/#Plotting","page":"Introduction to ODE Filters","title":"Plotting","text":"","category":"section"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"The result can be conveniently visualized through Plots.jl:","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"using Plots\nplot(sol)","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"A more detailed plotting tutorial for DifferentialEquations.jl + Plots.jl is provided here; most of the features work exactly as expected.","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"The uncertainties here are very low compared to the function value so we can't really see them. Just to demonstrate that they're there, let's solve the explicit EK0 solver, low order and higher tolerance levels:","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"using Logging; Logging.disable_logging(Logging.Warn) # hide\nsol = solve(prob, EK0(order=1), abstol=1e-2, reltol=1e-1)\nLogging.disable_logging(Logging.Debug) # hide\nplot(sol, denseplot=false)","category":"page"},{"location":"getting_started/","page":"Introduction to ODE Filters","title":"Introduction to ODE Filters","text":"There it is!","category":"page"},{"location":"solvers/#Solvers-and-Options","page":"Solvers and Options","title":"Solvers and Options","text":"","category":"section"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"ProbNumDiffEq.jl provides mainly the following two solvers, both based on extended Kalman filtering and smoothing. For the best results we suggest using EK1, but note that it relies on the Jacobian of the vector field.","category":"page"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"All solvers are compatible with DAEs in mass-matrix ODE form, and specialize on second-order ODEs.","category":"page"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"EK1\nEK0","category":"page"},{"location":"solvers/#ProbNumDiffEq.EK1","page":"Solvers and Options","title":"ProbNumDiffEq.EK1","text":"EK1(; order=3, smooth=true,\n      diffusionmodel=DynamicDiffusion(),\n      initialization=TaylorModeInit(),\n      kwargs...)\n\nGaussian ODE filter with first order vector field linearization\n\norder: Order of the integrated Brownian motion (IBM) prior.\nsmooth: Turn smoothing on/off; smoothing is required for dense output.\ndiffusionmodel: See Diffusion models and calibration.\ninitialization: See Initialization.\n\nSome additional kwargs relating to implicit solvers are supported; check out DifferentialEquations.jl's Extra Options page. Right now, we support autodiff, chunk_size, and diff_type. In particular, autodiff=false can come in handy to use finite differences instead of ForwardDiff.jl to compute Jacobians.\n\nReferences\n\n\n\n\n\n","category":"type"},{"location":"solvers/#ProbNumDiffEq.EK0","page":"Solvers and Options","title":"ProbNumDiffEq.EK0","text":"EK0(; order=3, smooth=true,\n      diffusionmodel=DynamicDiffusion(),\n      initialization=TaylorModeInit())\n\nGaussian ODE filter with zeroth order vector field linearization\n\norder: Order of the integrated Brownian motion (IBM) prior.\nsmooth: Turn smoothing on/off; smoothing is required for dense output.\ndiffusionmodel: See Diffusion models and calibration.\ninitialization: See Initialization.\n\nReferences\n\n\n\n\n\n","category":"type"},{"location":"solvers/#Diffusion-models-and-calibration","page":"Solvers and Options","title":"Diffusion models and calibration","text":"","category":"section"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"In a nutshell: \"Dynamic\" diffusion models allow the diffusion to change in-between each solver step and are recommended in combination with adaptive step-sizes. \"Fixed\" diffusion models keep the diffusion constant and can be helpful in fixed-step settings or for debugging by reducing the complexity of the models.","category":"page"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"For more information on the influence of diffusions, check out","category":"page"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"N. Bosch, P. Hennig, F. Tronarp: Calibrated Adaptive Probabilistic ODE Solvers (2021)","category":"page"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"DynamicDiffusion\nFixedDiffusion\nDynamicMVDiffusion\nFixedMVDiffusion","category":"page"},{"location":"solvers/#ProbNumDiffEq.DynamicDiffusion","page":"Solvers and Options","title":"ProbNumDiffEq.DynamicDiffusion","text":"DynamicDiffusion()\n\nRecommended with adaptive steps\n\nA local diffusion parameter is estimated at each step. Works well with adaptive steps.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#ProbNumDiffEq.FixedDiffusion","page":"Solvers and Options","title":"ProbNumDiffEq.FixedDiffusion","text":"FixedDiffusion(; initial_diffusion=1.0, calibrate=true)\n\nRecommended with fixed steps\n\nIf calibrate=true, the probabilistic solution is calibrated once at the end of the solve. An initial diffusion parameter can be passed, but this only has an effect if calibrate=false - which is typically not recommended.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#ProbNumDiffEq.DynamicMVDiffusion","page":"Solvers and Options","title":"ProbNumDiffEq.DynamicMVDiffusion","text":"DynamicMVDiffusion()\n\nOnly works with the EK0\n\nA multi-variate version of DynamicDiffusion, where instead of a scalar a vector-valued diffusion is estimated. When using the EK0, this can be helpful when the scales of the different dimensions vary a lot.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#ProbNumDiffEq.FixedMVDiffusion","page":"Solvers and Options","title":"ProbNumDiffEq.FixedMVDiffusion","text":"FixedMVDiffusion(; initial_diffusion=1.0, calibrate=true)\n\nOnly works with the EK0\n\nA multi-variate version of FixedDiffusion, where instead of a scalar a vector-valued diffusion is estimated. When using the EK0, this can be helpful when the scales of the different dimensions vary a lot.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#Initialization","page":"Solvers and Options","title":"Initialization","text":"","category":"section"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"TaylorModeInit\nClassicSolverInit","category":"page"},{"location":"solvers/#ProbNumDiffEq.TaylorModeInit","page":"Solvers and Options","title":"ProbNumDiffEq.TaylorModeInit","text":"TaylorModeInit()\n\nRecommended\n\nExact initialization via Taylor-mode automatic differentiation. Uses TaylorIntegration.jl. In case of errors, try ClassicSolverInit.\n\nReferences:\n\nN. Krämer, P. Hennig: Stable Implementation of Probabilistic ODE Solvers (2020)\n\n\n\n\n\n","category":"type"},{"location":"solvers/#ProbNumDiffEq.ClassicSolverInit","page":"Solvers and Options","title":"ProbNumDiffEq.ClassicSolverInit","text":"ClassicSolverInit(; alg=OrdinaryDiffEq.Tsit5(), init_on_du=false)\n\nExact initialization with a classic ODE solver. The solver to be used can be set with the alg keyword argument. init_on_du specifies if ForwardDiff.jl should be used to compute the jacobian and initialize on the exact second derivative.\n\nNot recommended for large solver orders, say order>4.\n\n\n\n\n\n","category":"type"},{"location":"dae/#Solving-DAEs-with-Probabilistic-Numerics","page":"Differential Algebraic Equations","title":"Solving DAEs with Probabilistic Numerics","text":"","category":"section"},{"location":"dae/","page":"Differential Algebraic Equations","title":"Differential Algebraic Equations","text":"ProbNumDiffEq.jl provides probabilistic numerical solvers for differential algebraic equations (DAEs). Currently, we recommend using the semi-implicit EK1 algorithm.","category":"page"},{"location":"dae/","page":"Differential Algebraic Equations","title":"Differential Algebraic Equations","text":"note: Note\nFor a more general tutorial on DAEs, solved with classic solvers, check out the DifferentialEquations.jl DEA tutorial.","category":"page"},{"location":"dae/#Solving-a-Mass-Matrix-DAE-with-the-EK1","page":"Differential Algebraic Equations","title":"Solving a Mass-Matrix DAE with the EK1","text":"","category":"section"},{"location":"dae/","page":"Differential Algebraic Equations","title":"Differential Algebraic Equations","text":"using ProbNumDiffEq, Plots\n\nfunction rober(du,u,p,t)\n  y₁,y₂,y₃ = u\n  k₁,k₂,k₃ = p\n  du[1] = -k₁*y₁ + k₃*y₂*y₃\n  du[2] =  k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2\n  du[3] =  y₁ + y₂ + y₃ - 1\n  nothing\nend\nM = [1. 0  0\n     0  1. 0\n     0  0  0]\nf = ODEFunction(rober,mass_matrix=M)\nprob_mm = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1e5), (0.04, 3e7, 1e4))\nusing Logging; Logging.disable_logging(Logging.Warn) # hide\nsol = solve(prob_mm, EK1(), reltol=1e-8, abstol=1e-8)\nLogging.disable_logging(Logging.Debug) # hide\nplot(sol, xscale=:log10, tspan=(1e-6, 1e5), layout=(3,1), legend=false, ylabel=[\"u₁(t)\" \"u₂(t)\" \"u₃(t)\"], xlabel=[\"\" \"\" \"t\"], denseplot=false)","category":"page"},{"location":"dae/#References","page":"Differential Algebraic Equations","title":"References","text":"","category":"section"},{"location":"dae/","page":"Differential Algebraic Equations","title":"Differential Algebraic Equations","text":"[1] N. Bosch, F. Tronarp, P. Hennig: Pick-and-Mix Information Operators for Probabilistic ODE Solvers (2022) (link)","category":"page"},{"location":"index3dae/#Fun-Fact:-Solving-an-Index-3-DAE","page":"-","title":"Fun-Fact: Solving an Index-3 DAE","text":"","category":"section"},{"location":"index3dae/","page":"-","title":"-","text":"The following is based on the \"Automatic Index Reduction of DAEs\" tutorial by ModelingToolkit.jl, which demonstrates how the classic Rodas4 solver fails to solve a DAE due to the fact that it is of index 3; which is why ModelingToolkit's automatic index reduction is so useful. It turns out that our probabilistic numerical solvers can directly solve the index-3 DAE:","category":"page"},{"location":"index3dae/","page":"-","title":"-","text":"using ProbNumDiffEq, LinearAlgebra\nfunction pendulum!(du, u, p, t)\n    x, dx, y, dy, T = u\n    g, L = p\n    du[1] = dx; du[2] = T*x\n    du[3] = dy; du[4] = T*y - g\n    du[5] = x^2 + y^2 - L^2\nend\npendulum_fun! = ODEFunction(pendulum!, mass_matrix=Diagonal([1,1,1,1,0]))\nu0 = [1.0, 0, 0, 0, 0]; p = [9.8, 1]; tspan = (0, 10.0)\npendulum_prob = ODEProblem(pendulum_fun!, u0, tspan, p)\nsol = solve(pendulum_prob, EK1())\nplot(sol)","category":"page"},{"location":"index3dae/","page":"-","title":"-","text":"prob_index3 = ODEProblem(modelingtoolkitize(pendulum_prob), Pair[], tspan)\n\ntraced_sys = modelingtoolkitize(pendulum_prob)\npendulum_sys = structural_simplify(dae_index_lowering(traced_sys))\nprob_index1 = ODEProblem(pendulum_sys, Pair[], tspan)\n\ntraced_sys = modelingtoolkitize(pendulum_prob)\npendulum_sys = structural_simplify(dae_index_lowering(traced_sys))\nprob_odae = ODAEProblem(pendulum_sys, Pair[], tspan)","category":"page"},{"location":"#Probabilistic-Numerical-Differential-Equation-Solvers","page":"Home","title":"Probabilistic Numerical Differential Equation Solvers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Banner)","category":"page"},{"location":"","page":"Home","title":"Home","text":"ProbNumDiffEq.jl provides probabilistic numerical solvers to the DifferentialEquations.jl ecosystem. The implemented ODE filters solve differential equations via Bayesian filtering and smoothing and compute not just a single point estimate of the true solution, but a posterior distribution that contains an estimate of its numerical approximation error.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For a short intro video, check out our poster presentation at JuliaCon2021.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"For more probabilistic numerics check out the ProbNum Python package. It implements probabilistic ODE solvers, but also probabilistic linear solvers, Bayesian quadrature, and many filtering and smoothing implementations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Run Julia, enter ] to bring up Julia's package manager, and add the ProbNumDiffEq.jl package:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]\n(v1.7) pkg> add ProbNumDiffEq.jl","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"For a quick introduction check out the \"Solving ODEs with Probabilistic Numerics\" tutorial.","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Two extended Kalman filtering-based probabilistic solvers: the explicit EK0 and semi-implicit EK1.\nAdaptive step-size selection (by default with PI control)\nOn-line uncertainty calibration, for multiple different measurement models\nDense output\nSampling from the solution\nCallback support\nConvenient plotting through a Plots.jl recipe\nAutomatic differentiation via ForwardDiff.jl\nSupports arbitrary precision numbers via BigFloats.jl\nSpecialized solvers for second-order ODEs (demo will be added)\nCompatible with DAEs in mass-matrix ODE form (demo will be added)","category":"page"},{"location":"#Benchmarks","page":"Home","title":"Benchmarks","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Multi-Language Wrapper Benchmark: ProbNumDiffEq.jl vs. OrdinaryDiffEq.jl, Hairer's FORTRAN solvers, Sundials, LSODA, MATLAB, and SciPy.","category":"page"},{"location":"#references","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"N. Bosch, F. Tronarp, P. Hennig: Pick-and-Mix Information Operators for Probabilistic ODE Solvers (2022)\nN. Krämer, N. Bosch, J. Schmidt, P. Hennig: Probabilistic ODE Solutions in Millions of Dimensions (2021)\nN. Bosch, P. Hennig, F. Tronarp: Calibrated Adaptive Probabilistic ODE Solvers (2021)\nF. Tronarp, S. Särkkä, and P. Hennig: Bayesian ODE Solvers: The Maximum A Posteriori Estimate (2021)\nN. Krämer, P. Hennig: Stable Implementation of Probabilistic ODE Solvers (2020)\nH. Kersting, T. J. Sullivan, and P. Hennig: Convergence Rates of Gaussian Ode Filters (2020)\nF. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective (2019)\nM. Schober, S. Särkkä, and P. Hennig: A Probabilistic Model for the Numerical Solution of Initial Value Problems (2018)","category":"page"},{"location":"","page":"Home","title":"Home","text":"A much more detailed list of references, not only on ODE filters but on probabilistic numerics in general, can be found on the probabilistic-numerics.org homepage.","category":"page"}]
}
