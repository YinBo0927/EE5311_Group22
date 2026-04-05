# EE5311 CA1 Blog — Group 22

**Beyond ODEs: Solving the Heat Equation with Physics-Informed Neural Networks in Julia**

Yin Bo, Fang An, Liu Cheng, Niu Cheng, Jiang Yuxing

National University of Singapore, 2025

---

## Repository Structure

```
EE5311_Group22/
├── index.html                  # Blog webpage (open in browser)
├── code/                       # All Julia source code
│   ├── pinn_1d.jl              # 1D heat equation PINN solver
│   ├── pinn_inverse.jl         # Inverse problem: estimate α from noisy data
│   ├── fdm_compare.jl          # Finite difference method comparison
│   └── plot_all.jl             # 2D heat equation visualisation & supplementary plots
├── outputs/                    # Training logs (actual run results)
│   ├── pinn_1d_output.txt      # Forward problem training output
│   └── pinn_inverse_output.txt # Inverse problem training output
├── static/
│   ├── images/                 # Figures used in the blog
│   │   ├── fig1_surface.png    # PINN vs analytical solution (3D surface)
│   │   ├── fig2_error.png      # Absolute error heatmap
│   │   ├── fig3_loss.png       # Training loss curve (Adam + L-BFGS)
│   │   ├── fig4_slices.png     # Solution at fixed time slices
│   │   ├── fig5_alpha_conv.png # α convergence (inverse problem)
│   │   └── fig6_inverse_fit.png# Inverse PINN fit vs noisy data
│   ├── css/                    # Stylesheets (Bulma framework)
│   └── js/                     # JavaScript (carousel, slider)
└── README.md                   # This file
```

## Viewing the Blog

Open `index.html` in any modern browser. No server needed — it is a static HTML page.

If deploying to GitHub Pages, push this repository and enable Pages from the `master` branch. The blog will be available at:

```
https://<username>.github.io/EE5311_Group22/
```

## Running the Code

### Prerequisites

- **Julia 1.12+** (tested on 1.12.5)
- ~3 GB disk space for packages
- ~2-3 GB RAM during training

### Step 1: Install Dependencies

Run once to install all required packages (~10-15 minutes for first-time compilation):

```bash
julia -e '
using Pkg
Pkg.add([
    "NeuralPDE", "Lux", "ModelingToolkit", "Optimization",
    "OptimizationOptimisers", "OptimizationOptimJL", "LineSearches",
    "DomainSets", "Plots", "ComponentArrays"
])
'
```

### Step 2: Run the Scripts

All scripts should be run from the `code/` directory. Figures are saved to the current working directory.

**1D Heat Equation (Forward Problem):**

```bash
cd code
julia pinn_1d.jl
```

- Trains a PINN to solve the 1D heat equation
- Adam (3000 iters) + L-BFGS (1000 iters), ~2-3 minutes total
- Outputs: `fig1_surface.png`, `fig2_error.png`, `fig3_loss.png`, `fig4_slices.png`

**Inverse Problem (Estimate α from Noisy Data):**

```bash
julia pinn_inverse.jl
```

- Generates 300 synthetic observations with 2% noise
- Estimates thermal diffusivity α starting from initial guess 0.5 (true: 0.1)
- Adam (5000 iters) + L-BFGS (2000 iters), ~5-8 minutes total
- Outputs: `fig5_alpha_conv.png`, `fig6_inverse_fit.png`
- To test other noise levels, edit `σ_noise` on line 22 (e.g. `0.0` or `0.05`)

**FDM Comparison:**

```bash
julia fdm_compare.jl
```

- Solves the same 1D heat equation using explicit finite differences
- Runs in < 1 second
- Outputs: `fig_fdm_surface.png`, `fig_fdm_error.png`, `fig_fdm_compare.png`

**Supplementary Plots (2D Heat Equation + Activation Functions):**

```bash
julia plot_all.jl
```

- 2D heat equation analytical solution visualisation
- Activation function comparison (tanh vs sigmoid second derivatives)
- Outputs: `fig6_2d_heat.png`, `fig_2d_surface.png`, `fig_activation.png`

### Updating the Blog Figures

After running the scripts, copy the generated figures to the blog:

```bash
cp fig*.png ../static/images/
```

Then refresh `index.html` in your browser.

## Key Results

| Experiment | Metric | Value |
|---|---|---|
| 1D PINN (forward) | Max absolute error | 2.23e-3 |
| 1D PINN (forward) | Final loss | 2.055e-5 |
| FDM (Nx=100) | Max absolute error | ~1e-5 |
| Inverse problem (σ=0.02) | Estimated α | 0.0993 |
| Inverse problem (σ=0.02) | Relative error | 0.7% |

## Contributions

| Member | Sections |
|---|---|
| **Yin Bo** | §6: Inverse problem formulation & implementation |
| **Fang An** | §1: Problem formulation, mathematical background |
| **Liu Cheng** | §2: 1D PINN implementation, package compatibility |
| **Niu Cheng** | §3 & §4: 2D extension, FDM comparison |
| **Jiang Yuxing** | §5: Training experiments, practical guidelines |

All members contributed to proofreading and revising the final blog.

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
2. Zubov, K., et al. (2021). NeuralPDE: Automating Physics-Informed Neural Networks. *arXiv:2107.09443*.
3. EE5311 Course Notes v3.0.0, §3.5.
