# ============================================================
# EE5311 CA1 — 1D Heat Equation PINN
# Solves: ∂u/∂t = α ∂²u/∂x²
# IC: u(0,x) = sin(πx),  BC: u(t,0) = u(t,1) = 0
# Analytical: u(t,x) = exp(-α π² t) sin(πx)
# ============================================================

using NeuralPDE, Lux, ModelingToolkit, Optimization
using OptimizationOptimisers, OptimizationOptimJL, LineSearches
using DomainSets: Interval
using Plots

# ── Step 1: 用 ModelingToolkit 符号化定义 PDE ──────────────
@parameters t x
@variables u(..)

Dt  = Differential(t)
Dxx = Differential(x)^2

α = 0.1
eq = Dt(u(t, x)) ~ α * Dxx(u(t, x))

# ── Step 2: 初始条件 + 边界条件 ────────────────────────────
bcs = [
    u(0, x) ~ sin(π * x),
    u(t, 0) ~ 0.0,
    u(t, 1) ~ 0.0
]

# ── Step 3: 定义求解域 ──────────────────────────────────────
domains = [
    t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)
]

# ── Step 4: 定义神经网络 ────────────────────────────────────
chain = Lux.Chain(
    Dense(2, 20, Lux.tanh),
    Dense(20, 20, Lux.tanh),
    Dense(20, 1)
)

# ── Step 5: 选择训练策略 + 离散化 ──────────────────────────
strategy       = QuasiRandomTraining(200)
discretization = PhysicsInformedNN(chain, strategy)

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)

# ── Step 6: 记录 Loss 曲线 ──────────────────────────────────
global loss_history = Float64[]

callback = function (p, l)
    push!(loss_history, l)
    if length(loss_history) % 500 == 0
        println("Iter $(length(loss_history))  |  Loss = $(round(l, sigdigits=4))")
    end
    return false
end

# ── Step 7: 两阶段训练 ─────────────────────────────────────
println("=== Phase 1: Adam (3000 iters) ===")
res = Optimization.solve(
    prob,
    OptimizationOptimisers.Adam(0.005);
    maxiters = 3000,
    callback = callback
)

println("\n=== Phase 2: L-BFGS (1000 iters) ===")
prob2 = remake(prob, u0 = res.minimizer)
res2  = Optimization.solve(
    prob2,
    OptimizationOptimJL.LBFGS();
    maxiters = 1000,
    callback = callback
)

println("\n✓ Training complete. Final loss = $(round(res2.objective, sigdigits=4))")

# ── Step 8: 提取并计算误差 ─────────────────────────────────
phi = discretization.phi
ts  = 0:0.01:1
xs  = 0:0.01:1

u_pred  = [first(phi([t_val, x_val], res2.minimizer))
           for t_val in ts, x_val in xs]

u_exact = [exp(-α * π^2 * t_val) * sin(π * x_val)
           for t_val in ts, x_val in xs]

abs_error  = abs.(u_pred .- u_exact)
max_error  = maximum(abs_error)
mean_error = sum(abs_error) / length(abs_error)

println("\nMax  absolute error: $(round(max_error,  sigdigits=3))")
println("Mean absolute error: $(round(mean_error, sigdigits=3))")

# ── Step 9: 画图 ────────────────────────────────────────────

# 图1：PINN解 vs 解析解 三维曲面对比
p1 = surface(collect(xs), collect(ts), u_pred,
    xlabel = "x", ylabel = "t", zlabel = "u",
    title  = "PINN Solution",
    color  = :viridis, camera = (30, 60))

p2 = surface(collect(xs), collect(ts), u_exact,
    xlabel = "x", ylabel = "t", zlabel = "u",
    title  = "Analytical Solution",
    color  = :plasma, camera = (30, 60))

fig1 = plot(p1, p2, layout = (1, 2), size = (1000, 420),
            plot_title = "Figure 1 — 1D Heat Equation: PINN vs Analytical")
savefig(fig1, "fig1_surface.png")
println("Saved fig1_surface.png")

# 图2：绝对误差热力图
fig2 = heatmap(collect(xs), collect(ts), abs_error,
    xlabel = "x", ylabel = "t",
    title  = "Figure 2 — Absolute Error  (max = $(round(max_error, sigdigits=2)))",
    color  = :hot, clims = (0, max_error), size = (650, 500))
savefig(fig2, "fig2_error.png")
println("Saved fig2_error.png")

# 图3：Loss 训练曲线（Adam + L-BFGS）
n_adam = 3000
fig3 = plot(1:n_adam, loss_history[1:n_adam],
    label = "Adam (lr=0.005)", color = :steelblue,
    yaxis = :log10, lw = 1.5)
if length(loss_history) > n_adam
    plot!(fig3,
          (n_adam + 1):length(loss_history),
          loss_history[(n_adam + 1):end],
          label = "L-BFGS", color = :darkorange, lw = 1.5)
end
vline!(fig3, [n_adam],
    color = :gray, ls = :dash,
    label = "Switch Adam → L-BFGS", lw = 1)
plot!(fig3,
    xlabel = "Iteration", ylabel = "Loss (log scale)",
    title  = "Figure 3 — PINN Training Loss Curve",
    size   = (800, 380), legend = :topright, grid = true)
savefig(fig3, "fig3_loss.png")
println("Saved fig3_loss.png")

# 图4：三个时间截面对比（PINN vs Analytical）
xs_vec  = collect(xs)
ts_list = [0.1, 0.5, 1.0]
colors  = [:red, :green, :blue]

fig4 = plot(size = (850, 420),
    title  = "Figure 4 — Solution at Fixed Time Slices",
    xlabel = "x", ylabel = "u(t,x)", legend = :topright)

for (t_val, col) in zip(ts_list, colors)
    t_idx        = argmin(abs.(collect(ts) .- t_val))
    u_pred_line  = u_pred[t_idx, :]
    u_exact_line = u_exact[t_idx, :]
    plot!(fig4, xs_vec, u_exact_line,
          label = "Analytical t=$(t_val)", color = col, lw = 2,   ls = :solid)
    plot!(fig4, xs_vec, u_pred_line,
          label = "PINN      t=$(t_val)", color = col, lw = 1.5, ls = :dash)
end
savefig(fig4, "fig4_slices.png")
println("Saved fig4_slices.png")

println("\n✓ All figures saved to current directory.")