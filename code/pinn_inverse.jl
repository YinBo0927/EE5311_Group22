# ============================================================
# EE5311 CA1 — Inverse Heat Conduction Problem
# Given noisy observations of u(t,x), estimate unknown α
# Forward PDE: ∂u/∂t = α ∂²u/∂x²
# True α = 0.1, initial guess = 0.5
#
# NOTE: 使用多输出模式 (chain 包在数组中) 以兼容 NeuralPDE v5
#       的 param_estim 代码路径，详见代码注释。
# ============================================================

using NeuralPDE, Lux, ModelingToolkit, Optimization
using OptimizationOptimisers, OptimizationOptimJL, LineSearches
using DomainSets: Interval
using Random, Plots

Random.seed!(42)

# ── Step 1: 生成合成观测数据（模拟传感器测量）────────────────
α_true = 0.1
u_exact(t, x) = exp(-α_true * π^2 * t) * sin(π * x)

n_obs   = 300
t_obs   = rand(n_obs)
x_obs   = rand(n_obs)
σ_noise = 0.02                        # 噪声水平，可改为 0.0 / 0.05 进行对比

u_obs = [u_exact(t_obs[i], x_obs[i]) + σ_noise * randn() for i in 1:n_obs]

println("Generated $n_obs observations with noise σ = $σ_noise")
println("True α = $α_true,  initial guess α₀ = 0.5")

# ── Step 2: 定义含未知参数 α_inv 的 PDE ─────────────────────
@parameters t x
@parameters α_inv                      # 待估计的热扩散系数
@variables  u(..)

Dt  = Differential(t)
Dxx = Differential(x)^2

eq = Dt(u(t, x)) ~ α_inv * Dxx(u(t, x))   # α_inv 替代固定的 0.1

bcs = [
    u(0, x) ~ sin(π * x),
    u(t, 0) ~ 0.0,
    u(t, 1) ~ 0.0
]

domains = [
    t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)
]

# ── Step 3: 神经网络 ────────────────────────────────────────
# 包在数组中 → 触发 NeuralPDE 的多输出代码路径
# 这样 param_estim=true 时内部会正确分解 θ.depvar.u
inner_chain = Lux.Chain(
    Dense(2, 20, Lux.tanh),
    Dense(20, 20, Lux.tanh),
    Dense(20, 1)
)
chain = [inner_chain]

# ── Step 4: 数据拟合损失（附加到 PINN 损失上）──────────────
# additional_loss 签名: (phi, θ_depvar, p_pde)
#   phi       = Phi 数组 (多输出模式)
#   θ_depvar  = θ.depvar，包含 .u 字段（神经网络参数）
#   p_pde     = θ.p（PDE 待估参数向量）
function data_loss(phi, θ_depvar, p_pde)
    pred = [first(phi[1]([t_obs[i], x_obs[i]], θ_depvar.u)) for i in 1:n_obs]
    return 10.0 * sum(abs2, pred .- u_obs) / n_obs
end

# ── Step 5: 以反问题模式离散化 ──────────────────────────────
strategy       = QuasiRandomTraining(300)
discretization = PhysicsInformedNN(chain, strategy;
                                    param_estim     = true,
                                    additional_loss = data_loss)

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)],
                               [α_inv];
                               initial_conditions = Dict(α_inv => 0.5))

prob = discretize(pde_system, discretization)

# ── Step 6: 训练，同时记录 loss 和 α 的收敛历史 ─────────────
global loss_hist = Float64[]
global α_hist    = Float64[]

callback = function (state, l)
    push!(loss_hist, l)
    # state 是 OptimizationState; state.u 是当前优化变量 ComponentArray
    # state.u.p[1] 是 PDE 参数 α 的当前估计值
    α_cur = state.u.p[1]
    push!(α_hist, α_cur)
    if length(loss_hist) % 500 == 0
        println("Iter $(length(loss_hist))  |  Loss = $(round(l, sigdigits=4))  " *
                "|  α_est = $(round(α_cur, sigdigits=4))")
    end
    return false
end

println("\n=== Phase 1: Adam (5000 iters) ===")
res = Optimization.solve(prob,
    OptimizationOptimisers.Adam(0.003);
    maxiters = 5000, callback = callback)

println("\n=== Phase 2: L-BFGS (2000 iters) ===")
prob2 = remake(prob, u0 = res.minimizer)
res2  = Optimization.solve(prob2,
    OptimizationOptimJL.LBFGS();
    maxiters = 2000, callback = callback)

# ── 输出最终结果 ──────────────────────────────────────────
α_final = res2.minimizer.p[1]
rel_err = abs(α_final - α_true) / α_true * 100

println("\n" * "="^50)
println("  INVERSE PROBLEM RESULT")
println("="^50)
println("  True α       = $α_true")
println("  Estimated α  = $(round(α_final, sigdigits=5))")
println("  Relative err = $(round(rel_err, sigdigits=3))%")
println("="^50)

# ── Step 7: 画图 ──────────────────────────────────────────

# 图5：α 估计值随迭代次数的收敛曲线
n_adam = 5000
fig5 = plot(α_hist, lw = 2, color = :blue, label = "α estimate")
hline!([α_true], lw = 2, ls = :dash, color = :red, label = "True α = $α_true")
vline!([n_adam], color = :gray, ls = :dot, label = "Adam → L-BFGS", lw = 1)
plot!(xlabel = "Iteration", ylabel = "α",
      title  = "Figure 5 — Convergence of Estimated Thermal Diffusivity",
      size   = (750, 400), legend = :right, grid = true)
savefig(fig5, "fig5_alpha_conv.png")
println("Saved fig5_alpha_conv.png")

# 图6：t = 0.1 处的拟合效果 + 观测散点
phi_arr  = discretization.phi          # Phi 数组
xs_plot  = 0:0.01:1
t_fix    = 0.1

u_pred_line  = [first(phi_arr[1]([t_fix, xv], res2.minimizer.depvar.u)) for xv in xs_plot]
u_exact_line = [u_exact(t_fix, xv) for xv in xs_plot]

# 选取 t ∈ [0.05, 0.15] 的观测点做散点图
mask = abs.(t_obs .- t_fix) .< 0.05
fig6 = scatter(x_obs[mask], u_obs[mask],
    ms = 4, alpha = 0.5, color = :gray, label = "Noisy obs (σ=$σ_noise)")
plot!(collect(xs_plot), u_exact_line,
    lw = 2.5, color = :red, label = "Exact (α = $α_true)")
plot!(collect(xs_plot), u_pred_line,
    lw = 2, ls = :dash, color = :blue,
    label = "PINN (α̂ = $(round(α_final, sigdigits=3)))")
plot!(xlabel = "x", ylabel = "u(0.1, x)",
      title  = "Figure 6 — Inverse PINN Prediction vs Noisy Data at t = 0.1",
      size   = (750, 420), legend = :topright)
savefig(fig6, "fig6_inverse_fit.png")
println("Saved fig6_inverse_fit.png")

println("\n✓ All inverse problem figures saved.")
println("Tip: 修改 σ_noise (0.0 / 0.02 / 0.05) 重新运行可生成噪声对比表格数据")
