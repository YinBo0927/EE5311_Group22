# ============================================================
# EE5311 CA1 — 汇总图表
# 包括：2D 热方程可视化 + 所有对比图拼接
# ============================================================

using Plots

α = 0.1

# ── 2D 热方程解析解可视化 ───────────────────────────────────
# ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
# 解析解：u = exp(-2α π² t) sin(πx) sin(πy)

xs2 = range(0, 1, length = 51)
ys2 = range(0, 1, length = 51)

# 三个时间点的热力图
t_list = [0.05, 0.1, 0.3]
plots_2d = []

for t_val in t_list
    u2 = [exp(-2*α*π^2*t_val) * sin(π*xv) * sin(π*yv)
          for xv in xs2, yv in ys2]
    p = heatmap(collect(xs2), collect(ys2), u2,
        xlabel = "x", ylabel = "y",
        title  = "t = $(t_val)  (u_max=$(round(maximum(u2), digits=3)))",
        color  = :viridis, aspect_ratio = :equal,
        clims  = (0, 1))
    push!(plots_2d, p)
end

fig_2d = plot(plots_2d..., layout = (1, 3), size = (1100, 380),
    plot_title = "Figure 6 — 2D Heat Equation Analytical Solution")
savefig(fig_2d, "fig6_2d_heat.png")
println("Saved fig6_2d_heat.png")

# ── 2D 热方程三维曲面（t=0.1）───────────────────────────────
u2_surf = [exp(-2*α*π^2*0.1) * sin(π*xv) * sin(π*yv)
           for xv in xs2, yv in ys2]

fig_2d_surf = surface(collect(xs2), collect(ys2), u2_surf,
    xlabel = "x", ylabel = "y", zlabel = "u",
    title  = "2D Heat Equation — u(0.1, x, y)",
    color  = :viridis, camera = (35, 50), size = (600, 500))
savefig(fig_2d_surf, "fig_2d_surface.png")
println("Saved fig_2d_surface.png")

# ── Activation function comparison ──────────────────────────
# 可视化 tanh vs sigmoid 的二阶导数（解释为何用 tanh）
z = range(-3, 3, length = 300)

tanh_val   = tanh.(z)
sig_val    = 1 ./ (1 .+ exp.(-z))
dtanh2_dz2 = -2 .* tanh_val .* (1 .- tanh_val.^2)       # d²(tanh)/dz²
dsig2_dz2  = sig_val .* (1 .- sig_val) .* (1 .- 2 .* sig_val)  # d²(σ)/dz²

fig_act = plot(layout = (1, 2), size = (900, 380))
plot!(fig_act[1], collect(z), [tanh_val, sig_val],
    label = ["tanh" "sigmoid"], color = [:blue :red],
    lw = 2, xlabel = "z", ylabel = "f(z)", title = "Activation Functions")
plot!(fig_act[2], collect(z), [dtanh2_dz2, dsig2_dz2],
    label = ["d²tanh/dz²" "d²sigmoid/dz²"], color = [:blue :red],
    lw = 2, xlabel = "z", ylabel = "f''(z)",
    title = "Second Derivatives (key for PDE loss)")
savefig(fig_act, "fig_activation.png")
println("Saved fig_activation.png")

println("\n✓ All supplementary figures saved.")