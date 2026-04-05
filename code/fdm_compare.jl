# ============================================================
# EE5311 CA1 — Finite Difference Method (FDM)
# Explicit scheme for 1D heat equation
# Stability condition: r = α·Δt/Δx² ≤ 0.5
# ============================================================

using Plots

α = 0.1

# ── FDM 求解器 ──────────────────────────────────────────────
function solve_heat_fdm(α, Nx, Nt, T_end)
    dx = 1.0 / Nx
    dt = T_end / Nt
    r  = α * dt / dx^2

    # CFL 稳定性检验
    @assert r ≤ 0.5 "不稳定！r = $r > 0.5，请增大 Nt 或减小 Nx。"
    println("FDM: Nx=$Nx, Nt=$Nt, r=$(round(r, digits=4))  ← 稳定")

    x = range(0, 1, length = Nx + 1)
    t = range(0, T_end, length = Nt + 1)
    u = zeros(Float64, Nx + 1, Nt + 1)

    # 初始条件
    u[:, 1] = sin.(π .* x)

    # 时间步进（边界保持为 0）
    for n in 1:Nt
        for i in 2:Nx
            u[i, n+1] = u[i, n] + r * (u[i+1, n] - 2*u[i, n] + u[i-1, n])
        end
    end

    return collect(x), collect(t), u
end

# ── 运行 FDM ────────────────────────────────────────────────
x_fdm, t_fdm, u_fdm = solve_heat_fdm(α, 100, 5000, 1.0)

# ── 计算解析解 ──────────────────────────────────────────────
u_exact_fdm = [exp(-α * π^2 * t_val) * sin(π * x_val)
               for x_val in x_fdm, t_val in t_fdm]

# FDM 误差
fdm_error = abs.(u_fdm .- u_exact_fdm)
println("FDM Max  error: $(round(maximum(fdm_error), sigdigits=3))")
println("FDM Mean error: $(round(sum(fdm_error)/length(fdm_error), sigdigits=3))")

# ── 画图：FDM 三维曲面 ──────────────────────────────────────
fig_fdm = surface(x_fdm, t_fdm, u_fdm',
    xlabel = "x", ylabel = "t", zlabel = "u",
    title  = "FDM Solution (Nx=100, Nt=5000)",
    color  = :viridis, camera = (30, 60), size = (600, 450))
savefig(fig_fdm, "fig_fdm_surface.png")

# ── 画图：FDM 误差热力图 ─────────────────────────────────────
fig_fdm_err = heatmap(x_fdm, t_fdm, fdm_error',
    xlabel = "x", ylabel = "t",
    title  = "FDM Absolute Error (max=$(round(maximum(fdm_error), sigdigits=2)))",
    color  = :hot, size = (650, 480))
savefig(fig_fdm_err, "fig_fdm_error.png")

# ── 画图：三时间截面对比 ────────────────────────────────────
fig_compare = plot(size = (900, 430),
    title  = "Figure 4 — Analytical vs FDM at Fixed Time Slices",
    xlabel = "x", ylabel = "u(t,x)", legend = :topright)

for (t_val, col) in zip([0.1, 0.5, 1.0], [:red, :green, :blue])
    t_idx = findfirst(t -> abs(t - t_val) < 1e-4 * 1.5, t_fdm)
    if t_idx === nothing
        t_idx = argmin(abs.(t_fdm .- t_val))
    end
    plot!(fig_compare, x_fdm,
          [exp(-α * π^2 * t_val) * sin(π * xv) for xv in x_fdm],
          label = "Analytical t=$(t_val)", color = col, lw = 2)
    plot!(fig_compare, x_fdm, u_fdm[:, t_idx],
          label = "FDM t=$(t_val)",        color = col, lw = 1.5, ls = :dash)
end
savefig(fig_compare, "fig_fdm_compare.png")

println("\n✓ FDM figures saved.")

# ── 打印对比表格 ─────────────────────────────────────────────
println("\n" * "="^60)
println("METHOD COMPARISON TABLE")
println("="^60)
println("Aspect              | FDM (Nx=100)    | PINN")
println("-"^60)
println("Max abs error       | ~6e-6           | ~5e-4")
println("Computation time    | < 1 second      | Minutes (training)")
println("Mesh dependency     | Yes             | No (mesh-free)")
println("Irregular geometry  | Needs meshing   | Naturally handled")
println("Inverse problems    | Separate setup  | Embed in loss")
println("="^60)