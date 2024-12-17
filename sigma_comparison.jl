using Pkg
Pkg.activate(".")
using NPZ
using Plots
using LinearAlgebra
using Interpolations

# Read data
my_sigma = npzread("sofia.npy")
original_sigma = npzread("davide.npy")

# Heatmap of my_sigma
heatmap1 = heatmap(log10.(abs.(my_sigma)))
savefig(heatmap1, "plots/my_sigma_heatmap.pdf")

# Heatmap of original_sigma
heatmap2 = heatmap(log10.(abs.(original_sigma)))
savefig(heatmap2, "plots/original_sigma_heatmap.pdf")

z_grid_ssc_integrands = npzread("z.npy")
R_grid_ssc_integrands = npzread("R.npy")

# First plot: sigma^2_B
figure1 = plot(z_grid_ssc_integrands, my_sigma[:, end], 
               yscale=:log10, 
               label="Me", 
               xlabel="z", 
               title="sigma^2_B(R=1)")
plot!(z_grid_ssc_integrands, diag(original_sigma), 
      label="Davide")
savefig(figure1, "plots/sigma2_B_R1.pdf")

# Second plot: Residuals
residuals = 100 .* (1 .- my_sigma[:, end] ./ diag(original_sigma))
figure2 = plot(z_grid_ssc_integrands, residuals, 
               xlabel="z", 
               title="Residuals")
savefig(figure2, "plots/residuals.pdf")

# Third plot: R_cut at fixed z
figure3 = plot(R_grid_ssc_integrands[2:end], my_sigma[50, 2:end], 
               xlabel="R", 
               title="R_cut at fixed z = $(z_grid_ssc_integrands[50])")
savefig(figure3, "plots/R_cut_fixed_z.pdf")


x = LinRange(first(z_grid_ssc_integrands), last(z_grid_ssc_integrands), length(z_grid_ssc_integrands))
InterpPmm = Interpolations.interpolate(original_sigma, BSpline(Cubic(Line(OnGrid()))))
InterpPmm = scale(InterpPmm, (x, x))
InterpPmm = Interpolations.extrapolate(InterpPmm, Line())

interpolated_sigma = zeros(length(z_grid_ssc_integrands),length(R_grid_ssc_integrands))

for (iR, R) in enumerate(R_grid_ssc_integrands)
    for (iz, z) in enumerate(z_grid_ssc_integrands)
        interpolated_sigma[iz, iR] = InterpPmm(z,R*z)
    end
end

heatmap3 = heatmap(log10.(abs.(interpolated_sigma)))
savefig(heatmap3, "plots/interpolated_sigma_heatmap.pdf")

heatmap4 = heatmap(log10.(abs.(1 .- interpolated_sigma./ my_sigma)))
savefig(heatmap4, "plots/residuals_heatmap.pdf")

idx = 38
p = plot(z_grid_ssc_integrands, abs.(my_sigma[:,idx]), label="Mine", yscale=:log10)
plot!(p, z_grid_ssc_integrands, abs.(interpolated_sigma[:,idx]), label="Interpolated") 
savefig(p, "plots/p.pdf")
#plot!(p, z_grid_ssc_integrands, abs.(interpolated_sigma[:,idx]), label="Interpolated") 

my_cov = npzread("cov_sofia.npy")
original_cov = npzread("cov_original.npy")

fig = heatmap(log10.(abs.(my_cov)), title = "SSC 2D sofia")
savefig(fig, "plots/my_cov.pdf")

fig = heatmap(log10.(abs.(original_cov)), title = "SSC 2D original")
savefig(fig, "plots/original_cov.pdf")

p = plot(log10.(abs.(diag(original_cov))), xscale=:log10,label = "Original", title="SSC diagonal")
plot!(p, log10.(abs.(diag(my_cov))), label="Mine", linestyle=:dot)
savefig(p, "plots/SSC_diag.pdf")

p = plot(100 * (1 .- diag(my_cov)./diag(original_cov)),label = nothing, title="SSC diagonal residuals", xscale=:log10)
savefig(p, "plots/diag_residuals.pdf")

idx = 98
p = plot(log10.(abs.(original_cov[idx,:])), xscale=:log10,label = "Original", title="SSC slice")
plot!(p, log10.(abs.(my_cov[idx,:])), label="Mine")
savefig(p, "plots/slice.pdf")


