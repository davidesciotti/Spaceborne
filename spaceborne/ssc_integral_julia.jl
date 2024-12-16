using NPZ
using LoopVectorization
using YAML
using DataInterpolations

function SSC_integral_6D_trapz(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array::Array)
    """ "brute-force" implementation, returns a 6D array. many args are unnecessary, but I keep the same format for a 
    more agile comparison against the other functions
    """
    # TODO zbins should be passed to the functions, args should be
    zbins = size(d2ClAB_dVddeltab, 2)
    result = zeros(nbl, nbl, zbins, zbins, zbins, zbins)

    @tturbo for ell1 in 1:nbl
        for ell2 in 1:nbl
            for zi in 1:zbins
                for zj in 1:zbins
                    for zk in 1:zbins
                        for zl in 1:zbins
                            for z1_idx in 1:z_steps
                                for z2_idx in 1:z_steps
                                    result[ell1, ell2, zi, zj, zk, zl] += cl_integral_prefactor[z1_idx] * cl_integral_prefactor[z2_idx] *
                                    d2ClAB_dVddeltab[ell1, zi, zj, z1_idx] * d2ClCD_dVddeltab[ell2, zk, zl, z2_idx] * sigma2[z1_idx, z2_idx]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return (dz^2) .* result
end

function get_simpson_weights(n::Int)
    number_intervals = floor((n-1)/2)
    weight_array = zeros(n)
    if n == number_intervals*2+1
        for i in 1:number_intervals
            weight_array[Int((i-1)*2+1)] += 1/3
            weight_array[Int((i-1)*2+2)] += 4/3
            weight_array[Int((i-1)*2+3)] += 1/3
        end
    else
        weight_array[1] += 0.5
        weight_array[2] += 0.5
        for i in 1:number_intervals
            weight_array[Int((i-1)*2+1)+1] += 1/3
            weight_array[Int((i-1)*2+2)+1] += 4/3
            weight_array[Int((i-1)*2+3)+1] += 1/3
        end
        weight_array[length(weight_array)]   += 0.5
        weight_array[length(weight_array)-1] += 0.5
        for i in 1:number_intervals
            weight_array[Int((i-1)*2+1)] += 1/3
            weight_array[Int((i-1)*2+2)] += 4/3
            weight_array[Int((i-1)*2+3)] += 1/3
        end
        weight_array ./= 2
    end
    return weight_array
end


function SSC_integral_4D_trapz(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array::Array)
    """ this version takes advantage of the symmetries between redshift pairs.
    """

    zpairs_AB = size(ind_AB, 1)
    zpairs_CD = size(ind_CD, 1)
    num_col = size(ind_AB, 2)

    dz = z_array[2]-z_array[1]

    result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

    @tturbo for ell1 in 1:nbl
        for ell2 in 1:nbl  # this could be further optimized by computing only upper triangular ells, but not with tturbo?
            for zij in 1:zpairs_AB
                for zkl in 1:zpairs_CD
                    for z1_idx in 1:z_steps
                        for z2_idx in 1:z_steps

                            zi, zj, zk, zl = ind_AB[zij, num_col - 1], ind_AB[zij, num_col], ind_CD[zkl, num_col - 1], ind_CD[zkl, num_col]

                            result[ell1, ell2, zij, zkl] += cl_integral_prefactor[z1_idx] * cl_integral_prefactor[z2_idx] *
                            d2ClAB_dVddeltab[ell1, zi, zj, z1_idx] * d2ClCD_dVddeltab[ell2, zk, zl, z2_idx] * sigma2[z1_idx, z2_idx]

                        end
                    end
                end
            end
        end
    end
    return (dz^2) .* result
end

function SSC_integral_4D_simps(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array::Array)
    """ this version takes advantage of the symmetries between redshift pairs.
    """

    simpson_weights = get_simpson_weights(length(z_array))
    z_step = (last(z_array)-first(z_array)) /(length(z_array)-1)


    zpairs_AB = size(ind_AB, 1)
    zpairs_CD = size(ind_CD, 1)
    num_col = size(ind_AB, 2)

    result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

    @tturbo for ell1 in 1:nbl
        for ell2 in 1:nbl  # this could be further optimized by computing only upper triangular ells (for LLLL, GLGL, GGGG only), but not with tturbo
            for zij in 1:zpairs_AB
                for zkl in 1:zpairs_CD
                    for z1_idx in 1:z_steps
                        for z2_idx in 1:z_steps

                            zi, zj, zk, zl = ind_AB[zij, num_col - 1], ind_AB[zij, num_col], ind_CD[zkl, num_col - 1], ind_CD[zkl, num_col]

                            result[ell1, ell2, zij, zkl] += cl_integral_prefactor[z1_idx] * cl_integral_prefactor[z2_idx] *
                            d2ClAB_dVddeltab[ell1, zi, zj, z1_idx] *
                            d2ClCD_dVddeltab[ell2, zk, zl, z2_idx] * sigma2[z1_idx, z2_idx] *
                            simpson_weights[z1_idx] * simpson_weights[z2_idx]
                        end
                    end
                end
            end
        end
    end
    return (z_step^2) .* result
end

function SSC_integral_4D_simps_sofia(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array::Array,  R_array::Array)
    """ This version uses the z-R grid instead of z1-z2.
    """ 

    simpson_weights_z = get_simpson_weights(length(z_array))
    simpson_weights_R = get_simpson_weights(length(R_array))
    z_step = (last(z_array)-first(z_array)) / (length(z_array)-1)
    R_step = (last(R_array)-first(R_array)) / (length(R_array)-1)
    nR = length(R_array)


    zpairs_AB = size(ind_AB, 1)
    zpairs_CD = size(ind_CD, 1)
    num_col = size(ind_AB, 2)

    prefactor_interpolator = AkimaInterpolation(cl_integral_prefactor, z_array, extrapolate=true)
    cl_integral_prefactor_R = zeros(length(z_array), length(R_array))

    for (ridx, r) in enumerate(R_array)
        cl_integral_prefactor_R[:, ridx] = prefactor_interpolator.(z_array*r)
    end

    d2ClAB_dVddeltab_R = zeros(32,5,5,length(z_array), length(R_array))
    d2ClCD_dVddeltab_R = zeros(32,5,5,length(z_array), length(R_array))

    for l in 1:32
        for a in 1:5
            for b in 1:5
                for (ridx, r) in enumerate(R_array)
                    interp_AB = AkimaInterpolation(d2ClAB_dVddeltab[l,a,b,:], z_array, extrapolate=true)
                    d2ClAB_dVddeltab_R[l,a,b,:,ridx] = interp_AB.(z_array*r)

                    interp_CD = AkimaInterpolation(d2ClCD_dVddeltab[l,a,b,:], z_array, extrapolate=true)
                    d2ClCD_dVddeltab_R[l,a,b,:,ridx] = interp_CD.(z_array*r)
                end
            end
        end
    end

    result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

    @tturbo for ell1 in 1:nbl
        for ell2 in 1:nbl  # this could be further optimized by computing only upper triangular ells (for LLLL, GLGL, GGGG only), but not with tturbo
            for zij in 1:zpairs_AB
                for zkl in 1:zpairs_CD
                    for z_idx in 1:z_steps
                        for R_idx in 1:nR

                            zi, zj, zk, zl = ind_AB[zij, num_col - 1], ind_AB[zij, num_col], ind_CD[zkl, num_col - 1], ind_CD[zkl, num_col]

                            result[ell1, ell2, zij, zkl] += z_array[z_idx]*cl_integral_prefactor[z_idx] * cl_integral_prefactor_R[z_idx, R_idx] *
                            ( d2ClAB_dVddeltab[ell1, zi, zj, z_idx] * d2ClCD_dVddeltab_R[ell2, zk, zl, z_idx, R_idx]+
                              d2ClCD_dVddeltab[ell2, zk, zl, z_idx] * d2ClAB_dVddeltab_R[ell1, zi, zj, z_idx, R_idx]) * sigma2[z_idx, R_idx] *
                            simpson_weights_z[z_idx] * simpson_weights_R[R_idx]

                        end
                    end
                end
            end
        end
    end
    return (z_step*R_step) .* result
end


# function SSC_integral_4D_opmpson_(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array::Array)
#     """ this version tries to use the KE approximation, to check its impact on the results.
#     """
#     # TODO test this function!
      # ! Fabien says this is most likely wrong, find correct mapping between these approximations

#     simpson_weights = get_simpson_weights(length(z_array))
#     z_step = (last(z_array)-first(z_array)) /(length(z_array)-1)

#     npzwrite("$(output_path)/simpson_weights.npy", simpson_weights)

#     zpairs_AB = size(ind_AB, 1)
#     zpairs_CD = size(ind_CD, 1)
#     num_col = size(ind_AB, 2)

#     result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

#     @tturbo for ell1 in 1:nbl
#         for ell2 in 1:nbl  # this could be further optimized by computing only upper triangular ells (for LLLL, GLGL, GGGG only), but not with tturbo
#             for zij in 1:zpairs_AB
#                 for zkl in 1:zpairs_CD
#                     for zstep_idx in 1:z_steps

#                             zi, zj, zk, zl = ind_AB[zij, num_col - 2], ind_AB[zij, num_col - 1], ind_CD[zkl, num_col - 2], ind_CD[zkl, num_col - 1]

#                             result[ell1, ell2, zij, zkl] += cl_integral_prefactor[zstep_idx]
#                             d2ClAB_dVddeltab[ell1, zi, zj, zstep_idx] *
#                             d2ClCD_dVddeltab[ell2, zk, zl, zstep_idx] * 
#                             sigma2[zstep_idx] *
#                             simpson_weights[zstep_idx]

#                         end
#                     end
#                 end
#             end
#         end
#     return z_step .* result
# end

folder_name = ARGS[1]
integration_type = ARGS[2]

# import arrays:
# the ones actually used in the integration
d2CLL_dVddeltab = npzread("./$(folder_name)/d2CLL_dVddeltab.npy")
d2CGL_dVddeltab = npzread("./$(folder_name)/d2CGL_dVddeltab.npy")
d2CGG_dVddeltab = npzread("./$(folder_name)/d2CGG_dVddeltab.npy")
sigma2          = npzread("./$(folder_name)/sigma2.npy")
z_grid = npzread("./$(folder_name)/z_grid.npy") 
cl_integral_prefactor = npzread("./$(folder_name)/cl_integral_prefactor.npy")
ind_auto = npzread("./$(folder_name)/ind_auto.npy")
ind_cross = npzread("./$(folder_name)/ind_cross.npy")
nbl = size(d2CLL_dVddeltab, 1)
zbins = size(d2CLL_dVddeltab, 2)

# ind file (triu, row-major), for the optimized version
num_col = size(ind_auto, 2)

# check that the z_grid are the same
dz = z_grid[2]-z_grid[1]
z_steps = length(z_grid)

# julia is 1-based, python is 0-based
ind_auto = ind_auto .+ 1
ind_cross = ind_cross .+ 1

# this is for the 3x2pt covariance
probe_combinations = (("L", "L"), ("G", "L"), ("G", "G"))

println("\n*** Computing SSC integral with Julia ****")
println("nbl: ", nbl)
println("zbins: ", zbins)
println("z_steps: ", z_steps)
println("probe_combinations: ", probe_combinations)
println("integration_type: ", integration_type)
println("*****************")

# some sanity checks
@assert length(z_grid) == z_steps
@assert size(d2CLL_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(d2CGL_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(d2CGG_dVddeltab) == (nbl, zbins, zbins, z_steps)
if integration_type == "sofia"
    r_grid = npzread("./$(folder_name)/r_grid.npy") 
    r_steps = length(r_grid)
    @assert length(r_grid) == r_steps
    @assert size(sigma2) == (z_steps, r_steps)
else 
    @assert size(sigma2) == (z_steps, z_steps)
end


@assert size(cl_integral_prefactor) == (z_steps,)
@assert size(ind_auto) == (zbins*(zbins+1)/2, num_col)
@assert size(ind_cross) == (zbins^2, num_col)

d2Cl_dVddeltab_dict = Dict(("L", "L") => d2CLL_dVddeltab,
                            ("G", "L") => d2CGL_dVddeltab,
                            ("G", "G") => d2CGG_dVddeltab)

ind_dict = Dict(("L", "L") => ind_auto,
                ("G", "L") => ind_cross,
                ("G", "G") => ind_auto)

if integration_type == "trapz"
    ssc_integral_4d_func = SSC_integral_4D_trapz
elseif integration_type == "simps"
    ssc_integral_4d_func = SSC_integral_4D_simps
elseif integration_type == "trapz-6D"
    ssc_integral_4d_func = SSC_integral_6D_trapz
elseif integration_type == "sofia"
    ssc_integral_4d_func = SSC_integral_4D_simps_sofia
else
    error("Integration type not recognized")
end


cov_ssc_dict_8d = Dict{Tuple{String, String, String, String}, Array{Float64, 4}}()
if integration_type == "trapz-6D"
    cov_ssc_dict_8d = Dict{Tuple{String, String, String, String}, Array{Float64, 6}}()
end

for row in 1:length(probe_combinations)
    for col in 1:length(probe_combinations)

        probe_A, probe_B = probe_combinations[row]
        probe_C, probe_D = probe_combinations[col]

        if col >= row  # upper triangle and diagonal
            println("Computing cov_SSC_$(probe_A)$(probe_B)_$(probe_C)$(probe_D), zbins = $(zbins)")

            if integration_type == "sofia"
                cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)] =
                @time ssc_integral_4d_func(
                    d2Cl_dVddeltab_dict[probe_A, probe_B],
                    d2Cl_dVddeltab_dict[probe_C, probe_D],
                    ind_dict[probe_A, probe_B],
                    ind_dict[probe_C, probe_D],
                    nbl, z_steps, cl_integral_prefactor, 
                    sigma2, z_grid, r_grid)

                # save
                npzwrite("./$(folder_name)/cov_SSC_spaceborne_$(probe_A)$(probe_B)$(probe_C)$(probe_D)_4D.npy", cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)])
            else 
                cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)] =
                @time ssc_integral_4d_func(
                    d2Cl_dVddeltab_dict[probe_A, probe_B],
                    d2Cl_dVddeltab_dict[probe_C, probe_D],
                    ind_dict[probe_A, probe_B],
                    ind_dict[probe_C, probe_D],
                    nbl, z_steps, cl_integral_prefactor, 
                    sigma2, z_grid)

                # save
                npzwrite("./$(folder_name)/cov_SSC_spaceborne_$(probe_A)$(probe_B)$(probe_C)$(probe_D)_4D.npy", cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)])
            end


            # free memory
            delete!(cov_ssc_dict_8d, (probe_A, probe_B, probe_C, probe_D))

        end

    end  # loop over probes
end  # loop over probes

