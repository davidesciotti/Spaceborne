using NPZ
using LoopVectorization
using YAML

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

function SSC_integral_4D_simps(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, 
    cl_integral_prefactor, sigma2, z_array::Array, R_array::Array, is_auto)
    """ this version takes advantage of the symmetries between redshift pairs.
    is_auto is not used, but it is kept for consistency with the SSC_integral_4D_simps_reparam function.
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


function SSC_integral_4D_simps_reparam(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, 
    cl_integral_prefactor, sigma2, z_array::Array, is_auto::Bool)
    """ this version takes advantage of the symmetries between redshift pairs AND ell pairs (for the auto-blocks).
    It has been validated against the original implementation, but the performance gain is not significant 
    (it actually seems slower...). 
    """

    simpson_weights = get_simpson_weights(length(z_array))
    z_step = (last(z_array)-first(z_array)) /(length(z_array)-1)


    zpairs_AB = size(ind_AB, 1)
    zpairs_CD = size(ind_CD, 1)
    num_col = size(ind_AB, 2)

    result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

    # Number of unique (ℓ₁, ℓ₂) pairs with ℓ₁ ≤ ℓ₂.
    ell_pairs = div(nbl * (nbl + 1), 2)

    # Precompute mapping arrays: for each k = 1:ell_pairs, store the corresponding (ℓ₁, ℓ₂)
    ell1_array = Vector{Int}(undef, ell_pairs)
    ell2_array = Vector{Int}(undef, ell_pairs)
    k = 1
    for i in 1:nbl
        for j in i:nbl
            ell1_array[k] = i
            ell2_array[k] = j
            k += 1
        end
    end

    if is_auto
        # Loop over the unique ell pairs using a single index k.
        @tturbo for k in 1:ell_pairs
            # Retrieve ℓ₁ and ℓ₂ corresponding to this flattened index.
            ell1 = ell1_array[k]
            ell2 = ell2_array[k]            
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

        # ! ONLY THE AUTO-BLOCKS ARE DIAGONAL IN ELL1, ELL2! 
        # # Post-process: symmetrize the result to fill in the lower triangle.
        for ell1 in 1:nbl
            for ell2 in ell1+1:nbl
                for zij in 1:zpairs_AB
                    for zkl in 1:zpairs_CD
                        result[ell2, ell1, zkl, zij] = result[ell1, ell2, zij, zkl]
                    end
                end
            end
        end
    
    else
        @tturbo for ell1 in 1:nbl
            for ell2 in 1:nbl  
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
    end

    return (z_step^2) .* result
end




function SSC_integral_KE_4D_simps(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, 
    cl_integral_prefactor, sigma2, z_array::Array, is_auto::Bool)
    """ this version takes advantage of the symmetries between redshift pairs, and implements the KE approximation
    (see )
    """

    simpson_weights = get_simpson_weights(length(z_array))
    z_step = (last(z_array)-first(z_array)) /(length(z_array)-1)

    zpairs_AB = size(ind_AB, 1)
    zpairs_CD = size(ind_CD, 1)
    num_col = size(ind_AB, 2)

    result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

    # @tturbo for ell1 in 1:nbl
    for ell1 in 1:nbl
        for ell2 in 1:nbl  # this could be further optimized by computing only upper triangular ells (for LLLL, GLGL, GGGG only), but not with tturbo
            for zij in 1:zpairs_AB
                for zkl in 1:zpairs_CD
                    for z_idx in 1:z_steps  # this is the integration variable

                        zi, zj, zk, zl = ind_AB[zij, num_col - 1], ind_AB[zij, num_col], ind_CD[zkl, num_col - 1], ind_CD[zkl, num_col]

                        result[ell1, ell2, zij, zkl] += cl_integral_prefactor[z_idx]*
                        d2ClAB_dVddeltab[ell1, zi, zj, z_idx] *
                        d2ClCD_dVddeltab[ell2, zk, zl, z_idx] * sigma2[z_idx] *
                        simpson_weights[z_idx]

                    end
                end
            end
        end
    end
    return result .* z_step
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

#TODO: dropping here Blast functions for Clenshaw-Curtis integration

function get_clencurt_weights(min::Number, max::Number, N::Number)
    CC_obj = FastTransforms.chebyshevmoments1(Float64, N)
    w = FastTransforms.clenshawcurtisweights(CC_obj)
    w = (max - min) / 2 * w

    return w
end

function get_clencurt_weights_R_integration(N::Int)

    w = get_clencurt_weights(-1, 1, N)

    index = div(N + 3, 2)
    w = w[index:end]
    w[1]/=2 #TODO: investigate if there are better solutions, this is not the analytic solution.

    return w
end

"""function SSC_integral_4D_simps_zR(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array::Array,  R_array::Array)

    simpson_weights_z = get_simpson_weights(length(z_array))
    z_step = (last(z_array)-first(z_array)) / (length(z_array)-1)
    
    nR = length(R_array)
    w_R = get_clencurt_weights_R_integration(2*nR+1)


    zpairs_AB = size(ind_AB, 1)
    zpairs_CD = size(ind_CD, 1)
    num_col = size(ind_AB, 2)

    prefactor_interpolator = AkimaInterpolation(cl_integral_prefactor, z_array, extrapolation = ExtrapolationType.Extension)
    cl_integral_prefactor_R = zeros(length(z_array), length(R_array))

    for (ridx, r) in enumerate(R_array)
        cl_integral_prefactor_R[:, ridx] = prefactor_interpolator.(z_array*r)
    end

    #TODO: horrible hard coding of the numbers
    #same in the for loop below
    d2ClAB_dVddeltab_R = zeros(29,3,3,length(z_array), length(R_array))
    d2ClCD_dVddeltab_R = zeros(29,3,3,length(z_array), length(R_array))

    for l in 1:29
        for a in 1:3
            for b in 1:3
                for (ridx, r) in enumerate(R_array)
                    interp_AB = AkimaInterpolation(d2ClAB_dVddeltab[l,a,b,:], z_array, extrapolation= ExtrapolationType.Extension)
                    d2ClAB_dVddeltab_R[l,a,b,:,ridx] = interp_AB.(z_array*r)

                    interp_CD = AkimaInterpolation(d2ClCD_dVddeltab[l,a,b,:], z_array, extrapolation= ExtrapolationType.Extension)
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
                            simpson_weights_z[z_idx] * w_R[R_idx]

                        end
                    end
                end
            end
        end
    end
    return z_step .* result
end"""

function SSC_integral_4D_simps_zR(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, cl_integral_prefactor, sigma2, z_array::Array,  R_array::Array)
    """ This version uses the z-R grid instead of z1-z2. The R grid is linear, the abopve commented version uses clenshaw curtis.
    """

    simpson_weights_z = get_simpson_weights(length(z_array))
    simpson_weights_R = get_simpson_weights(length(R_array))
    z_step = (last(z_array)-first(z_array)) / (length(z_array)-1)
    R_step = (last(R_array)-first(R_array)) / (length(R_array)-1)
    nR = length(R_array)


    zpairs_AB = size(ind_AB, 1)
    zpairs_CD = size(ind_CD, 1)
    num_col = size(ind_AB, 2)

    prefactor_interpolator = AkimaInterpolation(cl_integral_prefactor, z_array, extrapolation= ExtrapolationType.Extension)
    cl_integral_prefactor_R = zeros(length(z_array), length(R_array))

    for (ridx, r) in enumerate(R_array)
        cl_integral_prefactor_R[:, ridx] = prefactor_interpolator.(z_array*r)
    end

    #TODO: horrible hard coding of the numbers
    #same in the for loop below
    d2ClAB_dVddeltab_R = zeros(29,3,3,length(z_array), length(R_array))
    d2ClCD_dVddeltab_R = zeros(29,3,3,length(z_array), length(R_array))

    for l in 1:29
        for a in 1:3
            for b in 1:3
                for (ridx, r) in enumerate(R_array)
                    interp_AB = AkimaInterpolation(d2ClAB_dVddeltab[l,a,b,:], z_array, extrapolation= ExtrapolationType.Extension)
                    d2ClAB_dVddeltab_R[l,a,b,:,ridx] = interp_AB.(z_array*r)

                    interp_CD = AkimaInterpolation(d2ClCD_dVddeltab[l,a,b,:], z_array, extrapolation= ExtrapolationType.Extension)
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

folder_name = ARGS[1]
integration_type = ARGS[2]

# import arrays:
# the ones actually used in the integration
d2CLL_dVddeltab = npzread("$(folder_name)/d2CLL_dVddeltab.npy")
d2CGL_dVddeltab = npzread("$(folder_name)/d2CGL_dVddeltab.npy")
d2CGG_dVddeltab = npzread("$(folder_name)/d2CGG_dVddeltab.npy")
sigma2          = npzread("$(folder_name)/sigma2.npy")
z_grid = npzread("$(folder_name)/z_grid.npy") #previously z_integrands
R_grid = npzread("$(folder_name)/R_grid.npy")
cl_integral_prefactor = npzread("$(folder_name)/cl_integral_prefactor.npy")
ind_auto = npzread("$(folder_name)/ind_auto.npy")
ind_cross = npzread("$(folder_name)/ind_cross.npy")
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

println("\n*** Computing SSC integral in Julia ****")
println("specs:")
println("nbl: ", nbl)
println("zbins: ", zbins)
println("z_steps: ", z_steps)
println("probe_combinations: ", probe_combinations)
println("integration_type: ", integration_type)
println("*****************\n")

# some sanity checks
@assert length(z_grid) == z_steps
@assert size(d2CLL_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(d2CGL_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(d2CGG_dVddeltab) == (nbl, zbins, zbins, z_steps)
# @assert size(sigma2) == (z_steps, z_steps)
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
elseif integration_type == "simps_KE_approximation"
    ssc_integral_4d_func = SSC_integral_KE_4D_simps
elseif integration_type == "trapz-6D"
    ssc_integral_4d_func = SSC_integral_6D_trapz
elseif integration_type == "simps_zR"
    ssc_integral_4d_func = SSC_integral_4D_simps_zR
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
            println("Computing SSC covariance block $(probe_A)$(probe_B)_$(probe_C)$(probe_D)"); flush(stdout)

            cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)] =
            @time ssc_integral_4d_func(
                d2Cl_dVddeltab_dict[probe_A, probe_B],
                d2Cl_dVddeltab_dict[probe_C, probe_D],
                ind_dict[probe_A, probe_B],
                ind_dict[probe_C, probe_D],
                nbl, z_steps, cl_integral_prefactor,
                sigma2, z_grid, R_grid)

            # save
            npzwrite("$(folder_name)/cov_SSC_spaceborne_$(probe_A)$(probe_B)$(probe_C)$(probe_D)_4D.npy", cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)])

            # free memory
            delete!(cov_ssc_dict_8d, (probe_A, probe_B, probe_C, probe_D))

        end

    end  # loop over probes
end  # loop over probes

