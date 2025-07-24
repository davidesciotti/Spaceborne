using Pkg

# List of packages to check and potentially install
required_packages = ["NPZ", "LoopVectorization", "YAML", "Tullio", "CUDA"]

for pkg_name in required_packages
    try
        # attempts to load the package; convert the string to a symbol using Meta
        eval(Meta.parse("using $pkg_name"))
        println("Package '$pkg_name' is loaded.")
    catch e
        # If 'using' fails, it means the package isn't available or there's an error.
        println("Error loading package '$pkg_name': $e")
        println("Attempting to install '$pkg_name'...")
        try
            Pkg.add(pkg_name)
            println("Package '$pkg_name' installed successfully. Retrying load...")
            # After successful installation, try loading again
            eval(Meta.parse("using $pkg_name"))
            println("Package '$pkg_name' loaded after installation.")
        catch install_e
            println("Failed to install and load package '$pkg_name': $install_e")
        end
    end
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


function SSC_integral_4D_trapz(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, 
    z_steps, cl_integral_prefactor, sigma2, z_array::Array)
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
    cl_integral_prefactor, sigma2, z_array::Array)
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


function SSC_integral_KE_4D_simps(d2ClAB_dVddeltab, d2ClCD_dVddeltab, ind_AB, ind_CD, nbl, z_steps, 
    cl_integral_prefactor, sigma2, z_array::Array)
    """ this version takes advantage of the symmetries between redshift pairs, and implements the KE approximation
    (see )
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


# ! GPU funcs
function to_device(arr::Array, device_type::Symbol)
    if device_type == :cpu
        return arr
    elseif device_type == :cuda
        # Check if CUDA is available and loaded
        if isdefined(Main, :CUDA) && CUDA.functional()
            return CUDA.cu(arr)
        else
            @warn "CUDA.jl is not functional or not loaded. Falling back to CPU for CUDA device type."
            return arr
        end
    elseif device_type == :metal
        # Check if Metal is available and loaded
        if isdefined(Main, :Metal) && Metal.functional()
            return Metal.MtlArray(arr)
        else
            @warn "Metal.jl is not functional or not loaded. Falling back to CPU for Metal device type."
            return arr
        end
    # Add AMDGPU/ROCm if needed
    # elseif device_type == :rocm
    #     if isdefined(Main, :AMDGPU) && AMDGPU.functional()
    #         return AMDGPU.roc(arr)
    #     else
    #         @warn "AMDGPU.jl is not functional or not loaded. Falling back to CPU for ROCm device type."
    #         return arr
    #     end
    else
        error("Unsupported device type: $device_type. Choose :cpu, :cuda, or :metal.")
    end
end


function SSC_integral_4D_simps_GPU(
    d2ClAB_dVddeltab, d2ClCD_dVddeltab,
    ind_AB, ind_CD, nbl, z_steps,
    cl_integral_prefactor, sigma2, z_array::AbstractVector
)
    device = :cuda
    println("Using device: $device")

    z_step = (z_array[end] - z_array[begin]) / (length(z_array) - 1)

    # Move everything to the GPU
    d2ClAB_dVddeltab = cu(d2ClAB_dVddeltab)
    d2ClCD_dVddeltab = cu(d2ClCD_dVddeltab)
    ind_AB = cu(ind_AB)
    ind_CD = cu(ind_CD)
    cl_integral_prefactor = cu(cl_integral_prefactor)
    sigma2 = cu(sigma2)
    simpson_weights = cu(get_simpson_weights(length(z_array)))

    zpairs_AB = size(ind_AB, 1)
    zpairs_CD = size(ind_CD, 1)
    num_col = size(ind_AB, 2)

    result = CUDA.zeros(Float64, nbl, nbl, zpairs_AB, zpairs_CD)

    threads_per_block_x = min(32, zpairs_AB) # Choose appropriate values, often powers of 2 up to 1024 total
    threads_per_block_y = min(32, zpairs_CD) # Check CUDA device limits
    threads_per_block = (threads_per_block_x, threads_per_block_y)

    # Number of blocks needed
    blocks_per_grid_x = cld(nbl, 1) # If processing one ell per block idx, need nbl blocks
    blocks_per_grid_y = cld(nbl, 1) # If processing one ell per block idx, need nbl blocks
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    @cuda threads=threads_per_block blocks=blocks_per_grid SSC_kernel!(
        result, d2ClAB_dVddeltab, d2ClCD_dVddeltab,
        ind_AB, ind_CD, cl_integral_prefactor, sigma2,
        simpson_weights, nbl, zpairs_AB, zpairs_CD, z_steps, num_col
    )

    return (z_step^2.0) .* result
end

function SSC_kernel!(
    result, d2ClAB_dVddeltab, d2ClCD_dVddeltab,
    ind_AB, ind_CD, cl_integral_prefactor, sigma2,
    simpson_weights, nbl, zpairs_AB, zpairs_CD, z_steps, num_col
)
    # Correctly map block and thread indices to problem dimensions
    # Based on launch config: blocks=(nbl, nbl), threads=(min(32, zpairs_AB), min(32, zpairs_CD))
    ell1 = blockIdx().x  # One block per ell1 (assuming nbl <= gridDim().x)
    ell2 = blockIdx().y  # One block per ell2 (assuming nbl <= gridDim().y)
    zij  = threadIdx().x # One thread.x per zij (within limits of blockDim().x)
    zkl  = threadIdx().y # One thread.y per zkl (within limits of blockDim().y)

    # Check bounds based on the actual dimensions
    # Important: Ensure the thread indices don't exceed the zpair dimensions
    # This check is crucial because threads_per_block might be smaller than zpairs_*
    if ell1 <= nbl && ell2 <= nbl && zij <= zpairs_AB && zkl <= zpairs_CD
        # Bounds check for thread indices within the block's assigned work
        # Only proceed if this thread's (zij, zkl) is within the valid range for this block
        # Since threads=(min(32, zpairs_AB), min(32, zpairs_CD)), this check is mostly
        # redundant *if* zpairs_AB/CD <= 32. But it's good practice.
        # If zpairs_AB or zpairs_CD could be > 32, you'd need a more complex launch
        # strategy (e.g., multiple blocks in the z dimensions or loops in the kernel).

        zi = ind_AB[zij, num_col - 1]
        zj = ind_AB[zij, num_col]
        zk = ind_CD[zkl, num_col - 1]
        zl = ind_CD[zkl, num_col]

        # Use Float64 for accumulation if inputs are Float64
        acc = 0.0 # Changed from 0.0f0 to 0.0 for Float64 consistency

        for z1 in 1:z_steps
            for z2 in 1:z_steps
                # Make sure all array accesses are within bounds
                # Assuming d2Cl arrays are (nbl, zbins, zbins, z_steps)
                # and indices zi,zj,zk,zl are within [1, zbins]
                # and sigma2, cl_integral_prefactor, simpson_weights are within [1, z_steps]
                acc += cl_integral_prefactor[z1] * cl_integral_prefactor[z2] *
                       d2ClAB_dVddeltab[ell1, zi, zj, z1] *
                       d2ClCD_dVddeltab[ell2, zk, zl, z2] *
                       sigma2[z1, z2] *
                       simpson_weights[z1] * simpson_weights[z2]
            end
        end
        result[ell1, ell2, zij, zkl] = acc
    end
    return
end



function get_default_device()
    if isdefined(Main, :CUDA) && CUDA.functional()
        return :cuda
    elseif isdefined(Main, :Metal) && Metal.functional()
        return :metal
    else
        @warn "No GPU backend available. Using CPU."
        return :cpu
    end
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
cl_integral_prefactor = npzread("$(folder_name)/cl_integral_prefactor.npy")
ind_auto = npzread("$(folder_name)/ind_auto.npy")
ind_cross = npzread("$(folder_name)/ind_cross.npy")
unique_probe_names = readlines("$(folder_name)/unique_probe_names.txt")
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


println("\n*** Computing SSC ****")
println("specs:")
println("nbl: ", nbl)
println("zbins: ", zbins)
println("z_steps: ", z_steps)
println("probe_combinations: ", unique_probe_names)
println("integration_type: ", integration_type)
println("*****************\n")

# some sanity checks
@assert length(z_grid) == z_steps
@assert size(d2CLL_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(d2CGL_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(d2CGG_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(sigma2) in [(z_steps, z_steps), (z_steps,)]  # for LR or KE approx
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
elseif integration_type == "simps_gpu"
    ssc_integral_4d_func = SSC_integral_4D_simps_GPU
elseif integration_type == "simps_KE_approximation"
    ssc_integral_4d_func = SSC_integral_KE_4D_simps
elseif integration_type == "trapz-6D"
    ssc_integral_4d_func = SSC_integral_6D_trapz
else
    error("Integration type not recognized")
end

# initialize array
cov_ssc_dict_8d = Dict{Tuple{String, String, String, String}, Array{Float64, 4}}()
if integration_type == "trapz-6D"
    cov_ssc_dict_8d = Dict{Tuple{String, String, String, String}, Array{Float64, 6}}()
end

for probe in unique_probe_names

    probe_A, probe_B, probe_C, probe_D = split(probe, "")
    
    println("Computing SSC covariance block $(probe_A)$(probe_B)_$(probe_C)$(probe_D)")

    # apparently, Julia doesn't like keyword arguments so much
    cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)] =
    @time ssc_integral_4d_func(
        d2Cl_dVddeltab_dict[(probe_A, probe_B)],
        d2Cl_dVddeltab_dict[(probe_C, probe_D)],
        ind_dict[(probe_A, probe_B)],
        ind_dict[(probe_C, probe_D)],
        nbl, 
        z_steps, 
        cl_integral_prefactor, 
        sigma2, 
        z_grid)

    # save
    npzwrite("$(folder_name)/cov_SSC_spaceborne_$(probe_A)$(probe_B)$(probe_C)$(probe_D)_4D.npy", cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)])

    # free memory
    delete!(cov_ssc_dict_8d, (probe_A, probe_B, probe_C, probe_D))

end  # loop over probes

