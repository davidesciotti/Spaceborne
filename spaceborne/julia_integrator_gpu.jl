using Pkg

# List of packages to check and potentially install
required_packages = ["NPZ", "LoopVectorization", "YAML"]
for pkg_name in required_packages
	try
		eval(Meta.parse("using $pkg_name"))
		println("Package '$pkg_name' is loaded.")
	catch e
		println("Error loading package '$pkg_name': $e")
		println("Attempting to install '$pkg_name'...")
		try
			Pkg.add(pkg_name)
			println("Package '$pkg_name' installed successfully. Retrying load...")
			eval(Meta.parse("using $pkg_name"))
			println("Package '$pkg_name' loaded after installation.")
		catch install_e
			println("Failed to install and load package '$pkg_name': $install_e")
		end
	end
end


function get_simpson_weights(n::Int)
	number_intervals = floor((n - 1) / 2)
	weight_array = zeros(n)
	if n == number_intervals * 2 + 1
		for i in 1:number_intervals
			weight_array[Int((i - 1) * 2 + 1)] += 1 / 3
			weight_array[Int((i - 1) * 2 + 2)] += 4 / 3
			weight_array[Int((i - 1) * 2 + 3)] += 1 / 3
		end
	else
		weight_array[1] += 0.5
		weight_array[2] += 0.5
		for i in 1:number_intervals
			weight_array[Int((i - 1) * 2 + 1)+1] += 1 / 3
			weight_array[Int((i - 1) * 2 + 2)+1] += 4 / 3
			weight_array[Int((i - 1) * 2 + 3)+1] += 1 / 3
		end
		weight_array[length(weight_array)]   += 0.5
		weight_array[length(weight_array)-1] += 0.5
		for i in 1:number_intervals
			weight_array[Int((i - 1) * 2 + 1)] += 1 / 3
			weight_array[Int((i - 1) * 2 + 2)] += 4 / 3
			weight_array[Int((i - 1) * 2 + 3)] += 1 / 3
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

	dz = z_array[2] - z_array[1]

	result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

	@tturbo for ell1 in 1:nbl
		for ell2 in 1:nbl
			for zij in 1:zpairs_AB
				for zkl in 1:zpairs_CD
					for z1_idx in 1:z_steps
						for z2_idx in 1:z_steps

							zi, zj, zk, zl = ind_AB[zij, num_col-1], ind_AB[zij, num_col], ind_CD[zkl, num_col-1], ind_CD[zkl, num_col]

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
	z_step = (last(z_array) - first(z_array)) / (length(z_array) - 1)

	zpairs_AB = size(ind_AB, 1)
	zpairs_CD = size(ind_CD, 1)
	num_col = size(ind_AB, 2)

	result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

	@tturbo for ell1 in 1:nbl
		for ell2 in 1:nbl
			for zij in 1:zpairs_AB
				for zkl in 1:zpairs_CD
					for z1_idx in 1:z_steps
						for z2_idx in 1:z_steps

							zi, zj, zk, zl = ind_AB[zij, num_col-1], ind_AB[zij, num_col], ind_CD[zkl, num_col-1], ind_CD[zkl, num_col]

							result[ell1, ell2, zij, zkl] +=
								cl_integral_prefactor[z1_idx] * cl_integral_prefactor[z2_idx] *
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
	"""

	simpson_weights = get_simpson_weights(length(z_array))
	z_step = (last(z_array) - first(z_array)) / (length(z_array) - 1)

	zpairs_AB = size(ind_AB, 1)
	zpairs_CD = size(ind_CD, 1)
	num_col = size(ind_AB, 2)

	result = zeros(nbl, nbl, zpairs_AB, zpairs_CD)

	@tturbo for ell1 in 1:nbl
		for ell2 in 1:nbl
			for zij in 1:zpairs_AB
				for zkl in 1:zpairs_CD
					for z_idx in 1:z_steps

						zi, zj, zk, zl = ind_AB[zij, num_col-1], ind_AB[zij, num_col], ind_CD[zkl, num_col-1], ind_CD[zkl, num_col]

						result[ell1, ell2, zij, zkl] += cl_integral_prefactor[z_idx] *
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


# Unified function that works with both CPU and GPU arrays
function SSC_integral_4D_simps_unified(
    d2ClAB_dVddeltab, d2ClCD_dVddeltab,
    ind_AB, ind_CD, nbl, z_steps,
    cl_integral_prefactor, sigma2, z_array
)
    """
    Unified version that automatically detects CPU/GPU based on input array types.
    Expects d2Cl arrays to be pre-shaped to 3D: (nbl, zpairs, z_steps)
    """
    
    # Detect device from input arrays
    is_gpu = isa(d2ClAB_dVddeltab, CuArray)
    device_name = is_gpu ? "CUDA" : "CPU"
    println("Using device: $device_name with Tullio")
    
    # Calculate integration parameters
    simpson_weights = get_simpson_weights(length(z_array))
    z_step = (z_array[end] - z_array[1]) / (length(z_array) - 1)
    
    # Get the element type from input arrays
    T = eltype(d2ClAB_dVddeltab)
    
    # Convert helper arrays to the same type and device as the main arrays
    if is_gpu
        cl_integral_prefactor_d = CuArray{T}(cl_integral_prefactor)
        simpson_weights_d = CuArray{T}(simpson_weights)
        sigma2_d = CuArray{T}(sigma2)
    else
        cl_integral_prefactor_d = convert(Array{T}, cl_integral_prefactor)
        simpson_weights_d = convert(Array{T}, simpson_weights)
        sigma2_d = convert(Array{T}, sigma2)
    end
    
    # Get dimensions
    zpairs_AB = size(d2ClAB_dVddeltab, 2)
    zpairs_CD = size(d2ClCD_dVddeltab, 2)
    
    # Allocate result array on the same device
    if is_gpu
        result = CUDA.zeros(T, nbl, nbl, zpairs_AB, zpairs_CD)
    else
        result = zeros(T, nbl, nbl, zpairs_AB, zpairs_CD)
    end
    
    # Perform the computation using Tullio
    # This works on both CPU and GPU arrays
    @tullio result[ell1, ell2, zij, zkl] = 
        cl_integral_prefactor_d[z1_idx] *
        cl_integral_prefactor_d[z2_idx] *
        d2ClAB_dVddeltab[ell1, zij, z1_idx] *
        d2ClCD_dVddeltab[ell2, zkl, z2_idx] *
        sigma2_d[z1_idx, z2_idx] *
        simpson_weights_d[z1_idx] *
        simpson_weights_d[z2_idx]
    
    # Convert result back to CPU if it was computed on GPU
    if is_gpu
        result = Array(result)
    end
    
    return (z_step^2) .* result
end


# Alternative: Pure CUDA kernel implementation (no Tullio dependency)
function SSC_integral_4D_simps_cuda_kernel(
    d2ClAB_dVddeltab, d2ClCD_dVddeltab,
    ind_AB, ind_CD, nbl, z_steps,
    cl_integral_prefactor, sigma2, z_array
)
    """
    Pure CUDA kernel implementation without Tullio.
    Falls back to CPU implementation if CUDA is not available.
    """
    
    if !CUDA.functional() || !isa(d2ClAB_dVddeltab, CuArray)
        # Fall back to CPU implementation
        return SSC_integral_4D_simps_cpu(
            d2ClAB_dVddeltab, d2ClCD_dVddeltab,
            ind_AB, ind_CD, nbl, z_steps,
            cl_integral_prefactor, sigma2, z_array
        )
    end
    
    println("Using device: CUDA with custom kernel")
    
    simpson_weights = get_simpson_weights(length(z_array))
    z_step = (z_array[end] - z_array[1]) / (length(z_array) - 1)
    
    T = eltype(d2ClAB_dVddeltab)
    
    # Move arrays to GPU
    cl_integral_prefactor_d = CuArray{T}(cl_integral_prefactor)
    simpson_weights_d = CuArray{T}(simpson_weights)
    sigma2_d = CuArray{T}(sigma2)
    
    zpairs_AB = size(d2ClAB_dVddeltab, 2)
    zpairs_CD = size(d2ClCD_dVddeltab, 2)
    
    result = CUDA.zeros(T, nbl, nbl, zpairs_AB, zpairs_CD)
    
    # Launch kernel
    threads = (16, 16)  # Adjust based on your GPU
    blocks = (cld(nbl, threads[1]), cld(nbl, threads[2]))
    
    @cuda threads=threads blocks=blocks ssc_kernel_3d!(
        result, d2ClAB_dVddeltab, d2ClCD_dVddeltab,
        cl_integral_prefactor_d, sigma2_d, simpson_weights_d,
        nbl, zpairs_AB, zpairs_CD, z_steps
    )
    
    return (z_step^2) .* Array(result)
end

function ssc_kernel_3d!(
    result, d2ClAB, d2ClCD,
    prefactor, sigma2, weights,
    nbl, zpairs_AB, zpairs_CD, z_steps
)
    ell1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ell2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if ell1 <= nbl && ell2 <= nbl
        for zij in 1:zpairs_AB
            for zkl in 1:zpairs_CD
                acc = zero(eltype(result))
                for z1 in 1:z_steps
                    for z2 in 1:z_steps
                        acc += prefactor[z1] * prefactor[z2] *
                               d2ClAB[ell1, zij, z1] *
                               d2ClCD[ell2, zkl, z2] *
                               sigma2[z1, z2] *
                               weights[z1] * weights[z2]
                    end
                end
                result[ell1, ell2, zij, zkl] = acc
            end
        end
    end
    return
end

# CPU fallback for the kernel version
function SSC_integral_4D_simps_cpu(
    d2ClAB_dVddeltab, d2ClCD_dVddeltab,
    ind_AB, ind_CD, nbl, z_steps,
    cl_integral_prefactor, sigma2, z_array
)
    println("Using device: CPU")
    
    simpson_weights = get_simpson_weights(length(z_array))
    z_step = (z_array[end] - z_array[1]) / (length(z_array) - 1)
    
    T = eltype(d2ClAB_dVddeltab)
    
    zpairs_AB = size(d2ClAB_dVddeltab, 2)
    zpairs_CD = size(d2ClCD_dVddeltab, 2)
    
    result = zeros(T, nbl, nbl, zpairs_AB, zpairs_CD)
    
    # Use LoopVectorization for CPU performance
    @tturbo for ell1 in 1:nbl
        for ell2 in 1:nbl
            for zij in 1:zpairs_AB
                for zkl in 1:zpairs_CD
                    for z1_idx in 1:z_steps
                        for z2_idx in 1:z_steps
                            result[ell1, ell2, zij, zkl] +=
                                cl_integral_prefactor[z1_idx] * cl_integral_prefactor[z2_idx] *
                                d2ClAB_dVddeltab[ell1, zij, z1_idx] *
                                d2ClCD_dVddeltab[ell2, zkl, z2_idx] * 
                                sigma2[z1_idx, z2_idx] *
                                simpson_weights[z1_idx] * simpson_weights[z2_idx]
                        end
                    end
                end
            end
        end
    end
    
    return (z_step^2) .* result
end


# Helper function to move arrays to GPU if needed
function to_gpu_if_available(arr::AbstractArray, force_gpu::Bool=false)
    T = eltype(arr)
    if CUDA.functional() && (force_gpu || T == Float32)
        return CuArray{T}(arr)
    else
        return arr
    end
end


# Main script
folder_name = ARGS[1]
integration_type = ARGS[2]

# Import arrays
d2CLL_dVddeltab = npzread("$(folder_name)/d2CLL_dVddeltab.npy")
d2CGL_dVddeltab = npzread("$(folder_name)/d2CGL_dVddeltab.npy")
d2CGG_dVddeltab = npzread("$(folder_name)/d2CGG_dVddeltab.npy")
sigma2 = npzread("$(folder_name)/sigma2.npy")
z_grid = npzread("$(folder_name)/z_grid.npy")
cl_integral_prefactor = npzread("$(folder_name)/cl_integral_prefactor.npy")
ind_auto = npzread("$(folder_name)/ind_auto.npy")
ind_cross = npzread("$(folder_name)/ind_cross.npy")
unique_probe_names = readlines("$(folder_name)/unique_probe_names.txt")
nbl = size(d2CLL_dVddeltab, 1)
zbins = size(d2CLL_dVddeltab, 2)

# Determine data type
T = eltype(d2CLL_dVddeltab)
println("Using data type: $T")

# Ensure helper arrays match the main data type
sigma2 = convert(Array{T}, sigma2)
cl_integral_prefactor = convert(Array{T}, cl_integral_prefactor)

# Julia is 1-based, Python is 0-based
ind_auto = ind_auto .+ 1
ind_cross = ind_cross .+ 1

num_col = size(ind_auto, 2)
dz = z_grid[2] - z_grid[1]
z_steps = length(z_grid)

println("\n*** Computing SSC ****")
println("specs:")
println("nbl: ", nbl)
println("zbins: ", zbins)
println("z_steps: ", z_steps)
println("probe_combinations: ", unique_probe_names)
println("integration_type: ", integration_type)
println("*****************\n")

# Sanity checks
@assert length(z_grid) == z_steps
@assert size(d2CLL_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(d2CGL_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(d2CGG_dVddeltab) == (nbl, zbins, zbins, z_steps)
@assert size(sigma2) in [(z_steps, z_steps), (z_steps,)]
@assert size(cl_integral_prefactor) == (z_steps,)
@assert size(ind_auto) == (zbins * (zbins + 1) / 2, num_col)
@assert size(ind_cross) == (zbins^2, num_col)

# Contract arrays from 4D to 3D
zpairs_auto = size(ind_auto, 1)
zpairs_cross = size(ind_cross, 1)

d2CLL_dVddeltab_3D = zeros(T, nbl, zpairs_auto, z_steps)
d2CGL_dVddeltab_3D = zeros(T, nbl, zpairs_cross, z_steps)
d2CGG_dVddeltab_3D = zeros(T, nbl, zpairs_auto, z_steps)

for zij in 1:zpairs_auto
    d2CLL_dVddeltab_3D[:, zij, :] .= d2CLL_dVddeltab[:, ind_auto[zij, num_col - 1], ind_auto[zij, num_col], :]
    d2CGG_dVddeltab_3D[:, zij, :] .= d2CGG_dVddeltab[:, ind_auto[zij, num_col - 1], ind_auto[zij, num_col], :]
end
for zij in 1:zpairs_cross
    d2CGL_dVddeltab_3D[:, zij, :] .= d2CGL_dVddeltab[:, ind_cross[zij, num_col - 1], ind_cross[zij, num_col], :]
end

# Decide whether to use GPU based on integration type and data type
use_gpu = (integration_type in ["simps_gpu", "simps_unified", "SSC_integral_4D_simps_unified", "simps_cuda_kernel"]) && 
          CUDA.functional() && (T == Float32 || integration_type == "SSC_integral_4D_simps_unified")

if use_gpu
    println("Moving arrays to GPU...")
    d2CLL_dVddeltab_3D = CuArray{T}(d2CLL_dVddeltab_3D)
    d2CGL_dVddeltab_3D = CuArray{T}(d2CGL_dVddeltab_3D)
    d2CGG_dVddeltab_3D = CuArray{T}(d2CGG_dVddeltab_3D)
end

# Create dictionaries based on whether we're using 3D or 4D arrays
if integration_type in ["SSC_integral_4D_simps_unified", "simps_cuda_kernel"]
    d2Cl_dVddeltab_dict = Dict(
        ("L", "L") => d2CLL_dVddeltab_3D,
        ("G", "L") => d2CGL_dVddeltab_3D,
        ("G", "G") => d2CGG_dVddeltab_3D
    )
else
    d2Cl_dVddeltab_dict = Dict(
        ("L", "L") => d2CLL_dVddeltab,
        ("G", "L") => d2CGL_dVddeltab,
        ("G", "G") => d2CGG_dVddeltab
    )
end

ind_dict = Dict(
    ("L", "L") => ind_auto,
    ("G", "L") => ind_cross,
    ("G", "G") => ind_auto
)

# Select integration function
if integration_type == "trapz"
    ssc_integral_4d_func = SSC_integral_4D_trapz
elseif integration_type == "simps"
    ssc_integral_4d_func = SSC_integral_4D_simps
elseif integration_type == "SSC_integral_4D_simps_unified"
    ssc_integral_4d_func = SSC_integral_4D_simps_unified
elseif integration_type == "simps_cuda_kernel"
    ssc_integral_4d_func = SSC_integral_4D_simps_cuda_kernel
elseif integration_type == "simps_KE_approximation"
    ssc_integral_4d_func = SSC_integral_KE_4D_simps
else
    error("Integration type not recognized: $integration_type")
end

# Initialize result dictionary
cov_ssc_dict_8d = Dict{Tuple{String, String, String, String}, Array{T, 4}}()

for probe in unique_probe_names
    probe_A, probe_B, probe_C, probe_D = split(probe, "")
    
    println("Computing SSC covariance block $(probe_A)$(probe_B)_$(probe_C)$(probe_D)")
    
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
            z_grid
        )
    
    # Save
    npzwrite("$(folder_name)/cov_SSC_spaceborne_$(probe_A)$(probe_B)$(probe_C)$(probe_D)_4D.npy", 
             cov_ssc_dict_8d[(probe_A, probe_B, probe_C, probe_D)])
    
    # Free memory
    delete!(cov_ssc_dict_8d, (probe_A, probe_B, probe_C, probe_D))
end