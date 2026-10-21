# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

module ALPScaleModel

# Gaussian and sinusoidal inputs -> finite Float32 LoG response profiles
#            | unit profile energy, equal family and octave weights
#            v
# centered sigma basis + constant-response constraint -> 12-column QR
#            v
# four response samples -> Float32 cubic coefficients in physical sigma
#
# The objective averages squared response error over the declared quadrature.
# Gaussian phases cover one pixel cell; wave directions cover [0,pi/2].
# No image files or fitted coefficients enter the solve.
# The model convolves vertically then horizontally. Runtime image filtering
# uses horizontal then vertical passes; Float32 rounding can differ by order.
#
# References: G. Koutaki and K. Uchimura, "Scale-space Processing Using
# Polynomial Representations", CVPR 2014 (polynomial scale-space framework).
# G. H. Golub and J. H. Welsch, "Calculation of Gauss Quadrature Rules",
# Mathematics of Computation 23(106), 1969 (Jacobi-matrix quadrature).
# The response objective and implementation are defined by this project.
using LinearAlgebra

const FIT_WINDOW = (Float64(1.9027314f0), Float64(4.1498866f0))
const BASE_SIGMAS = (0.5, 1.6)
const SCALE_QUADRATURE_COUNT = 32
const PHASE_QUADRATURE_COUNT = 3
const DIRECTION_QUADRATURE_COUNT = 8
const PROFILE_QUADRATURE_COUNT = 12

function quadrature(count)
    off_diagonal = [index / sqrt(4index * index - 1) for index in 1:count-1]
    decomposition = eigen(SymTridiagonal(zeros(count), off_diagonal))
    return decomposition.values, decomposition.vectors[1, :].^2
end

function log_quadrature(lower, upper, count)
    nodes, weights = quadrature(count)
    scales = exp.(log(lower) .+ (nodes .+ 1) .* ((log(upper) - log(lower)) / 2))
    return scales, weights
end

function profile_quadrature(sample_sigmas, base_sigma)
    lower, upper = Float64(first(sample_sigmas)), Float64(last(sample_sigmas))
    base = Float64(Float32(base_sigma))
    radius = ceil(Int, 3sqrt(upper^2 - base^2))
    boundaries = sort!(unique(vcat(
        [lower, upper],
        [sqrt(base^2 + (index / 3)^2) for index in 1:radius
            if lower < sqrt(base^2 + (index / 3)^2) < upper],
    )))
    scales, weights = Float64[], Float64[]
    for index in 1:length(boundaries)-1
        nodes, local_weights = log_quadrature(
            boundaries[index], boundaries[index + 1], PROFILE_QUADRATURE_COUNT,
        )
        append!(scales, nodes)
        append!(weights, local_weights .* (log(boundaries[index + 1] / boundaries[index]) / log(upper / lower)))
    end
    return scales, weights
end

function gaussian_half_kernel(target::Float32, base::Float32)
    sigma = max(sqrt(max(target * target - base * base, 0.0f0)), 0.1f0)
    radius = ceil(Int, 3.0f0 * sigma)
    full, total = Float32[], 0.0f0
    for offset in -radius:radius
        distance = Float32(offset) / sigma
        value = exp(-0.5f0 * distance * distance)
        push!(full, value)
        total += value
    end
    return Float32[full[radius + 1 + offset] / total for offset in 0:radius]
end

function gaussian_sample(sigma::Float32, phase_x, phase_y, x, y)
    distance2 = (Float32(x) - Float32(phase_x))^2 + (Float32(y) - Float32(phase_y))^2
    return exp(-distance2 / (2.0f0 * sigma * sigma))
end

function gaussian_patch(physical_sigma, base_sigma, phase_x, phase_y, target_sigma)
    sigma = Float32(sqrt(physical_sigma^2 + base_sigma^2))
    kernel = gaussian_half_kernel(Float32(target_sigma), Float32(base_sigma))
    radius = length(kernel) - 1
    columns = (-2-radius):(2+radius)
    intermediate = Matrix{Float32}(undef, 5, length(columns))
    for (column, x) in enumerate(columns), (row, y) in enumerate(-2:2)
        value = gaussian_sample(sigma, phase_x, phase_y, x, y) * kernel[1]
        for offset in 1:radius
            value += (
                gaussian_sample(sigma, phase_x, phase_y, x, y - offset) +
                gaussian_sample(sigma, phase_x, phase_y, x, y + offset)
            ) * kernel[offset + 1]
        end
        intermediate[row, column] = value
    end
    output = Matrix{Float32}(undef, 5, 5)
    for row in 1:5, column in 1:5
        center = radius + column
        value = intermediate[row, center] * kernel[1]
        for offset in 1:radius
            value += (intermediate[row, center - offset] + intermediate[row, center + offset]) * kernel[offset + 1]
        end
        output[row, column] = value
    end
    return output
end

function convolve(image, kernel)
    height, width = size(image)
    radius = length(kernel) - 1
    intermediate, output = similar(image), similar(image)
    for column in 1:width, row in 1:height
        value = image[row, column] * kernel[1]
        for offset in 1:radius
            value += (image[clamp(row - offset, 1, height), column] + image[clamp(row + offset, 1, height), column]) * kernel[offset + 1]
        end
        intermediate[row, column] = value
    end
    for column in 1:width, row in 1:height
        value = intermediate[row, column] * kernel[1]
        for offset in 1:radius
            value += (intermediate[row, clamp(column - offset, 1, width)] + intermediate[row, clamp(column + offset, 1, width)]) * kernel[offset + 1]
        end
        output[row, column] = value
    end
    return output
end

function laplacian_response(image, sigma)
    row, column = (size(image) .+ 1) .÷ 2
    laplacian = image[row, column - 1] + image[row, column + 1] +
        image[row - 1, column] + image[row + 1, column] - 4.0f0 * image[row, column]
    return Float64(laplacian * Float32(sigma)^2)
end

gaussian_response(sigma, base, phase_x, phase_y, target) =
    laplacian_response(gaussian_patch(sigma, base, phase_x, phase_y, target), target)

function wave_response(sigma, base, direction, target)
    frequency = sqrt(2.0) / sigma
    kernel = gaussian_half_kernel(Float32(target), Float32(base))
    side = 2(length(kernel) - 1) + 5
    center = (side + 1) ÷ 2
    amplitude = exp(-base^2 * frequency^2 / 2)
    image = Float32[
        amplitude * cos(frequency * (cos(direction) * (column - center) + sin(direction) * (row - center)))
        for row in 1:side, column in 1:side
    ]
    return laplacian_response(convolve(image, kernel), target)
end

function add_profile_rows!(design, targets, row, samples, responses, weight, profiles, profile_weights, contrasts, center, half_width)
    differences = contrasts' * samples
    constant = sum(samples) / 4
    for (index, sigma) in enumerate(profiles)
        coordinate = (sigma - center) / half_width
        basis = [coordinate^3, coordinate^2, coordinate, 1.0]
        root_weight = sqrt(weight * profile_weights[index])
        row += 1
        design[row, :] = root_weight * kron(differences, basis)
        targets[row] = root_weight * (responses[index] - constant)
    end
    return row
end

"""Fit the unit-energy Gaussian/wave response objective with exact constant-input constraints."""
function scale_transform(sample_sigmas::Vector{Float32})
    length(sample_sigmas) == 4 || error("scale model requires four sample sigmas")
    all(sigma -> isfinite(sigma) && sigma > 0, sample_sigmas) || error("invalid sample sigma")
    BLAS.set_num_threads(1)
    center = (Float64(first(sample_sigmas)) + Float64(last(sample_sigmas))) / 2
    half_width = (Float64(last(sample_sigmas)) - Float64(first(sample_sigmas))) / 2
    contrasts = Float64[
        1/sqrt(2) 1/sqrt(6) 1/sqrt(12);
        -1/sqrt(2) 1/sqrt(6) 1/sqrt(12);
        0 -2/sqrt(6) 1/sqrt(12);
        0 0 -3/sqrt(12);
    ]
    particular = Float64[0 0 0 0; 0 0 0 0; 0 0 0 0; 0.25 0.25 0.25 0.25]
    sigmas, sigma_weights = log_quadrature(FIT_WINDOW..., SCALE_QUADRATURE_COUNT)
    phases, phase_weights = quadrature(PHASE_QUADRATURE_COUNT)
    phases = phases ./ 2
    directions, direction_weights = quadrature(DIRECTION_QUADRATURE_COUNT)
    directions = (directions .+ 1) .* (pi / 4)
    grids = [profile_quadrature(sample_sigmas, base) for base in BASE_SIGMAS]
    row_count = SCALE_QUADRATURE_COUNT * (PHASE_QUADRATURE_COUNT^2 + DIRECTION_QUADRATURE_COUNT) * sum(length(grid[1]) for grid in grids)
    design = Matrix{Float64}(undef, row_count, 12)
    targets = Vector{Float64}(undef, row_count)
    row = 0
    for (base_index, base) in enumerate(BASE_SIGMAS), (index, sigma) in enumerate(sigmas), (x_index, phase_x) in enumerate(phases), (y_index, phase_y) in enumerate(phases)
        profiles, profile_weights = grids[base_index]
        samples = [gaussian_response(sigma, base, phase_x, phase_y, target) for target in sample_sigmas]
        peak = gaussian_response(sigma, base, phase_x, phase_y, Float32(sigma))
        responses = [gaussian_response(sigma, base, phase_x, phase_y, Float32(target)) for target in profiles]
        # Fix the numerical normalization order: known finite peak, then unit
        # integrated profile energy. Both stages are Float64.
        normalizer = -2peak
        samples = samples / normalizer
        responses = responses / normalizer
        energy = sqrt(sum(profile_weights .* responses.^2))
        samples, responses = samples / energy, responses / energy
        weight = 0.5 * (0.5 * sigma_weights[index] * phase_weights[x_index] * phase_weights[y_index])
        row = add_profile_rows!(design, targets, row, samples, responses, weight, profiles, profile_weights, contrasts, center, half_width)
    end
    for (base_index, base) in enumerate(BASE_SIGMAS), (index, sigma) in enumerate(sigmas), (direction_index, direction) in enumerate(directions)
        profiles, profile_weights = grids[base_index]
        responses = [wave_response(sigma, base, direction, target) for target in profiles]
        samples = [wave_response(sigma, base, direction, target) for target in sample_sigmas]
        energy = sqrt(sum(profile_weights .* responses.^2))
        weight = 0.25 * sigma_weights[index] * direction_weights[direction_index]
        row = add_profile_rows!(design, targets, row, samples / energy, responses / energy, weight, profiles, profile_weights, contrasts, center, half_width)
    end
    row == row_count || error("scale objective row count differs")
    factorization = qr(design)
    solution = factorization \ targets
    scaled = particular + reshape(solution, 4, 3) * contrasts'
    m, h = center, half_width
    conversion = [
        1/h^3 0 0 0;
        -3m/h^3 1/h^2 0 0;
        3m^2/h^3 -2m/h^2 1/h 0;
        -m^3/h^3 m^2/h^2 -m/h 1;
    ]
    transform = Float32.(conversion * scaled)
    all(isfinite, transform) || error("non-finite scale transform")
    return (;
        transform,
        design_condition = cond(Matrix(factorization.R)),
        weighted_residual_norm = sqrt(sum(abs2, design * solution - targets)),
        row_count,
    )
end

"""Apply the constrained cubic using three sample differences and the last sample.

The first three quantized weights determine the effective fourth column through
the constant constraint. The stored fourth entry records the rounded fit;
centered evaluation preserves equal finite samples by its arithmetic.
"""
function apply_transform(transform, samples::Vector{Float32})
    differences = samples[1:3] .- samples[4]
    coefficients = zeros(Float32, 4)
    for row in 1:4
        value = transform[row, 1] * differences[1]
        value += transform[row, 2] * differences[2]
        value += transform[row, 3] * differences[3]
        coefficients[row] = row == 4 ? samples[4] + value : value
    end
    return coefficients
end

end # module ALPScaleModel

"""Derive the four-plane scale table from the declared Gaussian/wave objective."""
function derive_scale_table()
    sample_sigmas = Float32.(round.(1.6 .* exp2.((0:3) ./ 2); digits = 6))
    return ALPScaleModel.scale_transform(sample_sigmas).transform
end
