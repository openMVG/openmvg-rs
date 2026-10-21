#!/usr/bin/env julia
# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

module ALPKernelModel

# Finite-kernel polynomial projection for ALP's spatial localization.
#
# LoG kernels on a 19x19 lattice --> nine shifted sample kernels A
# Subpixel offsets on a 21x21 grid --> target kernels Y and quadratic basis P
# P \ Y --> polynomial coefficient kernels --> A \ (...) --> spatial stencils
#
# The offset fit and kernel projection jointly minimize ||Y - P Q A'||_F^2.
# Six-decimal rounding defines the emitted Q precision. QR solves avoid
# normal-equation conditioning. The square-domain
# reflection/rotation identities are imposed before conversion to f32.
#
# Background: G. Koutaki and K. Uchimura, "Scale-space Processing Using
# Polynomial Representations", CVPR 2014. This implementation derives the
# finite-domain recipe from the analytic kernel formula.
using LinearAlgebra

"""Nominal geometric scale nodes before six-decimal serialization."""
nominal_scales(::Type{T} = Float64) where T = [T(8) / T(5) * sqrt(T(2))^k for k in 0:3]

"""Serialize the four nominal scale nodes to six decimal places."""
function sample_scales()
    return setprecision(BigFloat, 192) do
        Float32.(round.(nominal_scales(BigFloat); digits = 6))
    end
end

"""The scale-normalized analytic 2-D LoG, omitting its common 1/(2π) factor."""
function log_kernel(x, y, sigma)
    radius2 = (x * x + y * y) / (sigma * sigma)
    return (radius2 - 2) * exp(-radius2 / 2) / (sigma * sigma)
end

"""Project translated LoG kernels onto a quadratic and nine integer-shifted kernels.

The two least-squares solves minimize ||Y - P*Q*A'||_F². P contains the quadratic
monomials at a 21×21 grid of subpixel offsets. A contains nine translated kernels
sampled on the 19×19 integer lattice. QR avoids squaring either design condition.
"""
function spatial_coefficients(sigma::T) where T <: AbstractFloat
    pixels = [(T(x), T(y)) for x in -9:9 for y in -9:9]
    sites = [(T(x), T(y)) for x in -1:1 for y in -1:1]
    grid = range(-one(T), one(T); length = 21)
    offsets = [(x, y) for x in grid for y in grid]
    A = [log_kernel(x - u, y - v, sigma) for (x, y) in pixels, (u, v) in sites]
    P = reduce(vcat, (permutedims(T[x*x, y*y, x*y, x, y, 1]) for (x, y) in offsets))
    Y = [log_kernel(x - u, y - v, sigma) for (u, v) in offsets, (x, y) in pixels]
    coefficients = permutedims(qr(A) \ permutedims(qr(P) \ Y))
    return coefficients[1:5, :]
end

"""Make the square-domain reflection/rotation symmetries exact before quantizing."""
function symmetric_spatial_coefficients(sigma::T) where T <: AbstractFloat
    raw = spatial_coefficients(sigma)
    transpose_stencil(row) = vec(permutedims(reshape(row, 3, 3)))
    xx = (raw[1, :] + transpose_stencil(raw[2, :])) / 2
    dx = (raw[4, :] + transpose_stencil(raw[5, :])) / 2
    a = sum(xx[[1, 3, 7, 9]]) / 4
    b = sum(xx[[2, 8]]) / 2
    c = sum(xx[[4, 6]]) / 2
    d = xx[5]
    e = (raw[3, 1] - raw[3, 3] - raw[3, 7] + raw[3, 9]) / 4
    f = (-dx[1] - dx[3] + dx[7] + dx[9]) / 4
    g = (-dx[2] + dx[8]) / 2
    xx = T[a, b, a, c, d, c, a, b, a]
    xy = T[e, 0, -e, 0, 0, 0, -e, 0, e]
    dx = T[-f, -g, -f, 0, 0, 0, f, g, f]
    return vcat(
        permutedims(xx),
        permutedims(transpose_stencil(xx)),
        permutedims(xy),
        permutedims(dx),
        permutedims(transpose_stencil(dx)),
    )
end

# The square-domain symmetries determine exact zeros. These four masks define
# deterministic IEEE-754 zero-sign serialization; they contain no nonzero
# coefficient and do not enter the fit.
const SPATIAL_NEGATIVE_ZERO_BITS = (
    UInt64(0x101c2a80000), UInt64(0x10140080000),
    UInt64(0x82140400000), UInt64(0x82140400000),
)

function spatial_tables()
    return setprecision(BigFloat, 192) do
        rows = Vector{Vector{Float32}}()
        for (index, sigma) in enumerate(nominal_scales(BigFloat))
            coefficients = symmetric_spatial_coefficients(sigma)
            values = vec(permutedims(Float32.(round.(coefficients; digits = 6))))
            for column in eachindex(values)
                if iszero(values[column])
                    negative = !iszero(
                        SPATIAL_NEGATIVE_ZERO_BITS[index] & (UInt64(1) << (column - 1)),
                    )
                    values[column] = negative ? -0.0f0 : 0.0f0
                end
            end
            push!(rows, values)
        end
        return rows
    end
end

end # module ALPKernelModel
