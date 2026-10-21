#!/usr/bin/env julia
# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

using Printf
using LinearAlgebra
using SHA
include("alp_geometry_model.jl")

const GEOMETRY_GENERATOR_VERSION = "9"
const GEOMETRY_MATRIX_SHA256 = "76c69928a955781a70055cf94a2cf86214bd9a7d61da9b2ccf58abdb6fe3dc2c"

rust_float(value::Float32) = string(value)
rust_bits(value::Float32) = @sprintf("0x%08x", reinterpret(UInt32, value))

function matrix_digest(transform)
    bytes = UInt8[]
    for row in eachrow(transform), value in row
        bits = reinterpret(UInt32, value)
        for shift in (0, 8, 16, 24)
            push!(bytes, UInt8((bits >> shift) & 0xff))
        end
    end
    return bytes2hex(sha256(bytes))
end

function polynomial_samples(transform, anchor; step = 0.02, curvature = 1.0, offset = -0.5)
    stationary = Float64(log2(anchor / ALPGeometryModel.RUNTIME_ORIGIN)) + step
    coefficients = [0.0, curvature / 2, -curvature * stationary,
        offset + curvature * stationary^2 / 2]
    return Float32.(ALPGeometryModel.effective_quantized_matrix(transform) \ coefficients)
end

function reference_cases(result)
    transform=result.transform
    compatibility_lower=result.compatibility_lower
    anchor = 2.4f0
    minimum = polynomial_samples(transform, anchor)
    positive = polynomial_samples(transform, anchor; step = 0.125)
    negative = polynomial_samples(transform, anchor; step = -0.125)
    sigma_min, sigma_max = ALPGeometryModel.PUBLIC_SIGMA_DOMAIN
    specifications = [
        (name = "interior_minimum", samples = minimum, anchor = anchor, maximum = false, applied = true),
        (name = "interior_maximum", samples = -minimum, anchor = anchor, maximum = true, applied = true),
        (name = "positive_step_clamp", samples = positive, anchor = anchor, maximum = false, applied = true),
        (name = "negative_step_clamp", samples = negative, anchor = anchor, maximum = false, applied = true),
        (name = "lower_boundary_inward", samples = polynomial_samples(transform, sigma_min; step = 0.125), anchor = sigma_min, maximum = false, applied = true),
        (name = "upper_boundary_inward", samples = polynomial_samples(transform, sigma_max; step = -0.125), anchor = sigma_max, maximum = false, applied = true),
        (name = "constant_response_fallback", samples = ones(Float32, 4), anchor = anchor, maximum = false, applied = false),
        (name = "polarity_fallback", samples = minimum, anchor = anchor, maximum = true, applied = false),
        (name = "lower_domain_fallback", samples = polynomial_samples(transform, sigma_min; step = -0.125), anchor = sigma_min, maximum = false, applied = false),
        (name = "upper_domain_fallback", samples = polynomial_samples(transform, sigma_max; step = 0.125), anchor = sigma_max, maximum = false, applied = false),
        (name = "nonfinite_sample_fallback", samples = Float32[NaN, minimum[2], minimum[3], minimum[4]], anchor = anchor, maximum = false, applied = false),
        (name = "curvature_epsilon_fallback", samples = polynomial_samples(transform, anchor; curvature = Float64(ALPGeometryModel.CURVATURE_EPSILON) / 4, offset = 0.0), anchor = anchor, maximum = false, applied = false),
        (name = "zero_anchor_fallback", samples = minimum, anchor = 0.0f0, maximum = false, applied = false),
        (name = "nonfinite_anchor_fallback", samples = minimum, anchor = Inf32, maximum = false, applied = false),
    ]
    gaussian=result.cases[95]
    gaussian_samples=Float32.(gaussian.samples)
    gaussian_anchor=Float32(gaussian.sigma)
    append!(specifications,[
        (name="compatible_gaussian_minimum",samples=gaussian_samples,anchor=gaussian_anchor,maximum=false,applied=true),
        (name="compatible_gaussian_maximum",samples=-gaussian_samples,anchor=gaussian_anchor,maximum=true,applied=true),
    ])
    displaced_anchor=gaussian_anchor*exp2(1f0/16f0)
    append!(specifications,[
        (name="compatible_gaussian_stationary_minimum",samples=gaussian_samples,anchor=displaced_anchor,maximum=false,applied=true),
        (name="compatible_gaussian_stationary_maximum",samples=-gaussian_samples,anchor=displaced_anchor,maximum=true,applied=true),
    ])
    margin=only(filter(profile->profile.name=="margin_profile",profile_cases(result)))
    margin_beta=ALPGeometryModel.compatibility_beta(margin.samples,compatibility_lower)
    ALPGeometryModel.BETA_BASE<margin_beta<ALPGeometryModel.BETA_BOOST || error("reference profile is outside blend margin")
    push!(specifications,(name="compatibility_margin_maximum",samples=margin.samples,anchor=2.42f0,maximum=true,applied=true))
    push!(specifications,(name="compatibility_margin_stationary_maximum",samples=margin.samples,anchor=2.4f0*exp2(1f0/16f0),maximum=true,applied=true))
    for (samples, direction) in [(positive, 1), (negative, -1)]
        a, b, c, _ = ALPGeometryModel.apply_cubic(transform, samples)
        t = log2(anchor / ALPGeometryModel.RUNTIME_ORIGIN)
        derivative = (3.0f0 * a * t + 2.0f0 * b) * t + c
        curvature = 6.0f0 * a * t + 2.0f0 * b
        direction * (-derivative / curvature) > ALPGeometryModel.FINAL_MOVE_CAP/ALPGeometryModel.BETA_BASE || error("reference does not reach step clamp")
    end
    cases = map(specifications) do specification
        sigma = ALPGeometryModel.corrected_sigma(
            transform, specification.samples, specification.anchor, specification.maximum, compatibility_lower,
        )
        (sigma !== nothing) == specification.applied || error("reference branch differs: $(specification.name)")
        merge(specification, (; public_sigma = sigma === nothing ? specification.anchor : sigma))
    end
    cases[1].public_sigma == cases[2].public_sigma || error("reference polarity changes sigma")
    for name in ("compatible_gaussian_stationary_minimum","compatible_gaussian_stationary_maximum","compatibility_margin_stationary_maximum")
        case=only(filter(case->case.name==name,cases))
        abs(log2(case.public_sigma/case.anchor))>ALPGeometryModel.FINAL_MOVE_CAP || error("reference does not relax bounded fallback: $name")
    end
    for beta in (ALPGeometryModel.BETA_BASE,ALPGeometryModel.BETA_BOOST)
        _,center=ALPGeometryModel.blended_move(0.125f0,beta,0f0,-0.125f0,1f0)
        for neighbor in (prevfloat(beta),nextfloat(beta))
            _,nearby=ALPGeometryModel.blended_move(0.125f0,clamp(neighbor,ALPGeometryModel.BETA_BASE,ALPGeometryModel.BETA_BOOST),0f0,-0.125f0,1f0)
            abs(nearby-center)<=8eps(Float32)*0.125f0 || error("blend endpoint is discontinuous")
        end
    end
    _,quadratic=ALPGeometryModel.blended_move(0.125f0,ALPGeometryModel.BETA_BOOST,0f0,-0.125f0,1f0)
    quadratic==0.125f0 || error("quadratic stationary root differs")
    return cases
end


function profile_cases(result)
    # Exact Gaussian line plus a declared orthogonal profile perturbation.
    x=Float64.(collect(ALPGeometryModel.X))
    dx=x.-sum(x)/4
    direction=[1.,-1.,0.,0.]
    direction.-=dx*(dot(dx,direction)/dot(dx,dx))
    direction/=norm(direction)
    line=x.+2.4^2
    function perturbed(score)
        epsilon=sqrt(score*dot(line,line)/(1-score))
        z=line+epsilon*direction
        Float32.(x./z.^2)
    end
    entries=[
        (name="gaussian_line",samples=Float32.(x./line.^2)),
        (name="below_envelope",samples=perturbed(Float64(result.compatibility_lower)/4)),
        (name="margin_profile",samples=perturbed(2Float64(result.compatibility_lower))),
        (name="outside_envelope",samples=perturbed(8Float64(result.compatibility_lower))),
        (name="zero_response",samples=zeros(Float32,4)),
        (name="mixed_polarity",samples=Float32[1,-1,1,-1]),
        (name="nonfinite_response",samples=Float32[NaN,1,1,1]),
        (name="overflowed_quotient",samples=fill(nextfloat(0f0),4)),
    ]
    push!(entries,(name="polarity_reversed_line",samples=-entries[1].samples))
    push!(entries,(name="scaled_gaussian_line",samples=16f0.*entries[1].samples))
    map(entries) do entry
        statistic=ALPGeometryModel.compatibility_statistic(entry.samples)
        beta=ALPGeometryModel.compatibility_beta(entry.samples,result.compatibility_lower)
        merge(entry,(;statistic,beta))
    end
end

function generated_rust(result, cases, profiles)
    transform=result.transform
    io = IOBuffer()
    println(io, "// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors")
    println(io, "// SPDX-License-Identifier: MIT OR Apache-2.0\n")
    println(io, "// @generated by scripts/alp/generate_alp_geometry_tables.jl")
    println(io, "// Generator version: $GEOMETRY_GENERATOR_VERSION. Do not edit by hand.\n")
    constants = [
        ("LOG_SIGMA_ORIGIN", ALPGeometryModel.RUNTIME_ORIGIN),
        ("FIT_T_MIN", ALPGeometryModel.FIT_T_DOMAIN[1]),
        ("FIT_T_MAX", ALPGeometryModel.FIT_T_DOMAIN[2]),
        ("ACCEPT_SIGMA_MIN", ALPGeometryModel.PUBLIC_SIGMA_DOMAIN[1]),
        ("ACCEPT_SIGMA_MAX", ALPGeometryModel.PUBLIC_SIGMA_DOMAIN[2]),
        ("BLEND_BETA", ALPGeometryModel.BETA_BASE),
        ("BLEND_BOOST", ALPGeometryModel.BETA_BOOST),
        ("COMPATIBILITY_LOWER", result.compatibility_lower),
        ("COMPATIBILITY_UPPER", result.compatibility_upper),
        ("FINAL_MOVE_CAP_LOG2", ALPGeometryModel.FINAL_MOVE_CAP),
        ("CURVATURE_EPSILON", ALPGeometryModel.CURVATURE_EPSILON),
    ]
    for (name, value) in constants
        println(io, "pub(super) const ALP_GEOMETRY_$name: f32 = ", rust_float(value), ";")
    end
    println(io, "\n#[rustfmt::skip]")
    println(io, "pub(super) const ALP_GEOMETRY_SCALE_CUBIC_LOG2: [[f32; 4]; 4] = [")
    for row in eachrow(transform)
        println(io, "    [", join(rust_float.(row), ", "), "],")
    end
    println(io, "];\n")
    println(io, "/// Julia oracle cases for the anchor-to-public-sigma mapping.")
    println(io, "#[cfg(test)]\npub(super) struct AlpGeometryReferenceCase {")
    println(io, "    pub(super) name: &'static str,")
    println(io, "    pub(super) sample_bits: [u32; 4],")
    println(io, "    pub(super) anchor_sigma_bits: u32,")
    println(io, "    pub(super) is_maximum: bool,")
    println(io, "    pub(super) correction_applied: bool,")
    println(io, "    pub(super) expected_public_sigma_bits: u32,\n}\n")
    println(io, "#[cfg(test)]\n#[rustfmt::skip]")
    println(io, "pub(super) const ALP_GEOMETRY_REFERENCE_CASES: &[AlpGeometryReferenceCase] = &[")
    for case in cases
        println(io, "    AlpGeometryReferenceCase {")
        println(io, "        name: \"", case.name, "\",")
        println(io, "        sample_bits: [", join(rust_bits.(case.samples), ", "), "],")
        println(io, "        anchor_sigma_bits: ", rust_bits(case.anchor), ",")
        println(io, "        is_maximum: ", case.maximum, ",")
        println(io, "        correction_applied: ", case.applied, ",")
        println(io, "        expected_public_sigma_bits: ", rust_bits(case.public_sigma), ",\n    },")
    end
    println(io, "];")
    println(io, "\n#[cfg(test)]\npub(super) struct AlpGeometryProfileCase {")
    println(io, "    pub(super) name: &'static str,")
    println(io, "    pub(super) sample_bits: [u32; 4],")
    println(io, "    pub(super) statistic_bits: Option<u32>,")
    println(io, "    pub(super) beta_bits: u32,\n}\n")
    println(io, "#[cfg(test)]\n#[rustfmt::skip]")
    println(io, "pub(super) const ALP_GEOMETRY_PROFILE_CASES: &[AlpGeometryProfileCase] = &[")
    for profile in profiles
        statistic=profile.statistic===nothing ? "None" : "Some($(rust_bits(profile.statistic)))"
        println(io, "    AlpGeometryProfileCase {")
        println(io, "        name: \"", profile.name, "\",")
        println(io, "        sample_bits: [",join(rust_bits.(profile.samples),", "),"],")
        println(io, "        statistic_bits: ",statistic,",")
        println(io, "        beta_bits: ",rust_bits(profile.beta),",\n    },")
    end
    println(io, "];")
    return String(take!(io))
end

function generated_report(result, cases, rust)
    return """
    # ALP public-scale report

    Generated with Julia `$(VERSION)`, generator version `$GEOMETRY_GENERATOR_VERSION`.

    - Rust artifact SHA-256: `$(bytes2hex(sha256(rust)))`
    - Matrix SHA-256, row-major little-endian Float32: `$(matrix_digest(result.transform))`

    ## Target and objective

    Three equally weighted Gaussian regimes provide $(result.case_count) cases.
    Native bases 0.5 and 1.6 each use 21 log-uniform physical widths in `[1.7,4.0]`
    and phases `{-0.4,0,0.4}²`. The first-transition regime uses 21 fine widths
    in `[3.25,4.2]` with the same fine phases, blurs from base 0.5 to 3.2, and
    retains every second pixel; local widths and phases are halved.
    A native input has unit peak and width `Float32(sqrt(σ²+base²))`.

    Let `x=r/(-2 R(σtrue))`, where the known response is generated by the finite
    filter and normalization uses Float64. Set `t=log2(σtrue/1.6)`,
    `b=[t³,t²,t,1]`, `d=[3t²,2t,1,0]`, and `λ=(1/4)/ln(2)^4`.

    ```text
    Jsignal(T) = mean_case[(bᵀ T x + 1/2)² + λ(dᵀ T x)²].
    G = (Kfirstᵀ Kfirst + Krecurringᵀ Krecurring + Ktransitionᵀ Ktransition)/3.
    Jkernel(T) = q mean_probe[bᵀ T G Tᵀ b + λ dᵀ T G Tᵀ d].
    minimize Jsignal + Jkernel, subject to T·1=[0,0,0,1]ᵀ.
    ```

    The 41 independent probes are log-uniform over `[1.625,4.0]`.
    `q=2/(12·255²)` is a declared reference kernel-variance preference.
    The common Gram has no per-case normalization and is not an exact camera
    noise model. The derivative preference assigns one quarter to local
    octave-displacement risk under normalized continuum Gaussian curvature.
    This Gaussian calibration prior does not specify all image structures.

    ## Finite filters and solve

    Float32 Gaussian taps use `τ=max(√(s²-base²),0.1)`, radius `ceil(3τ)`,
    sequential normalization and paired additions. Signal convolution is
    vertical then horizontal; runtime filtering is horizontal then vertical,
    so rounding may differ. Responses use the five-point `s²` Laplacian.
    Native impulse surrogates have support `[-20,20]²`; the transition operator
    composes rounded taps in Float64 on `[-50,50]²` with coarse Laplacian step 2.
    A generated Cholesky factor of the 4×4 Gram supplies regularizer rows.

    Helmert contrasts eliminate the four constant constraints. The centered
    coordinate `u=(t-0.75)/0.75` leaves 12 free variables solved by Float64 QR,
    followed by conversion to the public log basis and Float32.

    - Design rows: `$(result.row_count)`
    - Centered design condition: `$(@sprintf("%.9g", result.centered_condition))`
    - Public-log design condition: `$(@sprintf("%.9g", result.public_log_condition))`
    - Weighted SSE: `$(@sprintf("%.9g", result.objective64))`
    - Effective quantized-coefficient weighted SSE: `$(@sprintf("%.9g", result.objective32))`

    ## Evaluation and scale policy

    Rust evaluates three differences `δj=rj-r4` with the first three quantized
    weights per row. The constant coefficient adds `r4`; the effective fourth
    column follows the constraint. Equal finite responses have exactly zero
    slope and curvature and report the anchor through fallback.

    ## Gaussian profile compatibility

    For an ideal centered Gaussian, `R(s)=-2 A σ² s²/(σ²+s²)²`, so
    `z=sqrt(s²/abs(R(s)))` is affine in `s²`, with positive slope and intercept.
    Four Float32 inverse square roots form `z`; division by its maximum bounds
    the line fit. Centered least squares computes `S=sum(residual²)/sum(z²)`.
    Non-finite, zero, mixed-sign or nonpositive-line inputs select base damping.
    This is a compatibility statistic, not a Gaussian probability or noise model.
    Finite filters, phase, mixtures and additive noise can affect it.

    The lower envelope `η` is `nextfloat(max(S))` over the same 567 generated
    Gaussian cases; the upper envelope is `4η`. Their bits are
    `$(rust_bits(result.compatibility_lower))` and `$(rust_bits(result.compatibility_upper))`.
    Damping is `4/5` when `S≤η`, `2/5` when `S≥4η` or unsupported, and
    linearly interpolated between those values inside the margin.
    Compatibility uses four square roots and a four-point line fit per eligible
    head. Its shape statistic selects how much of the finite-model stationary move
    to retain; it does not certify a correspondence or camera-noise probability.
    Gain invariance holds in exact arithmetic and is approximate in Float32.

    At the anchor, `d=p'=(3at+2b)t+c`, `h=p''=6at+2b`, and curvature must match
    polarity with epsilon `1e-6`. The fallback is
    `u=clamp(beta*(-d/h),±Float32(1/40))` with compatibility weight
    `w=(beta-2/5)/(4/5-2/5)`. A stationary offset solves
    `d+h*delta+3a*delta²=0`. Its matching-curvature root is
    `delta=-2d/(h+sign(h)*sqrt(h²-12ad))`. Require finite positive discriminant
    and finite nonzero denominator; the formula includes `a=0` without a
    degree tolerance. The proposal `u+w*(delta-u)` continuously blends this
    stationary point with the bounded fallback. Zero weight and invalid root
    arithmetic retain the bounded move, preserving infinite-ratio behavior.
    Accept a proposal only within the fitted scale domain, with finite matching
    curvature, no greater absolute derivative than at the bounded point, and
    response improvement toward the selected minimum or maximum. Evaluate the
    cubic response change as `delta*(p'+delta*(p''/2+a*delta))` at that bounded
    point, avoiding subtraction of large constant terms. Failed proposals keep
    the bounded result; invalid bounded results report the anchor.
    Fitting uses nominal Float64 origin 1.6; runtime uses Float32 1.6.
    Require `t∈[0,1.4999999f0]` and `σ∈[1.6f0,4.3f0]`; failure reports the exact
    anchor. The $(length(cases)) correction oracle cases cover polarity,
    clamps, boundaries, compatible Gaussian stationarity, margin continuity, flat/small-curvature and
    non-finite inputs with two ULP applied-sigma tolerance and exact branch/fallback
    expectations. Profile oracles check the ordered statistic and damping.

    The base envelope and Q fit define anchor position, octave ownership,
    orientation, descriptor support and ordering. This head corrects public
    sigma only. Descriptors interpolate stored Gaussian-plane gradients at the
    anchor. Consumers of published scale observe the corrected estimate.
    """
end

function write_or_check(path, contents, check)
    if check
        isfile(path) && read(path, String) == contents || error("generated file differs: $path")
    else
        write(path, contents)
    end
end

function main_geometry()
    all(argument -> argument == "--check", ARGS) || error("unknown generator argument")
    check = "--check" in ARGS
    result = ALPGeometryModel.derive()
    matrix_digest(result.transform) == GEOMETRY_MATRIX_SHA256 || error("public-scale matrix bits differ")
    cases = reference_cases(result)
    profiles = profile_cases(result)
    rust = generated_rust(result, cases, profiles)
    report = generated_report(result, cases, rust)
    crate_dir = normpath(joinpath(@__DIR__, "..", ".."))
    write_or_check(joinpath(crate_dir, "src", "extraction", "alp", "geometry_tables.rs"), rust, check)
    write_or_check(joinpath(@__DIR__, "alp_geometry_report.md"), report, check)
    println(check ? "ALP public-scale artifacts match" : "ALP public-scale artifacts generated")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main_geometry()
