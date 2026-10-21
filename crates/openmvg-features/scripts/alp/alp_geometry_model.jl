# SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Finite Gaussian cases + common kernel variance -> constrained 12-variable QR
#   -> log-scale cubic -> safeguarded stationary scale for public sigma.
module ALPGeometryModel
using LinearAlgebra

const NODES = Float32.(round.(1.6 .* exp2.((0:3)./2); digits=6))
const FIT_ORIGIN = 1.6
const RUNTIME_ORIGIN = 1.6f0
const DERIVATIVE_PREFERENCE = 1/4
const LAMBDA = DERIVATIVE_PREFERENCE/log(2.0)^4
const VARIANCE_PREFERENCE = 2.0
const Q = VARIANCE_PREFERENCE/(12*255.0^2)
const LOG_CENTER = 0.75
const LOG_HALF_WIDTH = 0.75
const PHASES = (-0.4,0.0,0.4)
const CURVATURE_EPSILON = 1f-6
const FIT_T_DOMAIN = (0f0,1.4999999f0)
const PUBLIC_SIGMA_DOMAIN = (1.6f0,4.3f0)

function half_kernel(target::Float32, base::Float32)
    tau = max(sqrt(max(target*target-base*base,0f0)),0.1f0)
    radius = ceil(Int,3f0*tau)
    full = Float32[]; total = 0f0
    for i=-radius:radius
        d=Float32(i)/tau; value=exp(-0.5f0*d*d)
        push!(full,value); total+=value
    end
    Float32[full[radius+1+i]/total for i=0:radius]
end

# Unit peak is declared at the native modeled input, not before an image PSF.
function blob(sigma,base,px,py,x,y)
    width=Float32(sqrt(sigma*sigma+base*base))
    distance=(Float32(x)-Float32(px))^2+(Float32(y)-Float32(py))^2
    exp(-distance/(2f0*width*width))
end

# Center-first paired Float32 additions, vertical pass then horizontal pass.
function patch(sigma,base,px,py,node)
    h=half_kernel(Float32(node),Float32(base)); radius=length(h)-1
    xs=(-2-radius):(2+radius)
    mid=Matrix{Float32}(undef,5,length(xs))
    for (ix,x) in enumerate(xs),(iy,y) in enumerate(-2:2)
        value=blob(sigma,base,px,py,x,y)*h[1]
        for k=1:radius
            value+=(blob(sigma,base,px,py,x,y-k)+blob(sigma,base,px,py,x,y+k))*h[k+1]
        end
        mid[iy,ix]=value
    end
    out=Matrix{Float32}(undef,5,5)
    for iy=1:5,ix=1:5
        j=radius+ix; value=mid[iy,j]*h[1]
        for k=1:radius
            value+=(mid[iy,j-k]+mid[iy,j+k])*h[k+1]
        end
        out[iy,ix]=value
    end
    out
end

function convolve(image,h)
    rows,cols=size(image); radius=length(h)-1
    mid=similar(image); out=similar(image)
    for x=1:cols,y=1:rows
        value=image[y,x]*h[1]
        for k=1:radius
            value+=(image[clamp(y-k,1,rows),x]+image[clamp(y+k,1,rows),x])*h[k+1]
        end
        mid[y,x]=value
    end
    for x=1:cols,y=1:rows
        value=mid[y,x]*h[1]
        for k=1:radius
            value+=(mid[y,clamp(x-k,1,cols)]+mid[y,clamp(x+k,1,cols)])*h[k+1]
        end
        out[y,x]=value
    end
    out
end

function response(image,node)
    cy,cx=(size(image).+1).÷2
    lap=image[cy,cx-1]+image[cy,cx+1]+image[cy-1,cx]+image[cy+1,cx]-4f0*image[cy,cx]
    lap*Float32(node)^2
end

function signal_cases()
    records=NamedTuple[]
    native=exp2.(range(log2(1.7),log2(4.0);length=21))
    fine=exp2.(range(log2(3.25),log2(4.2);length=21))
    for base in (0.5,1.6),sigma in native,px in PHASES,py in PHASES
        samples=Float64[response(patch(sigma,base,px,py,node),node) for node in NODES]
        known=Float64(response(patch(sigma,base,px,py,Float32(sigma)),Float32(sigma)))
        push!(records,(;sigma,base,px,py,samples,known,group=base==0.5 ? :native_first : :native_recurring))
    end
    for fine_sigma in fine,px in PHASES,py in PHASES
        side=97; center=(side+1)÷2
        image=Float32[blob(fine_sigma,0.5,px,py,x-center,y-center) for y=1:side,x=1:side]
        blurred=convolve(image,half_kernel(NODES[3],0.5f0))
        coarse=blurred[1:2:end,1:2:end]
        samples=Float64[response(convolve(coarse,half_kernel(node,NODES[1])),node) for node in NODES]
        sigma=fine_sigma/2
        known=Float64(response(convolve(coarse,half_kernel(Float32(sigma),NODES[1])),Float32(sigma)))
        push!(records,(;sigma,base=1.6,px=px/2,py=py/2,samples,known,group=:first_transition))
    end
    records
end

# Rounded native impulse-kernel surrogate, with enough zeros for full support.
function native_kernel(node,base)
    half=half_kernel(Float32(node),Float32(base)); radius=length(half)-1
    support=20; line=zeros(Float32,2support+1); center=support+1
    line[center]=half[1]
    for k=1:radius
        line[center-k]=half[k+1]; line[center+k]=half[k+1]
    end
    gaussian=Float32[line[y]*line[x] for y in eachindex(line),x in eachindex(line)]
    pad=zeros(Float32,size(gaussian).+2); pad[2:end-1,2:end-1]=gaussian
    out=similar(gaussian)
    for x in axes(out,2),y in axes(out,1)
        lap=pad[y+1,x]+pad[y+1,x+2]+pad[y,x+1]+pad[y+2,x+1]-4f0*gaussian[y,x]
        out[y,x]=lap*Float32(node)^2
    end
    Float64.(vec(out))
end

# Linear lifted transition operator: Float64 composition of declared Float32
# taps. The coarse Laplacian has spacing2 on the original lattice.
function transition_kernel(node)
    fine=half_kernel(NODES[3],0.5f0); coarse=half_kernel(Float32(node),NODES[1])
    support=50; center=support+1; line=zeros(Float64,2support+1)
    for i=-(length(fine)-1):(length(fine)-1),j=-(length(coarse)-1):(length(coarse)-1)
        line[center+i+2j]+=Float64(fine[abs(i)+1])*Float64(coarse[abs(j)+1])
    end
    gaussian=line*line'; out=zeros(size(gaussian)); scale=Float64(Float32(node)^2)
    for x=3:(length(line)-2),y=3:(length(line)-2)
        out[y,x]=(gaussian[y,x-2]+gaussian[y,x+2]+gaussian[y-2,x]+gaussian[y+2,x]-4gaussian[y,x])*scale
    end
    vec(out)
end

function kernel_prior()
    first=hcat((native_kernel(node,0.5) for node in NODES)...)
    recurring=hcat((native_kernel(node,1.6) for node in NODES)...)
    transition=hcat((transition_kernel(node) for node in NODES)...)
    grams=(first'*first,recurring'*recurring,transition'*transition)
    common=(grams[1]+grams[2]+grams[3])/3
    # LL'=common. This4×4 generated Gram factor is not the fit normal matrix.
    factor=Matrix(cholesky(Symmetric(common)).L)
    (;common,factor,grams,kernel_shapes=(size(first),size(recurring),size(transition)))
end

function contrasts()
    Float64[1/sqrt(2) 1/sqrt(6) 1/sqrt(12);
            -1/sqrt(2) 1/sqrt(6) 1/sqrt(12);
            0 -2/sqrt(6) 1/sqrt(12);
            0 0 -3/sqrt(12)]
end

function log_conversion()
    m,h=LOG_CENTER,LOG_HALF_WIDTH
    [1/h^3 0 0 0; -3m/h^3 1/h^2 0 0;
     3m^2/h^3 -2m/h^2 1/h 0; -m^3/h^3 m^2/h^2 -m/h 1]
end

function effective_quantized_matrix(transform)
    effective=Float64.(transform); target=[0.,0.,0.,1.]
    for row=1:4
        effective[row,4]=target[row]-sum(effective[row,1:3])
    end
    effective
end

function derive_transform()
    BLAS.set_num_threads(1)
    cases=signal_cases(); prior=kernel_prior()
    low,high=extrema(case.sigma for case in cases)
    scales=exp2.(range(log2(low),log2(high);length=41))
    rows=2length(cases)+8length(scales)
    A=Matrix{Float64}(undef,rows,16); y=zeros(Float64,rows); index=0
    for case in cases
        normalizer=-2case.known
        isfinite(normalizer) && normalizer>0 || error("Invalid known response")
        x=case.samples/normalizer
        t=log2(case.sigma/FIT_ORIGIN); basis=[t^3,t^2,t,1.]; slope=[3t^2,2t,1.,0.]
        sw=sqrt(1/length(cases))
        index+=1; A[index,:]=sw*kron(x,basis); y[index]=-0.5sw
        index+=1; A[index,:]=sw*sqrt(LAMBDA)*kron(x,slope)
    end
    for sigma in scales
        t=log2(sigma/FIT_ORIGIN); basis=[t^3,t^2,t,1.]; slope=[3t^2,2t,1.,0.]
        sw=sqrt(Q/length(scales))
        for column=1:4
            e=prior.factor[:,column]
            index+=1; A[index,:]=sw*kron(e,basis)
            index+=1; A[index,:]=sw*sqrt(LAMBDA)*kron(e,slope)
        end
    end
    index==rows || error("Design row count mismatch")
    Z=contrasts(); conversion=log_conversion()
    particular=zeros(4,4); particular[4,:].=0.25
    mapping=kron(Z,conversion)
    reduced=A*mapping; targets=y-A*vec(particular)
    factorization=qr(reduced); solution=factorization\targets
    transform64=conversion*(particular+reshape(solution,4,3)*Z')
    transform=Float32.(transform64)
    all(isfinite,transform) || error("Non-finite coefficients")
    effective32=effective_quantized_matrix(transform)
    residual64=A*vec(transform64)-y
    residual32=A*vec(effective32)-y
    constraint64=maximum(abs,transform64*ones(4)-[0.,0.,0.,1.])
    constraint64<1e-10 || error("Constant-input constraint failure")
    signal_rows=1:2length(cases); variance_rows=(2length(cases)+1):rows
    objective64=sum(abs2,residual64); objective32=sum(abs2,residual32)
    # These regenerated diagnostics are outputs only; QR is the fitting solve.
    H,g=2A'*A,2A'*y
    (;transform,transform64,effective32,cases,prior,prior_scales=scales,design=A,targets=y,H,g,
      row_count=rows,case_count=length(cases),constraint64,
      stored_constraint32=maximum(abs,Float64.(transform)*ones(4)-[0.,0.,0.,1.]),
      objective64,objective32,signal_objective64=sum(abs2,residual64[signal_rows]),
      variance_objective64=sum(abs2,residual64[variance_rows]),
      centered_condition=cond(Matrix(factorization.R)),
      public_log_condition=cond(A*kron(Z,Matrix{Float64}(I,4,4))))
end

function apply_cubic(transform,samples::AbstractVector{Float32})
    length(samples)==4 || error("Four samples required")
    delta=samples[1:3].-samples[4]; coefficients=zeros(Float32,4)
    for row=1:4
        value=transform[row,1]*delta[1]
        value+=transform[row,2]*delta[2]
        value+=transform[row,3]*delta[3]
        coefficients[row]=row==4 ? samples[4]+value : value
    end
    coefficients
end

const BETA_BASE = 2f0/5f0
const BETA_BOOST = 4f0/5f0
const MARGIN_FACTOR = 4f0
const FINAL_MOVE_CAP = Float32(1/40)
const X = ntuple(i->NODES[i]*NODES[i],4)
function sum4(values)
    value=values[1]+values[2]
    value+=values[3]
    value+=values[4]
    value
end
const MEAN_X=sum4(X)*0.25f0
const DX=ntuple(i->X[i]-MEAN_X,4)
const X_ENERGY=sum4(ntuple(i->DX[i]*DX[i],4))

"""Float32 relative line residual, or nothing for unsupported profiles.

All operation ordering is explicit. Uniform gain leaves the statistic unchanged
in exact arithmetic; Float32 rounding and inverse-root noise sensitivity remain.
"""
function compatibility_statistic(samples::AbstractVector{Float32})
    length(samples)==4 || return nothing
    all(isfinite,samples) || return nothing
    (all(>(0f0),samples) || all(<(0f0),samples)) || return nothing
    quotient=ntuple(i->X[i]/abs(samples[i]),4)
    all(q->isfinite(q) && q>0f0,quotient) || return nothing
    root=ntuple(i->sqrt(quotient[i]),4)
    all(z->isfinite(z) && z>0f0,root) || return nothing
    maximum_root=max(max(root[1],root[2]),max(root[3],root[4]))
    isfinite(maximum_root) && maximum_root>0f0 || return nothing
    z=ntuple(i->root[i]/maximum_root,4)
    all(v->isfinite(v) && v>0f0,z) || return nothing
    mean_z=sum4(z)*0.25f0
    dy=ntuple(i->z[i]-mean_z,4)
    numerator=sum4(ntuple(i->DX[i]*dy[i],4))
    slope=numerator/X_ENERGY
    intercept=mean_z-slope*MEAN_X
    all(isfinite,(mean_z,numerator,slope,intercept)) || return nothing
    slope>0f0 && intercept>0f0 || return nothing
    residual=ntuple(i->dy[i]-slope*DX[i],4)
    residual_energy=sum4(ntuple(i->residual[i]*residual[i],4))
    z_energy=sum4(ntuple(i->z[i]*z[i],4))
    all(isfinite,(residual_energy,z_energy)) && z_energy>0f0 || return nothing
    statistic=residual_energy/z_energy
    isfinite(statistic) && statistic>=0f0 || return nothing
    statistic
end

function compatibility_beta(samples::AbstractVector{Float32},lower::Float32)
    statistic=compatibility_statistic(samples)
    statistic===nothing && return BETA_BASE
    upper=MARGIN_FACTOR*lower
    isfinite(lower) && lower>0f0 && isfinite(upper) && upper>lower || return BETA_BASE
    statistic<=lower && return BETA_BOOST
    statistic>=upper && return BETA_BASE
    fraction=(statistic-lower)/(upper-lower)
    BETA_BOOST-(BETA_BOOST-BETA_BASE)*fraction
end

function derive()
    model=derive_transform()
    scores=Float32[]
    for case in model.cases
        score=compatibility_statistic(Float32.(case.samples))
        score===nothing && error("Unsupported generated Gaussian training profile")
        push!(scores,score)
    end
    training_maximum=maximum(scores)
    lower=nextfloat(training_maximum)
    upper=MARGIN_FACTOR*lower
    isfinite(lower) && lower>0f0 && isfinite(upper) || error("Invalid compatibility envelope")
    merge(model,(;training_maximum,compatibility_lower=lower,compatibility_upper=upper))
end

"""Blend the bounded fallback with the matching-curvature stationary root."""
function blended_move(raw_step::Float32,beta::Float32,a::Float32,derivative::Float32,curvature::Float32)
    damped=beta*raw_step
    bounded=clamp(damped,-FINAL_MOVE_CAP,FINAL_MOVE_CAP)
    weight=clamp((beta-BETA_BASE)/(BETA_BOOST-BETA_BASE),0f0,1f0)
    weight==0f0 && return bounded,bounded
    discriminant=curvature*curvature-12f0*a*derivative
    isfinite(discriminant) && discriminant>0f0 || return bounded,bounded
    radical=sqrt(discriminant)
    matched=curvature<0f0 ? -radical : radical
    denominator=curvature+matched
    isfinite(denominator) && denominator!=0f0 || return bounded,bounded
    root_move=-2f0*derivative/denominator
    isfinite(root_move) || return bounded,bounded
    proposal=bounded+weight*(root_move-bounded)
    isfinite(proposal) || return bounded,bounded
    bounded,proposal
end

function sigma_at(t::Float32)
    isfinite(t) && FIT_T_DOMAIN[1]<=t<=FIT_T_DOMAIN[2] || return nothing
    sigma=RUNTIME_ORIGIN*exp(t*Float32(log(2.0)))
    isfinite(sigma) && PUBLIC_SIGMA_DOMAIN[1]<=sigma<=PUBLIC_SIGMA_DOMAIN[2] || return nothing
    sigma
end

function corrected_sigma(transform,samples::AbstractVector{Float32},anchor::Float32,is_maximum::Bool,lower::Float32)
    isfinite(anchor) && anchor>0f0 || return nothing
    a,b,c,_=apply_cubic(transform,samples)
    t=log2(anchor/RUNTIME_ORIGIN)
    derivative=(3f0*a*t+2f0*b)*t+c
    curvature=6f0*a*t+2f0*b
    polarity=is_maximum ? curvature < -CURVATURE_EPSILON : curvature > CURVATURE_EPSILON
    all(isfinite,(t,derivative,curvature)) && polarity || return nothing
    beta=compatibility_beta(samples,lower)
    raw_step=-derivative/curvature
    bounded,proposal=blended_move(raw_step,beta,a,derivative,curvature)
    bounded_t=t+bounded
    fallback=sigma_at(bounded_t)
    fallback===nothing && return nothing
    proposal==bounded && return fallback
    isfinite(proposal) || return fallback
    proposed_t=t+proposal
    sigma=sigma_at(proposed_t)
    sigma===nothing && return fallback
    proposed_derivative=(3f0*a*proposed_t+2f0*b)*proposed_t+c
    proposed_curvature=6f0*a*proposed_t+2f0*b
    polarity=is_maximum ? proposed_curvature < -CURVATURE_EPSILON : proposed_curvature > CURVATURE_EPSILON
    bounded_derivative=(3f0*a*bounded_t+2f0*b)*bounded_t+c
    bounded_curvature=6f0*a*bounded_t+2f0*b
    delta=proposed_t-bounded_t
    gain=delta*(bounded_derivative+delta*(bounded_curvature*0.5f0+a*delta))
    improves=is_maximum ? gain>=0f0 : gain<=0f0
    all(isfinite,(proposed_derivative,proposed_curvature,bounded_derivative,bounded_curvature,gain)) &&
        polarity && abs(proposed_derivative)<=abs(bounded_derivative) && improves || return fallback
    sigma
end
end
