# ALP generated-table report

Generated with Julia `1.12.6`, generator version `4`.

- Rust artifact SHA-256: `f1395655c6f06549319c874864376f0871d56ce6613ac2fc3839c4415a044544`
- Scale matrix SHA-256, row-major little-endian Float32: `edbf0e59d94e65cb6d3b153d121603c4a3876a63895f87453e11781379c0bbf6`
- Sample-sigma bit digest: `020087eaeb96d8a7e701ccdc7545839223e9aaad92f3b50617d0b2036477b599`
- Spatial-Q bit digest: `7eea90997766cdca9be87adc856617b81217c429c26d092b6a70e778b14da4dd`

## Finite response objective

Four level responses map to `[a,b,c,d]` in `P(σ)=aσ³+bσ²+cσ+d`.
The fit minimizes equal-weight Gaussian and sinusoidal response error.
Each input and target profile is divided by its integrated response norm
`N²=⟨R(s)²⟩`, with the profile average uniform in log scale over the four-node
interval. The constant-response constraint is `T·1=[0,0,0,1]ᵀ`.

```text
E[⟨(v(s)ᵀ T r - R(s))²⟩log(s) / ⟨R(s)²⟩log(s)],
v(s) = [s³,s²,s,1],  T·1 = [0,0,0,1]ᵀ.
```

Here `r` contains the four input responses and `R(s)` is the target profile.

Gaussian inputs are `exp(-r²/(2(σ²+b²)))` with phase uniform in
`[-0.5,0.5]²`. Wave inputs are `exp(-b²k²/2)·cos(k(cosθ x+sinθ y))`, with `k=√2/σ`
and direction uniform in `[0,π/2]`. The bases `b=0.5` and `b=1.6` have equal
weight. Characteristic sigma is uniform in log scale over
`[1.9027314f0,4.1498866f0]`.

The quadrature uses 32 characteristic scales, 3 points per phase coordinate,
and 8 wave directions. Target-profile integration uses 12 points per interval
split at Gaussian-radius transitions `√(b²+(j/3)²)`. The four sample nodes
are `1.6, 2.262742, 3.2, 4.525483`.

The model uses normalized finite Gaussian coefficients, radius `ceil(3τ)`,
`τ=max(√(s²-b²),0.1)`, and a five-point `s²` Laplacian in Float32. Its
convolution order is vertical then horizontal; runtime image filtering is
horizontal then vertical. Rounding can differ between these orders.
Gaussian normalization applies its known finite peak before the integrated
norm; both normalization stages, quadrature, and QR use Float64.

Sigma is centered and scaled to `[-1,1]`. A fixed orthonormal contrast basis
eliminates the constant constraint, leaving 12 variables for a QR solve.
The matrix is converted to physical-sigma coefficients and rounded to Float32.

- Weighted design rows: `150144`
- Design condition: `144.710971`
- Weighted residual norm, `sqrt(weighted SSE)`: `0.00670486581`

## Float32 evaluation

All 16 fitted weights are serialized. Runtime uses the first three quantized
weights per row and sets `δj=rj-r3`, `j=0..2`. Each row starts with
`value=T[row,0]*δ0`, then adds `T[row,1]*δ1` and `T[row,2]*δ2`.
The constant row adds `r3` to that sum. The effective fourth column follows
the constant constraint; the stored fourth entry is the independently rounded
fit value. Equal finite responses give a flat polynomial by this arithmetic.
Rust consumes 5 generated coefficient reference vectors covering
zero, constant, signed interior responses, and a large constant reference.

## Spatial localization

Nominal nodes are `(8/5)·√2^k`, `k=0..3`, serialized to six decimal places.
The spatial model uses the analytic scale-normalized LoG
`K(x,y,s)=((x²+y²)/s²-2)·exp(-(x²+y²)/(2s²))/s²`, a 19×19 integer kernel
lattice, nine integer shifts, and a 21×21 offset grid in `[-1,1]²`.
Two QR solves fit the quadratic basis `[x²,y²,xy,x,y,1]` and project its
coefficient kernels onto the nine samples. Square-domain symmetries are
imposed at 192-bit precision before six-decimal rounding and Float32 conversion.
Exact zero signs are specified by four serialization masks.
Runtime row-major patches use permutation `[0,3,6,1,4,7,2,5,8]`.

## References

- G. Koutaki and K. Uchimura, *Scale-space Processing Using Polynomial
  Representations*, CVPR, 2014.
- G. H. Golub and J. H. Welsch, *Calculation of Gauss Quadrature Rules*,
  Mathematics of Computation 23(106), 1969.
