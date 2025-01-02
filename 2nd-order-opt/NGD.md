# Natural Gradient Descent Method 

## Inverse-Free Fast Natural Gradient Descent Method for Deep Learning

## Randomized K-FACs: Speeding up K-FAC with  Randomized Numerical Linear Algebra

### Key idea

- Owing to the exponential-average construction paradigm of the Kronecker factors that is typically used, their eigen-spectrum must decay.
- could save substantial computation by only focusing on the first few eigen-modes when inverting the Kronecker-factors. 
- Importantly, the spectrum decay happens over a constant number of modes irrespectively of the layer width. 
- This allows us to reduce the time complexity of kfac from cubic to quadratic in layer width

1. K-FAC with eigen-decomposition ->
2. Randomized SVD
3. Symmetric Randomized EVD (SREVD)

we only really need a constant (w.r.t. dM ) number of modes

This is good news for k-fac: its bottleneck was the scaling of evd with the net width

Proposition 3.1 tells us we have to retain at least nM rǫ = 29184 eigenmodes to ensure we only ignore eigenvalues satisfying λi ≤ 10−1.5λM

we have much less information in the K-factor estimate than would be required to estimate it accurately given its size dM ×dM

### Conclusion
with minimal accuracy loss, we may replace the full eigendecomposition performed by k-fac with rNLA algorithms which only approximate the strongest few modes.

Importantly, the eigen-spectrum decay was shown (theoretically and numerically) to be such that we only really need to keep a constant number of modes when maintaining a fixed, very good accuracy, irrespectively of what the layer width is! This allowed us to reduce the time complexity from O(d3M ) for k-fac  down to O(d2M (r+rl)) for Randomized K-FACs, where r and rl are constant w.r.t.