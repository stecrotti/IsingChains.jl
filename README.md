# IsingChains

[![Build Status](https://github.com/stecrotti/IsingChains.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stecrotti/IsingChains.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/stecrotti/IsingChains.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/stecrotti/IsingChains.jl)

A 1D Ising model with open boundary conditions, described by a Boltzmann distribution

>![equation](https://latex.codecogs.com/svg.image?p(\boldsymbol\sigma|\boldsymbol{J},&space;\boldsymbol{h},&space;\beta)&space;=&space;\frac{1}{Z_{\boldsymbol{J},&space;\boldsymbol{h},&space;\beta}}\exp\left[\beta\left(\sum_{i=1}^{N-1}J_i\sigma_i\sigma_{i+1}&space;&plus;\sum_{i=1}^Nh_i\sigma_i\right)\right],\quad\boldsymbol\sigma\in\\{-1,1\\}^N)

is exactly solvable in polynomial time.


| Quantity | Cost          |
| ------------- | ----------- |
| Normalization, Free energy      |  ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N)) |
| Sample a configuration      |  ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N)) |
| Average energy, Entropy |  ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N))  |
| Single-site distributions  | ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N))     |
| Joint distributions of pairs of neighbors | ![equation](https://latex.codecogs.com/svg.image?\mathcal{O}(N))     |

## Example
```
]add IsingChains
```
Construct a `IsingChain` instance
```
using IsingChains, Random

N = 10000
rng = MersenneTwister(0)
J = 2.0*randn(rng, N-1)
h = randn(rng, N)
β = 0.1
x = IsingChain(J, h, β)
```
Compute stuff
```
# normalization and free energy
Z = normalization(x)
F = free_energy(x)

# energy and probability of a configuration
σ = rand(rng, (-1,1), N) 
E = energy(x, σ)
prob = pdf(x, σ)

# a sample along with its log-probability 
σ, logp = sample(rng, x)

# single-site magnetizations <σᵢ>
m = site_magnetizations(x)

# nearest-neighbor magnetizations <σᵢσᵢ₊₁>
pneigs = neighbor_magnetizations(x)

# energy expected value
U = avg_energy(x)

# entropy
S = entropy(x)
```
