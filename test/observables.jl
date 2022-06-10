# Loop once over all 2^N states and compute observables
function Obs(f::Function)
    o = 0.0
    function measure(x::IsingChain, s) 
        o += f(x, s)
    end
end
function observables_bruteforce(x::IsingChain, 
        observables::Vector{<:Function})
    if nspins(x) > 10
        @warn "Exponential scaling alert"
    end
    for s in Iterators.product(fill((-1,1),nspins(x))...)
        for f! in observables  
            f!(x, s)
        end
    end
    [obs.o.contents for obs in observables]
end

N = 10
J = 0.5*randn(N-1)
h = 1.2*randn(N)
β = 2.3

x = IsingChain(J, h, β)

@testset "normalization" begin
    _normaliz = (x, s) -> exp(-x.β*energy(x, s))
    Z_bruteforce = observables_bruteforce(x, [Obs(_normaliz)])[1]
    @test x.F ≈ -1/β*log(Z_bruteforce)
end

@testset "magnetizations" begin
    m = site_magnetizations(x)
    _magnetiz = [Obs((x, s) -> pdf(x, s)*s[i]) for i in 1:nspins(x)]
    magnetiz_bruteforce = observables_bruteforce(x, _magnetiz)
    @test all(1:nspins(x)) do i 
        m[i] ≈ magnetiz_bruteforce[i]
    end
end

@testset "neighbor magnetizations" begin
    p = neighbor_magnetizations(x)
    _neig_magnetiz = [Obs((x, s) -> pdf(x, s)*s[i]*s[i+1]) 
                                        for i in 1:nspins(x)-1]
    neig_magnetiz_bruteforce = observables_bruteforce(x, _neig_magnetiz)
    @test all(1:nspins(x)-1) do i
        p[i] ≈ neig_magnetiz_bruteforce[i]
    end
end

# @testset "pair magnetizations" begin
#     p = pair_magnetizations(x)
#     _pair_magnetiz = [Obs((x, s) -> pdf(x, s)*s[i]*s[j]) 
#                                         for i in 1:nspins(x) for j in 1:nspins(x)]
#     pair_magnetiz_bruteforce = observables_bruteforce(x, _pair_magnetiz)
#     @test all(Iterators.product(1:nspins(x),1:nspins(x))) do (i,j) 
#         k = Int( (j-1)*N + i )
#         p[i,j] ≈ pair_magnetiz_bruteforce[k]
#     end
# end

# @testset "correlations" begin
#     m = site_magnetizations(x)
#     c = correlations(x)
#     _correl = [Obs((x, s) -> pdf(x, s)*(s[i]*s[j]-m[i]*m[j])) 
#                                         for i in 1:x.N for j in 1:x.N]
#     correl_bruteforce = observables_bruteforce(x, _correl)
#     @test all(Iterators.product(1:x.N,1:x.N)) do (i,j) 
#         k = Int( (j-1)*N + i )
#         isapprox( c[i,j], correl_bruteforce[k], atol=1e-4 )
#     end
# end

@testset "average energy" begin
    U = avg_energy(x)
    _energy = Obs((x,s) -> pdf(x,s)*energy(x,s))
    avg_energy_bruteforce = observables_bruteforce(x, [_energy])[1]
    @test U ≈ avg_energy_bruteforce
end

@testset "entropy" begin
    S = entropy(x)
    _entropy = Obs((x,s) -> -pdf(x,s)*log(pdf(x,s)))
    entropy_bruteforce = observables_bruteforce(x, [_entropy])[1]
    @test S ≈ entropy_bruteforce
end
