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
    @test free_energy(x) ≈ -1/β*log(Z_bruteforce)
    @test normalization(x) ≈ Z_bruteforce
end

@testset "magnetizations" begin
    m = site_magnetizations(x)
    _magnetiz = [Obs((x, s) -> pdf(x, s)*s[i]) for i in 1:nspins(x)]
    magnetiz_bruteforce = observables_bruteforce(x, _magnetiz)
    @test all(1:nspins(x)) do i 
        m[i] ≈ magnetiz_bruteforce[i]
    end
end

@testset "marginals" begin
    m = reduce(vcat, collect.(site_marginals(x)))
    _marg = [Obs((x, s) -> pdf(x, s)*(s[i]==a)) 
            for i in 1:nspins(x) for a in (1,-1)]
    marg_bruteforce = observables_bruteforce(x, _marg)
    @test all(1:2N) do i 
        m[i] ≈ marg_bruteforce[i]
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

@testset "pair magnetizations" begin
    p = pair_magnetizations(x)
    _pair_magnetiz = [Obs((x, s) -> pdf(x, s)*s[i]*s[j]) 
                                        for i in 1:N for j in 1:N]
    pair_magnetiz_bruteforce = observables_bruteforce(x, _pair_magnetiz)
    @test all(Iterators.product(1:N,1:N)) do (i,j) 
        k = Int( (j-1)*N + i )
        p[i,j] ≈ pair_magnetiz_bruteforce[k]
    end
end

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
