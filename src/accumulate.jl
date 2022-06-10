# store a 1-spin binary probability distribution. `p` is plus, `m` is minus
Bin{T} = NamedTuple{(:p,:m), Tuple{T,T}} where T
zero(::Bin{T}) where T = Bin{T}((zero(T),zero(T)))
zero(::Type{Bin{T}}) where T = Bin{T}((zero(T),zero(T)))
convert(::Type{Bin{T}}, t::Tuple{T,T}) where T = Bin{T}(t)

# store a 2-spin joint probability distribution. `pp` is plus-plus, etc.
Bin2{T} = NamedTuple{(:pp,:pm, :mp, :mm), Tuple{T,T,T,T}} where T
zero(::Bin2{T}) where T = Bin2{T}((zero(T),zero(T),zero(T),zero(T)))
zero(::Type{Bin2{T}}) where T = Bin2{T}((zero(T),zero(T),zero(T),zero(T)))
convert(::Type{Bin2{T}}, t::Tuple{T,T,T,T}) where T = Bin2{T}(t)

normalize(t::Bin{T}) where T = Bin{T}(( t[:p] / sum(t), t[:m] / sum(t) ))
magnetization(t::Bin) = (t[:p] - t[:m]) / sum(t)
magnetization(t::Bin2) = (t[:pp] + t[:mm]- t[:pm] - t[:mp]) / sum(t)

function accumulate_left!(l::OffsetVector{Bin{T}, Vector{Bin{T}}}, J, h, β) where T
    l[0] = zero(l[0])
    for i in 1:lastindex(h)-1
        l[i] = (1/β*logaddexp( β*(+h[i]+J[i]+l[i-1][:p]), β*(-h[i]-J[i]+l[i-1][:m]) ), 
                1/β*logaddexp( β*(+h[i]-J[i]+l[i-1][:p]), β*(-h[i]+J[i]+l[i-1][:m]) ) )
    end
    l
end
function accumulate_left(J::Vector{T}, h::Vector{T}, β::T) where T
    l = fill(zero(Bin{T}), 0:length(h)-1)
    accumulate_left!(l, J, h, β)
end

function accumulate_right!(r::OffsetVector{Bin{T}, Vector{Bin{T}}}, J, h, β) where T
    r[end] = zero(r[end])
    for i in lastindex(h):-1:2
        r[i] = (1/β*logaddexp( β*(+h[i]+J[i-1]+r[i+1][:p]), β*(-h[i]-J[i-1]+r[i+1][:m]) ), 
                1/β*logaddexp( β*(+h[i]-J[i-1]+r[i+1][:p]), β*(-h[i]+J[i-1]+r[i+1][:m]) ) )
    end
    r
end
function accumulate_right(J::Vector{T}, h::Vector{T}, β::T) where T
    r = fill(zero(Bin{T}), 2:length(h)+1)
    accumulate_right!(r, J, h, β)
end