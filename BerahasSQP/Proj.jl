# implement projection function
# A, B, C: variable, left bound, right bound
function Proj(A, B, C)
    if A <= B
        return B
    elseif A >= C
        return C
    else
        return A
    end
        
end
