## This function estimate the gradient and Hessian for AdapSQP

function Estgradf(nx,Xi,nabf_k,c_k,nab_xL_k,CovM,alpha,kap_grad,p_grad,C_grad,rho,MAX_SAMPLE = 1e5)
    while true
        bnabf_k = mean(rand(MvNormal(nabf_k, CovM), convert(Int64, floor(Xi))),dims = 2)
        bnab_xL_k = bnabf_k + nab_xL_k - nabf_k
        if Xi >= MAX_SAMPLE
            return bnabf_k, bnab_xL_k, Xi
        else
            Quant1 = min(kap_grad^2*alpha^2*norm([bnab_xL_k;c_k])^2, 1)
            Quant2 = C_grad*log(4*nx/p_grad)/Quant1
            if Xi >= Quant2
                return bnabf_k, bnab_xL_k, Xi
            else
                Xi *= rho
            end
        end
    end
end
