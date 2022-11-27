## This function estimate the gradient and Hessian for AdapSQP

function EstLandM(nx,nlam,Xi,G_k,c_k,nab_xL_k,nab_x2L_k,nab_x2c_k,sigma,CovM,nu,alpha,kap_grad,p_grad,C_grad,rho,MAX_SAMPLE = 1e5)
    while true
        bnab_xL_k = mean(rand(MvNormal(nab_xL_k, CovM), convert(Int64, floor(Xi))),dims = 2)
        Delta = rand(Normal(0,(sigma/Xi)^(1/2)), nx, nx)
        bnab_x2L_k = Hermitian(nab_x2L_k, :L) + (Delta + Delta')/2
        bT_k = zeros(nx, nlam)
        for i = 1:nlam
            bT_k[:, i] = Hermitian(nab_x2c_k[i],:L)*bnab_xL_k
        end
        bM_k = bnab_x2L_k * G_k' + bT_k

        if Xi >= MAX_SAMPLE
            return bnab_xL_k, bM_k, Xi
        else
            Quant1 = norm([bnab_xL_k+nu*bM_k*G_k*bnab_xL_k+G_k'c_k])^2
            Quant2 = norm([nu*G_k*G_k'*G_k*bnab_xL_k])^2
            Quant3 = min(kap_grad^2*alpha^2*(Quant1+Quant2) , 1)
            Quant4 = C_grad*log(4*nx/p_grad)/Quant3
            if Xi >= Quant4
                return bnab_xL_k, bM_k, Xi
            else
                Xi *= rho
            end
        end
    end
end
