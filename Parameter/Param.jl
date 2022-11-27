
NonAdap = Parameter.NonAdapParams(true,
                    [0.01, 0.1, 0.5, 1],   # const stepsize
                    [0.6, 0.9],            # decay stepsize
                    100000,                # Max_Iter
                    5,                     # Rep
                    1e-6,                  # EPS
                    1e-4,                  # EPS
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma


Berahas = Parameter.BerahasParams(true,
                    100000,  # Max_Iter
                    1e-6,    # EPS
                    1e-4,    # EPS
                    5,       # Rep
                    1,       # tau
                    1e-6,    # epsilon
                    0.5,     # sigma
                    1,       # xi
                    10,      # theta
                    [0.01, 0.1, 0.5, 1],   # const stepsize
                    [0.6, 0.9],            # decay stepsize
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma

Adap = Parameter.AdapParams(true,
                    100000,  # Max_Iter
                    1e-6,    # EPS_Step
                    1e-4,    # EPS_Res
                    0.001,   # nu
                    1,       # mu
                    1,       # epsilon
                    5,       # Rep
                    0.3,     # beta
                    1.2,     # rho
                    1.5,     # alpha_max
                    1,       # kap_grad
                    0.05,    # kap_f
                    0.1,     # p_grad
                    0.1,     # p_f
                    [1,5,10,50],                   # C_grad
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma


AdapL1 = Parameter.AdapL1Params(true,
                    100000,  # Max_Iter
                    1e-6,    # EPS_Step
                    1e-4,    # EPS_Res
                    1,       # mu
                    1,       # epsilon
                    5,       # Rep
                    0.3,     # beta
                    1.2,     # rho
                    1.5,     # alpha_max
                    1,       # kap_grad
                    0.05,    # kap_f
                    0.1,     # p_grad
                    0.1,     # p_f
                    [1,5,10,50],                   # C_grad
                    [1e-8, 1e-4, 1e-2, 1e-1, 1])   # Sigma
