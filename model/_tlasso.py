import numpy as np
from rpy2.robjects import r as r
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

from numpy import linalg as la

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + B) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def run_glasso(s, lg):
    print("Start R: glasso")
    r.library("huge")

    # Prepare the call for GLasso
    glasso = r("huge.glasso")

    rpy2.robjects.numpy2ri.activate()
    results = glasso(np.array(s), lg)
    wi = results[4][0]
    return wi

def run_jgl(s, lg, rg):
    print("Start R: JGL")
    r.library("JGL")

    # Prepare the call for GLasso
    glasso = r("JGL")

    rpy2.robjects.numpy2ri.activate()

    results = glasso(s,"group",lg, rg, 1,"equal",False, 500, 1e-5, robjects.NULL,True)
    wi = results[0]
    return wi

def delta_Y(Y, mu, W):
    out = (Y-mu).T.dot(W.dot((Y-mu)))
    return np.diagonal(out)

def tau(Y,nu,p,mu,W, group=False):
    
    if group: t =[(nu+p)/(nu+delta_Y(Y[i], mu[i].reshape((p,1)), W[i])) for i in range(len(Y))]
    else: t =[(nu+p)/(nu+delta_Y(Y[i], mu[i].reshape((p,1)), W)) for i in range(len(Y))]
    return t

def update_mu(Y,taus):
    return [np.sum(Y[i] * taus[i], axis=1) / sum(taus[i]) for i in range(len(Y))]

from scipy import linalg

def pd_inv(a):
    n = a.shape[0]
    I = np.identity(n)
    return linalg.solve(a, I, sym_pos = True, overwrite_b = True)

def update_Sigma_mu(Y,taus,group=False,mu=True):
    p = Y[0].shape[0]
    if mu:
        mu_u = update_mu(Y,taus)
        mu_u = [mu_u[i].reshape((p,1)) for i in range(len(Y))]
        Ymuw=[((Y[i]-mu_u[i])*np.sqrt(taus[i])).T for i in range(len(Y))]
    else: 
        Ymuw=[(Y[i]*np.sqrt(taus[i])).T for i in range(len(Y))]
        mu_u=0
    if group:
        Stdsi = [(1/Ymuw[i].shape[0])*Ymuw[i].T@Ymuw[i] for i in range(len(Y))]
        #print(Stdsi)
        Sigma_u = [(Ymuw[i]/(Ymuw[i]).std(0)) for i in range(len(Y))]#[np.corrcoef(Ymuw[i].T) for i in range(len(Y))]#[(Ymuw[i]/(Ymuw[i]).std(0)) for i in range(len(Y))]
    else:
        Ymuw = np.vstack(Ymuw).T
        Sigma_u = np.corrcoef(Ymuw)#Ymuw.dot((Y-mu_u).T)/n
        Stdsi = np.diag(1/Ymuw.std(1))
    return Ymuw, Sigma_u, Stdsi


    
def em_tlasso(Y, T, l,W0):
    p = Y[0].shape[0]
    nu = 3
    mu = [np.mean(Y[i], axis = 1) for i in range(len(Y))]

    
    W = W0.copy()
    for i in range(T):
        print(f"Iteration: {i}")
        # E-step 
        taus = tau(Y,nu,p,mu,W)
        # M-step
        mu_u, Sigma_u, Stdsi = update_Sigma_mu(Y,taus)
        #print(f"Learning rate:{l*Stdsi}")
        W_cor = run_glasso(Sigma_u, l)
        W = Stdsi@W_cor@Stdsi
        mu = mu_u
        
    return W, mu

def em_tlasso_noise(Y, N, T, l,r, W0, D0, glasso=False):
    p = Y[0].shape[0]
    nu = 3
    k = Y[0].shape[1]
    nk = N[0].shape[1]
    muY = [ np.array([0]*p) for i in range(len(Y))]
    muN = [ np.array([0]*p) for i in range(len(N))]
    
    W = W0.copy()
    D = D0.copy()
    for i in range(T):
        print(f"Iteration: {i}")
        # E-step 
        tausY = tau(Y,3,p,muY,W, group=True)
        tausN = tau(N,3,p,muN,D, group=True)
        # M-step
        _, Sigma_Y, Stdsi_Y = update_Sigma_mu(Y,tausY, mu=False, group=True)
        
        if glasso: W_cor = [run_glasso(np.corrcoef(Sigma_Y[i].T), l) for i in range(len(Y))]
        else: W_cor = run_jgl(Sigma_Y, l,r)
        Stdsi_YT = [run_glasso(Stdsi_Y[i], Stdsi_Y[i].max()) for i in range(len(Y))]
        Stdsi_YN = [np.diag(np.sqrt( np.diag(Stdsi_YT[i])*(1/(np.diag(W_cor[i])+1e-13)))) for i in range(len(Y))]
        W = [Stdsi_YN[i]@W_cor[i]@Stdsi_YN[i] for i in range(len(Y))]
        for i in range(len(Y)):
            print("Max L cor", abs(W_cor[i][W_cor[i]!=0]).max(),"Min L", abs(W_cor[i][W_cor[i]!=0]).min())
            print("Max L", abs(W[i][W[i]!=0]).max(),"Min L", abs(W[i][W[i]!=0]).min())

        Nmuw=[(N[i]*np.sqrt(tausN[i])).T for i in range(len(N))]
        
        # d_t is a scalar variance
        d_t = [(1/Nmuw[i].shape[0]*Nmuw[i].shape[1])*np.sum(Nmuw[i]**2) for i in range(len(N))]
        
        # D_t is the inverse diagonal of d_t*I
        D = [np.diag(np.array([1/d_t[i]]*Nmuw[i].shape[1])) for i in range(len(N))]
        
    return W, D, tausY, tausN
def em_tlasso_noise2(Y, N, T, l,r, W0, D0, glasso=False):
    p = Y[0].shape[0]
    nu = 3
    k = Y[0].shape[1]
    nk = N[0].shape[1]
    muY = [ np.array([0]*p) for i in range(len(Y))]
    muN = [ np.array([0]*p) for i in range(len(N))]
    
    W = W0.copy()
    D = D0.copy()
    for j in range(T):
        print(f"Iteration: {j}")
        # E-step 
        tausY = tau(Y,3,p,muY,W, group=True)
        tausN = tau(N,3,p,muN,D, group=True)
        # M-step
        Ymuw, Sigma_Y, Stdsi_Y = update_Sigma_mu(Y,tausY, mu=False, group=True)
        
        
        if glasso: W_cor = [run_glasso(np.corrcoef(Sigma_Y[i].T), l) for i in range(len(Y))]
        else: W_cor = run_jgl(Sigma_Y, l,r)
        for i in range(len(Y)): W_cor[i][abs(W_cor[i])<1e-6]=0 
        Stdsi_YT = [np.linalg.inv(Stdsi_Y[i]+1e-13) for i in range(len(Y))]
        Stdsi_YN = [np.diag(np.sqrt(1/Ymuw[i].std(0))) for i in range(len(Y))]
        W = [Stdsi_YN[i]@W_cor[i]@Stdsi_YN[i] for i in range(len(Y))]
        for i in range(len(Y)):
            print(np.diag(Stdsi_YN[i]).max())
            print("Max L cor", abs(W_cor[i][W_cor[i]!=0]).max(),"Min L", abs(W_cor[i][W_cor[i]!=0]).min())
            print("Max L", abs(W[i][W[i]!=0]).max(),"Min L", abs(W[i][W[i]!=0]).min())

        Nmuw=[(N[i]*np.sqrt(tausN[i])).T for i in range(len(N))]
        
        # d_t is a scalar variance
        d_t = [(1/Nmuw[i].shape[0]*Nmuw[i].shape[1])*np.sum(Nmuw[i]**2) for i in range(len(N))]
        
        # D_t is the inverse diagonal of d_t*I
        D = [np.diag(np.array([1/d_t[i]]*Nmuw[i].shape[1])) for i in range(len(N))]
        
    return W, D, tausY, tausN

def em_tlasso_noise3(Y, N, T, l,r, W0, D0, glasso=False):
    p = Y[0].shape[0]
    nu = 3
    k = Y[0].shape[1]
    nk = N[0].shape[1]
    muY = [ np.array([0]*p) for i in range(len(Y))]
    muN = [ np.array([0]*p) for i in range(len(N))]
    
    W = W0
    D = D0
    for j in range(T):
        print(f"Iteration: {j}")
        # E-step 
        tausY = tau(Y,3,p,muY,W, group=True)
        tausN = tau(N,3,p,muN,D, group=True)
        # M-step
        print("M-step")
        Ymuw, Sigma_Y, Stdsi_Y = update_Sigma_mu(Y,tausY, mu=False, group=True)
        
        
        
        #if glasso: 
        Ymuw = np.vstack(Ymuw)
        print(l*abs(np.cov(Ymuw.T)).max())
        #W = run_glasso(Ymuw,l*abs(np.cov(Ymuw.T)).max())
        #W=[W,W]
        W_cor =  run_glasso(np.corrcoef(Ymuw.T),l)
        D = np.diag(1/np.sqrt(Ymuw.std(0)))
        
        #print(D)
        #Stdsi_YT = run_glasso(np.cov(Ymuw.T), np.cov(Ymuw.T).max())
        #D = np.diag(np.sqrt( np.diag(Stdsi_YT)*(1/(np.diag(W_cor)+1e-13))))
        W=[D@W_cor@D,D@W_cor@D]
        #print(D.shape, W_cor.shape)
        #print(D)
       
        # else: 
        #     Ymuw = np.vstack(Ymuw)
        #     #print(l*min(max(Stdsi_Y[0]),max(Stdsi_Y[1])))
        #     W = run_glasso(Ymuw, l*abs(np.cov(Ymuw)).max())#run_jgl(Ymuw, l*min((abs(Stdsi_Y[0]).max()),(abs(Stdsi_Y[1]).max())),r)
        #     W=[W,W]
            #W_cor = run_jgl(Sigma_Y, l,r)
            
        #for i in range(len(Y)): W_cor[i][abs(W_cor[i])<1e-6]=0 
        #Stdsi_YT = [np.linalg.inv(Stdsi_Y[i]+1e-13) for i in range(len(Y))]
        
        #Stdsi_YN = [np.diag(np.sqrt(1/Ymuw[i].std(0))) for i in range(len(Y))]
        #[Stdsi_YN[i]@W_cor[i]@Stdsi_YN[i] for i in range(len(Y))]
        for i in range(len(Y)):
            #print("Max L cor", abs(W_cor[i][W_cor[i]!=0]).max(),"Min L", abs(W_cor[i][W_cor[i]!=0]).min())
            print("Max L", abs(W[i][W[i]!=0]).max(),"Min L", abs(W[i][W[i]!=0]).min())

        Nmuw=[(N[i]*np.sqrt(tausN[i])).T for i in range(len(N))]
        
        # d_t is a scalar variance
        d_t = [(1/Nmuw[i].shape[0]*Nmuw[i].shape[1])*np.sum(Nmuw[i]**2) for i in range(len(N))]
        
        # D_t is the inverse diagonal of d_t*I
        D = [np.diag(np.array([1/d_t[i]]*Nmuw[i].shape[1])) for i in range(len(N))]
        
    return W, D, tausY, tausN

def em_gtlasso(Y, T, l,W0,r):
    p = Y[0].shape[0]
    nu = 3
    mu = [np.mean(Y[i], axis = 1) for i in range(len(Y))]

    
    W = W0.copy()
    for i in range(T):
        print(f"Iteration: {i}")
        # E-step 
        taus = tau(Y,nu,p,mu,W, group=True)
        # M-step
        mu_u, Sigma_u, Stdsi = update_Sigma_mu(Y,taus, group=True)
        W_cor = run_jgl(Sigma_u, l,r)
        W = [(Stdsi[i])@W_cor[i]@(Stdsi[i]) for i in range(len(Y))]
        mu = mu_u 
        
    return W, mu


#em_tlasso_noise_no_noise

def em_tlasso_noise_no_noise(Y, T, l,r, W0, glasso=False):
    p = Y[0].shape[0]
    nu = 3
    k = Y[0].shape[1]
    muY = [ np.array([0]*p) for i in range(len(Y))]
    
    W = W0
    for j in range(T):
        print(f"Iteration: {j}")
        # E-step 
        tausY = tau(Y,3,p,muY,W, group=True)
        # M-step
        print("M-step")
        Ymuw, Sigma_Y, Stdsi_Y = update_Sigma_mu(Y,tausY, mu=False, group=True)
        
        
        
        #if glasso: 
        Ymuw = np.vstack(Ymuw)

        W_cor =  run_glasso(np.corrcoef(Ymuw.T),l)
        D = np.diag(1/np.sqrt(Ymuw.std(0)))
        
        W=[D@W_cor@D,D@W_cor@D]
        for i in range(len(Y)):
            print("Max L", abs(W[i][W[i]!=0]).max(),"Min L", abs(W[i][W[i]!=0]).min())

                
        
    return W, tausY


def em_tlasso_noise_with_mu(Y, N, T, l,r, W0, D0, glasso=False,with_mu=True, corr=True):
    p = Y[0].shape[0]
    nu = 3
    k = Y[0].shape[1]
    nk = N[0].shape[1]
    muY = [ np.array([0]*p) for i in range(len(Y))]
    muN = [ np.array([0]*p) for i in range(len(N))]
    
    W = W0
    D = D0
    is_converged = False
    for j in range(T):
        print(f"Iteration: {j}")
        # E-step 
        tausY = tau(Y,3,p,muY,W0, group=True)
        tausN = tau(N,3,p,muN,D0, group=True)
        # M-step
        print("M-step")
        if with_mu:
            muY = update_mu(Y, tausY)
            #muN = update_mu(N, tausN)
        Ymuw, Sigma_Y, Stdsi_Y = update_Sigma_mu(Y, tausY, mu=with_mu, group=True)
        Nmuw, _, _ = update_Sigma_mu(N, tausN, mu=with_mu, group=True)

        Ymuw = np.vstack(Ymuw)
        print(abs(np.cov(Ymuw.T)).max())
        n = Ymuw.shape[0]
        print(n)
        Ymuwcov =(1/(n-1))*Ymuw.T@Ymuw #np.cov(Ymuw.T)
        D = np.sqrt(np.diag(Ymuwcov))
        Dinv = np.linalg.inv(np.diag(D))
        if corr:
            W = run_glasso(Dinv@Ymuwcov@Dinv , l)
            W = 0.5*(W+W.T)
            if np.linalg.eigvals(W).min()<0:
                print("Detected a negative eigenvalue")
                W = nearestPD(W)
            W = [Dinv@W@Dinv , Dinv@W@Dinv ]
        else:
            W =  run_glasso(Ymuwcov,l)
            W = 0.5 * (W + W.T)
            if np.linalg.eigvals(W).min()<0:
                print("Detected a negative eigenvalue")
                W = nearestPD(W)
            W = [W,W]
        print("Max eig", np.linalg.eigvals(W[0].copy() -W0[0].copy()).max())

        for i in range(len(Y)):
            print("Max L", abs(W[i][W[i]!=0]).max(),"Min L", abs(W[i][W[i]!=0]).min())


        # d_t is a scalar variance
        d_t = [(1/Nmuw[i].shape[0]*Nmuw[i].shape[1])*np.sum(Nmuw[i]**2) for i in range(len(N))]
        
        # D_t is the inverse diagonal of d_t*I
        D = [np.diag(np.array([1/d_t[i]]*Nmuw[i].shape[1])) for i in range(len(N))]
        D0 = D
        if abs(np.linalg.eigvals(W[0].copy() - W0[0].copy()).max())<=1e-4:
            is_converged=True
            break
        # if np.linalg.eigvals(W[0]).min() < 0:
        #     print("Detected a negative eigenvalue")
        #     W[0] = nearestPD(W[0])
        W0 = [W[0].copy()]
        
    return W, D, tausY, tausN,muY,muN, is_converged


def em_tlasso_no_noise_with_mu(Y, T, l, W0, with_mu=True, corr=True):
    p = Y[0].shape[0]

    muY = [np.array([0] * p) for i in range(len(Y))]
    is_converged = False
    W = W0
    for j in range(T):
        print(f"Iteration: {j}")
        # E-step
        tausY = tau(Y, 3, p, muY, W, group=True)
        # M-step
        print("M-step")
        if with_mu:
            muY = update_mu(Y, tausY)
            # muN = update_mu(N, tausN)
        Ymuw, Sigma_Y, Stdsi_Y = update_Sigma_mu(Y, tausY, mu=with_mu, group=True)

        Ymuw = np.vstack(Ymuw)
        print(abs(np.cov(Ymuw.T)).max())
        n = Ymuw.shape[0]
        print(n)
        Ymuwcov = (1 / (n - 1)) * Ymuw.T @ Ymuw  # np.cov(Ymuw.T)
        D = np.sqrt(np.diag(Ymuwcov))
        Dinv = np.linalg.inv(np.diag(D))

        if corr:
            W = run_glasso(Dinv@Ymuwcov@Dinv , l)
            W = 0.5*(W+W.T)
            if np.linalg.eigvals(W).min()<0:
                print("Detected a negative eigenvalue")
                W = nearestPD(W)
            W = [Dinv@W@Dinv , Dinv@W@Dinv ]
        else:
            W =  run_glasso(Ymuwcov,l)
            W = 0.5 * (W + W.T)
            if np.linalg.eigvals(W).min()<0:
                print("Detected a negative eigenvalue")
                W = nearestPD(W)
            W = [W,W]
        # D = np.diag(1/Ymuw.std(0)) # sqrt was added, why?

        # W=[D@W_cor@D,D@W_cor@D]
        for i in range(len(Y)):
            print("Max L", abs(W[i][W[i] != 0]).max(), "Min L", abs(W[i][W[i] != 0]).min())
        print("Max eig", np.linalg.eigvals(W[0].copy() -W0[0].copy()).max())

        if abs(np.linalg.eigvals(W[0].copy() - W0[0].copy()).max())<=1e-4:
            is_converged=True
            break
        W0 = [W[0].copy()]

        # Nmuw=[(N[i]*np.sqrt(tausN[i])).T for i in range(len(N))]

        # d_t is a scalar variance

        # D_t is the inverse diagonal of d_t*I

    return W, tausY, muY, is_converged


# def em_tlasso_noise_with_mu(Y, N, T, l, r, W0, D0, glasso=False, with_mu=True):
#     p = Y[0].shape[0]
#     nu = 3
#     k = Y[0].shape[1]
#     nk = N[0].shape[1]
#     muY = [np.array([0] * p) for i in range(len(Y))]
#     muN = [np.array([0] * p) for i in range(len(N))]
#
#     W = W0
#     D = D0
#     for j in range(T):
#         print(f"Iteration: {j}")
#         # E-step
#         tausY = tau(Y, 3, p, muY, W, group=True)
#         tausN = tau(N, 3, p, muN, D, group=True)
#         # M-step
#         print("M-step")
#         if with_mu:
#             muY = update_mu(Y, tausY)
#             muN = update_mu(N, tausN)
#         Ymuw, Sigma_Y, Stdsi_Y = update_Sigma_mu(Y, tausY, mu=with_mu, group=True)
#         Nmuw, _, _ = update_Sigma_mu(N, tausN, mu=with_mu, group=True)
#
#         Ymuw = np.vstack(Ymuw)
#         print(abs(np.cov(Ymuw.T)).max())
#         n = Ymuw.shape[0]
#         print(n)
#         Ymuwcov = (1 / (n - 1)) * Ymuw.T @ Ymuw  # np.cov(Ymuw.T)
#         D = np.sqrt(np.diag(Ymuwcov))
#         Dinv = np.linalg.inv(np.diag(D))
#         # W = run_glasso(Ymuwcov , l)
#         # W = [W , W ]
#         W = run_glasso(Dinv @ Ymuwcov @ Dinv, l)
#         W = [Dinv @ W @ Dinv, Dinv @ W @ Dinv]
#         # D = np.diag(1/Ymuw.std(0)) # sqrt was added, why?
#
#         # W=[D@W_cor@D,D@W_cor@D]
#         for i in range(len(Y)):
#             print("Max L", abs(W[i][W[i] != 0]).max(), "Min L", abs(W[i][W[i] != 0]).min())
#
#         # Nmuw=[(N[i]*np.sqrt(tausN[i])).T for i in range(len(N))]
#
#         # d_t is a scalar variance
#         d_t = [(1 / Nmuw[i].shape[0] * Nmuw[i].shape[1]) * np.sum(Nmuw[i] ** 2) for i in range(len(N))]
#
#         # D_t is the inverse diagonal of d_t*I
#         D = [np.diag(np.array([1 / d_t[i]] * Nmuw[i].shape[1])) for i in range(len(N))]
#
#     return W, D, tausY, tausN, muY, muN


def em_tlasso_noise_jgl(Y, N, T, l, r, W0, D0, glasso=False):
    p = Y[0].shape[0]
    nu = 3
    muY = [np.array([0] * p) for _ in range(len(Y))]
    muN = [np.array([0] * p) for _ in range(len(N))]

    W = W0
    D = D0
    for j in range(T):
        print(f"Iteration: {j}")
        # E-step
        tausY = tau(Y, nu, p, muY, W, group=True)
        tausN = tau(N, nu, p, muN, D, group=True)
        # M-step
        print("M-step")
        muY = update_mu(Y, tausY)
        muN = update_mu(N, tausN)
        Ymuw, Sigma_Y, Stdsi_Y = update_Sigma_mu(Y, tausY, mu=True, group=True)
        Nmuw, _, _ = update_Sigma_mu(N, tausN, mu=True, group=True)

        #Ymuw = np.vstack(Ymuw)
        print((Ymuw[0].std(0)).shape[0])
        #D = [np.diag(1 / Ymuw[i].std(0)) for i in range(len(N))]
        n= [Ymuw[i].shape[0] for i in range(len(N))]
        Ymuwcov =[(1/(n[i]-1))*Ymuw[i].T@Ymuw[i] for i in range(len(N))]
        D = [np.sqrt(np.diag(Ymuwcov[i])) for i in range(len(N))]
        Dinv = [np.linalg.inv(np.diag(D[i])) for i in range(len(N))]

        Ymuw_std = [Ymuw[i] for i in range(len(N))]
        print(len(Ymuw_std))
        print(Ymuw_std[0].shape, Ymuw_std[1].shape)
        W = run_jgl(Ymuw_std, l, 0.1)


       # W = [Dinv[i] @ W_cor[i] @ Dinv[i] for i in range(len(N))]
        for i in range(len(Y)):
            print("Max L", abs(W[i][W[i] != 0]).max(), "Min L", abs(W[i][W[i] != 0]).min())

        #Nmuw = [(N[i] * np.sqrt(tausN[i])).T for i in range(len(N))]

        # d_t is a scalar variance
        d_t = [(1 / Nmuw[i].shape[0] * Nmuw[i].shape[1]) * np.sum(Nmuw[i] ** 2) for i in range(len(N))]

        # D_t is the inverse diagonal of d_t*I
        D = [np.diag(np.array([1 / d_t[i]] * Nmuw[i].shape[1])) for i in range(len(N))]

    return W, D, tausY, tausN, muY,muN