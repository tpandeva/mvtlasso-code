import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import pickle
from sklearn.decomposition import FastICA, PCA

import torch.nn as nn
import geotorch
from model._tlasso import em_tlasso_noise_with_mu
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import coo_matrix

def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = np.random.chisquare(df, n) / df
    z = np.random.multivariate_normal(np.zeros(d), S, (n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal







def number_ICA_components(dataset, N_perm=50):
    k = np.linalg.matrix_rank(dataset) 

    power_perm = np.empty((N_perm, k))
    power_perm[:] = np.nan

    ica = FastICA(n_components=k, fun='logcosh')
    ICA_true = ica.fit(dataset).mixing_
    power_true = np.sum(ICA_true ** 2, axis=0)
    power_true = power_true[np.argsort(-power_true)]
    power_true /= np.sum(power_true)

    for k in range(N_perm):
        expr_perm = np.apply_along_axis(np.random.permutation, 1, dataset)
        ICA_perm = ica.fit(expr_perm).mixing_
        power_perm_k = np.sum(ICA_perm ** 2, axis=0)
        power_perm_k = power_perm_k[np.argsort(-power_perm_k)]
        power_perm_k /= np.sum(power_perm_k)
        power_perm[k, :] = power_perm_k
    pval = np.sum(np.transpose(power_perm) >= power_true.reshape((-1,1)), axis=1) / N_perm
    optPC = np.sum(pval <= 0.05)
    
    return optPC

def whiten(X,n_components):
    D, n_features, n_samples = X.shape
    XT = np.zeros((D,n_components,n_samples))

    K = np.zeros((D,n_components,n_features))
    Kinv = np.zeros((D,n_features,n_components))
    for i in range(len(X)):
        X_mean = X[i].mean(axis=-1)
        X[i] -= X_mean[:, np.newaxis]

        u, d, _ = randomized_svd(X[i], n_components=n_components)

        del _
        Ki = ((u/ d).T)*np.sqrt(n_samples/2)
        Xw = np.dot(Ki, X[i])
        
        XT[i] = Xw
        K[i] = Ki
        Kinv[i]= (u * d)/np.sqrt(n_samples/2)


    return K,Kinv, XT



def fit_ica_tlasso(Xs, ks, device, T, l,r, to_whiten, seed=1, params=None, scaling=True):
    np.random.seed(seed)
    num_views = 1

    L_t = [np.eye(Xs[i].shape[0]) for i in range(num_views)]
    D_t = [np.eye(Xs[i].shape[0]) for i in range(num_views)]
    
    L_t, D_t, tauZ, tauN, muY, muN = em_tlasso_noise_with_mu(Xs, Xs, T, l,r, L_t, D_t,with_mu=True)

    return L_t

@hydra.main(config_path="config/", config_name="config.yaml")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    # ray.shutdown()
    # runtime_envs = {"working_dir": "/var/scratch/tpandeva/siica/."}
    # ray.init(runtime_env = runtime_envs)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    s = cfg.seed
    path = cfg.file
    data_file = cfg.data_file
    store_as_alias = cfg.alias
    to_whiten= cfg.whiten
    device = torch.device('cpu')
    T = cfg.T
    l=cfg.l
    r=cfg.r
    reduction = cfg.reduction
    # 50 sources Sigma, 50 sources D, 100 genes
    n = 100
    d1 = cfg.d1
    d2 = cfg.d2
    Theta = np.eye(n)
    size = int(0.5 * n * (n - 1))

    D = 0.01 * np.eye(n)
    Theta[np.triu_indices(n, k=1)] = np.random.choice([-1, 0, 1], size=size, p=[0.01, 0.98, 0.01])
    Theta += Theta.T
    adj = Theta.copy()
    adj[adj != 0] = 1
    print(adj.sum() - n)
    pickle.dump(adj, open(os.path.join(script_dir,"data/adj_simulated.pickle"),"wb"))
    np.fill_diagonal(Theta, (adj.sum(0)))
    Sigma = np.linalg.inv(Theta)
    #Xs = pickle.load(open(os.path.join(script_dir,"data/simulated.pickle"),"rb"))
    ks = cfg.k
    # for X in Xs:
    #
    #     # dff = os.path.join(script_dir, df)
    #     # X = pd.read_csv(dff, sep=",")
    #     # X = X.drop(X.columns[0], axis=1)
    #
    #     if reduction:
    #         if cfg.k is None:
    #             ks.append(number_ICA_components(X))
    #         else:
    #             ks = cfg.k
    #         params = None
    #     else:
    #         ks.append(X.shape[0])
    #         params = cfg.k
    #
        # X = X.to_numpy() # maybe .T
        # Xs.append(X)
    params = cfg.k
    adj_em = [0,0]
    adj = pickle.load(open(os.path.join(script_dir,"data/adj_simulated.pickle"),"rb"))


    for s in range(cfg.start,cfg.end):
        S = multivariate_t_rvs(np.array([0] * n), Sigma, df=3, n=d1).T
        Z = multivariate_t_rvs(np.array([0] * n), D, df=3, n=d2).T
        Y1 = np.concatenate((S[:, :d1 // 2], Z[:, :d2 // 2]), axis=1)
        #Y2 = np.concatenate((S[:, d1 // 2:], Z[:, d2 // 2:]), axis=1)
        A1 = np.random.normal(loc=1, scale=0.1, size=((d1 + d2) // 2) ** 2).reshape(((d1 + d2) // 2, (d1 + d2) // 2))
        #A2 = np.random.normal(loc=1, scale=1.0, size=((d1 + d2) // 2) ** 2).reshape(((d1 + d2) // 2, (d1 + d2) // 2))

        # A[d1:,:] = 0.001*A[d1:,:]
        X1 = Y1 @ A1
        if d2 == 0:
            X1 = S[:, :d1 // 2] @ A1
        #X2 = Y2 @ A2
        Xs = [X1]

        Wi=fit_ica_tlasso(Xs, ks, device, T, l, r, to_whiten, s, params, cfg.scaling)


        adj_Wi = [abs(Wi[i].copy()) for i in range(2)]
        for i in range(2):
            adj_Wi[i][adj_Wi[i]!=0]=1
            print("Edges:", 0.5*(adj_Wi[i].sum()-adj_Wi[i].shape[0]),
                  "TPR:",((adj*adj_Wi[i]).sum()-adj_Wi[i].shape[0])/(adj.sum()-adj_Wi[i].shape[0]),
                  "FPR:",((adj_Wi[i]-adj_Wi[i]*adj).sum())/((1-adj).sum()))
            adj_Wi[i] = coo_matrix(adj_Wi[i])
        adj_em= [adj_em[i]+adj_Wi[i] for i in range(2)]
        del Wi, adj_Wi

        adj_em_eval = [adj_em[i].toarray().copy() for i in range(2)]

        for i in range(2):
            adj_em_eval[i][adj_em_eval[i]<adj_em_eval[i][0,0]/2] = 0
            adj_em_eval[i][adj_em_eval[i]!=0]=1
            print("Edges:", 0.5*(adj_em_eval[i].sum()-adj_em_eval[i].shape[0]), "TP:",0.5*((adj*adj_em_eval[i]).sum()-adj_em_eval[i].shape[0]) )








        open_file = open(os.path.join(script_dir,f"{path}/res_{store_as_alias}_{l}_{cfg.start}_{cfg.end}.pickle"), "wb")
        pickle.dump( [adj_em,adj], open_file)
        open_file.close()


if __name__ == '__main__':
    main()