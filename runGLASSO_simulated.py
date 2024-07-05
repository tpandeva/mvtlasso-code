import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import pickle
from sklearn.decomposition import FastICA, PCA
from model._tlasso import run_glasso
from sklearn.utils.extmath import randomized_svd


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
    return m + z / np.sqrt(x)[:, None]  # same output format as random.multivariate_normal


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
    pval = np.sum(np.transpose(power_perm) >= power_true.reshape((-1, 1)), axis=1) / N_perm
    optPC = np.sum(pval <= 0.05)

    return optPC


def whiten(X, n_components):
    D, n_features, n_samples = X.shape
    XT = np.zeros((D, n_components, n_samples))

    K = np.zeros((D, n_components, n_features))
    Kinv = np.zeros((D, n_features, n_components))
    for i in range(len(X)):
        X_mean = X[i].mean(axis=-1)
        X[i] -= X_mean[:, np.newaxis]

        u, d, _ = randomized_svd(X[i], n_components=n_components)

        del _
        Ki = ((u / d).T) * np.sqrt(n_samples / 2)
        Xw = np.dot(Ki, X[i])

        XT[i] = Xw
        K[i] = Ki
        Kinv[i] = (u * d) / np.sqrt(n_samples / 2)

    return K, Kinv, XT


def fit_ica_tlasso(Xs, ks, device, T, l, r, to_whiten, seed=1, params=None, scaling=True, corr=True):
    np.random.seed(seed)
    num_views = 1


    Xg = np.hstack(Xs)
    L_t = run_glasso(np.cov(Xg),l)
    return [L_t,L_t]


@hydra.main(config_path="config/", config_name="config.yaml")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    s = cfg.seed
    path = cfg.file

    store_as_alias = cfg.alias
    to_whiten = cfg.whiten
    device = torch.device('cpu')
    T = cfg.T
    l = cfg.l
    r = cfg.r

    tprs, fprs = [], []
    for s in range(cfg.start, cfg.end):
        n = cfg.n
        d1 = cfg.d1
        d2 = cfg.d2
        Theta = np.eye(n)

        D = cfg.sigma * np.eye(n)
        ks = [cfg.d1 // 2]  # cfg.k

        params = [cfg.d1 // 2]  # cfg.k


        size = int(0.5 * n * (n - 1))
        Theta[np.triu_indices(n, k=1)] = np.random.choice([-1, 0, 1], size=size, p=[0.01, 0.98, 0.01])
        Theta += Theta.T
        adj = Theta.copy()
        adj[adj != 0] = 1
        print("True edges", adj.sum() - n)
        np.fill_diagonal(Theta, (adj.sum(0)))
        Sigma = np.linalg.inv(Theta)

        S = multivariate_t_rvs(np.array([0] * n), Sigma, df=3, n=d1).T
        Z = multivariate_t_rvs(np.array([0] * n), D, df=3, n=d2).T
        Y1 = np.concatenate((S[:, :d1 // 2], Z[:, :d2 // 2]), axis=1)
        A1 = np.random.normal(loc=1, scale=0.1, size=((d1 + d2) // 2) ** 2).reshape(((d1 + d2) // 2, (d1 + d2) // 2))

        X1 = Y1 @ A1
        if d2 == 0:
            X1 = S[:, :d1 // 2] @ A1

        Xs = [X1]

        Wi = fit_ica_tlasso(Xs, ks, device, T, l, r, to_whiten, s, params, cfg.scaling, cfg.corr)

        adj_Wi = [abs(Wi[i].copy()) for i in range(2)]
        for i in range(1):
            adj_Wi[i][adj_Wi[i] != 0] = 1
            print("Edges:", 0.5 * (adj_Wi[i].sum() - adj_Wi[i].shape[0]),
                  "TP:", 0.5 * ((adj * adj_Wi[i]).sum() - adj_Wi[i].shape[0]),
                  "TPR:", ((adj * adj_Wi[i]).sum() - adj_Wi[i].shape[0]) / (adj.sum() - adj_Wi[i].shape[0]),
                  "FPR:", ((adj_Wi[i] - adj_Wi[i] * adj).sum()) / ((1 - adj).sum()))
            tprs.append(((adj * adj_Wi[i]).sum() - adj_Wi[i].shape[0]) / (adj.sum() - adj_Wi[i].shape[0]))
            fprs.append(((adj_Wi[i] - adj_Wi[i] * adj).sum()) / ((1 - adj).sum()))
        del Wi, adj_Wi


        print(f"Mean TPR: {np.mean(np.array(tprs))}, Mean FPR: {np.mean(np.array(fprs))}")
        open_file = open(os.path.join(script_dir, f"{path}/res_{store_as_alias}_{l}_{cfg.start}_{cfg.end}.pickle"),
                         "wb")
        pickle.dump([tprs, fprs], open_file)
        open_file.close()


if __name__ == '__main__':
    main()