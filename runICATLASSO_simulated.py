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
from model._tlasso import em_tlasso_no_noise_with_mu,em_tlasso_noise_with_mu
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


class TICA(nn.Module):
    def __init__(self, n_in, n_out, scaling=True):
        super().__init__()
        if scaling:
            self.W = nn.Linear(n_in, n_out, bias=False)
            self.mask = torch.eye(n_in, dtype=bool)
        self.Q = nn.Linear(n_in, n_out, bias=False)
        #geotorch.positive_definite(self.W, "weight")
        
        self.scaling=scaling
        geotorch.orthogonal(self.Q, "weight")
        #self.W.weight = self.W.weight.clone().contiguous()


    def forward(self, Xw):
        if self.scaling:
            self.W.weight.data *= self.mask
            S = self.Q(self.W(Xw))
        else: S = self.Q(Xw)
        return S


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


def fit_ica_tlasso(Xs, ks, device, T, l, r, to_whiten, seed=1, params=None, scaling=True):
    np.random.seed(seed)
    num_views = len(ks)

    L_t, D_t, Tau, adj_Wo = None, None, None, None

    Xws, Wts = [], []
    Ik, Ink = [], []
    models = []
    param_list = []
    for i in range(num_views):
        X = Xs[i]
        adj = pickle.load(open(os.path.join(script_dir, "data/adj_simulated.pickle"), "rb"))
        n = X.shape[1]
        # X = X[:,np.random.choice(n, size=int(n*0.95), replace=False)]

        ica = FastICA(n_components=X.shape[0])
        Xica = ica.fit_transform(X)

        print(Xica.shape)
        Xws.append(Xica[:, np.argsort(-(ica.mixing_ ** 2).sum(0))])
        Ik.append(np.diag(np.array([1] * params[i] + [0] * (n - params[i]))))
        Ink.append(np.diag(np.array([0] * params[i] + [1] * (n - params[i]))))
        models.append(TICA(Xica.shape[1], Xica.shape[1], scaling).to(device))
        param_list += list(models[i].parameters())

    optim = torch.optim.LBFGS(param_list, line_search_fn="strong_wolfe", max_iter=5000)

    Wts = [np.eye(Xws[i].shape[1]) for i in range(num_views)]
    Q = None

    for t in range(T):

        if L_t is None:

            S_all = [Xws[i][:, :params[i]] for i in range(num_views)]
            # S_all = np.hstack(S_all)
            Z_all = [Xws[i][:,params[i]:] for i in range(num_views)]
            L_t = [np.eye(Xws[i].shape[0]) for i in range(num_views)]

            # d_t is a scalar variance
            d_t = [(1/Z_all[i].shape[0]*Z_all[i].shape[1])*np.sum(Z_all[i]**2) for i in range(num_views)]
            # D_t is the inverse diagonal of d_t*I
            D_t = [np.diag(np.array([1/d_t[i]]*Z_all[i].shape[0])) for i in range(num_views)]
            # D_t = [0.2*np.eye(Z_all[i].shape[0]) for i in range(num_views)]
            if adj_Wo is None:
                adj_Wo = [abs(L_t[i].copy()) for i in range(num_views)]
            for i in range(num_views): adj_Wo[i][adj_Wo[i] != 0] = 1
            adj_Wi = [abs(L_t[i].copy()) for i in range(num_views)]
            for i in range(num_views):
                print(adj_Wi[i][0, 0])
                adj_Wi[i][adj_Wi[i] != 0] = 1
            for i in range(num_views):
                print("Old Edges:", 0.5 * (adj_Wo[i].sum() - adj_Wi[i].shape[0]), "Edges:",
                      0.5 * (adj_Wi[i].sum() - adj_Wi[i].shape[0]), "Intersection:",
                      0.5 * ((adj_Wo[i] * adj_Wi[i]).sum() - adj_Wi[i].shape[0]), "TP:",
                      0.5 * ((adj * adj_Wi[i]).sum() - adj_Wi[i].shape[0]))
            # print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
            adj_Wo = adj_Wi
            # L_t, D_t, tauZ, tauN, muY, muN
            L_t, D_t, tauZ, tauN, muY, muN = em_tlasso_noise_with_mu(S_all, Z_all, 1, l,r, L_t, D_t, glasso=True, with_mu=True)
            muY_tensor = [torch.from_numpy(muY[i]).float().to(device) for i in range(num_views)]
            muN_tensor = [torch.from_numpy(muN[i]).float().to(device) for i in range(num_views)]

            print("shape of mu", muY_tensor[0].shape)
            adj_Wi = [abs(L_t[i].copy()) for i in range(num_views)]
            for i in range(num_views): adj_Wi[i][adj_Wi[i] != 0] = 1
            for i in range(num_views):
                print("Old Edges:", 0.5 * (adj_Wo[i].sum() - adj_Wi[i].shape[0]), "Edges:",
                      0.5 * (adj_Wi[i].sum() - adj_Wi[i].shape[0]), "Intersection:",
                      0.5 * ((adj_Wo[i] * adj_Wi[i]).sum() - adj_Wi[i].shape[0]), "TP:",
                      0.5 * ((adj * adj_Wi[i]).sum() - adj_Wi[i].shape[0]))
            # print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
            adj_Wo = adj_Wi

            Tau = [np.diag(np.concatenate((tauZ[i],tauN[i]))) for i in range(num_views)]
            #Tau = [np.diag(tauY[i]) for i in range(num_views)]

        for i in range(num_views):

            for _ in range(2):

                D1 = np.sqrt(Ik[i])
                D2 = np.sqrt(Ink[i])
                if Tau is not None:
                    D1 = np.sqrt(Ik[i] @ Tau[i])
                    D2 = np.sqrt(Ink[i]@Tau[i])

                Z = torch.from_numpy(Xws[i]).float().to(device)

                D1_tensor = torch.from_numpy(D1).float().to(device)
                D2_tensor =  torch.from_numpy(D2).float().to(device)
                L_t_tensor = torch.from_numpy(L_t[i]).float().to(device)
                D_t_tensor =  torch.from_numpy(D_t[i]).float().to(device)

                def loss_closure():
                    optim.zero_grad()

                    ZQ = models[i](Z)
                    ZQm = ZQ - muY_tensor[i][:, None]
                    ZQn = ZQ - muN_tensor[i][:, None]
                    loss = torch.trace(D1_tensor @ (ZQm.T @ (
                                L_t_tensor @ ZQm)) @ D1_tensor) +torch.trace(D2_tensor@(ZQn.T@(D_t_tensor@ZQn))@D2_tensor)#
                    if scaling: loss -= torch.linalg.slogdet(models[i].W.weight * torch.eye(Z.shape[1]))[1]

                    loss.backward()
                    return loss

                optim.step(loss_closure)

        Z = [torch.from_numpy(Xws[i] @ Wts[i]).float().to(device) for i in range(num_views)]
        Z_np = [models[i](Z[i]).detach().cpu().numpy()[:, :params[i]] for i in range(num_views)]
        # Z_np = Xws
        N_np = [models[i](Z[i]).detach().cpu().numpy()[:, params[i]:] for i in range(num_views)]
        # for i in range(num_views):
        #     print(models[i].W.weight)
        #     print(models[i].Q.weight)
        #     print(Z_np[i].std(0))
        #

        L_t, D_t, tauZ, tauN, muY, muN = em_tlasso_noise_with_mu(Z_np,N_np, 1, l,r, L_t,D_t, with_mu=True)

        muY_tensor = [torch.from_numpy(muY[i]).float().to(device) for i in range(num_views)]
        muN_tensor = [torch.from_numpy(muN[i]).float().to(device) for i in range(num_views)]

        adj_Wi = [abs(L_t[i].copy()) for i in range(num_views)]
        for i in range(num_views): adj_Wi[i][adj_Wi[i] != 0] = 1
        for i in range(num_views):
            print("Old Edges:", 0.5 * (adj_Wo[i].sum() - adj_Wi[i].shape[0]), "Edges:",
                  0.5 * (adj_Wi[i].sum() - adj_Wi[i].shape[0]), "Intersection:",
                  0.5 * ((adj_Wo[i] * adj_Wi[i]).sum() - adj_Wi[i].shape[0]), "TP:",
                  0.5 * ((adj * adj_Wi[i]).sum() - adj_Wi[i].shape[0]))
        # print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
        adj_Wo = adj_Wi

        Tau = [np.diag(np.concatenate((tauZ[i],tauN[i]))) for i in range(num_views)]
        #Tau = [np.diag(tauY[i]) for i in range(num_views)]

    return L_t


def fit_ica_tlasso_no_noise(Xs, ks, device, T, l,r, to_whiten, seed=1, params=None, scaling=True):
    np.random.seed(seed)
    num_views = len(ks)

    
    L_t, D_t, Tau, adj_Wo = None, None, None, None
    

    
    Xws, Wts = [], []
    Ik, Ink= [],[]
    models = []
    param_list = []
    for i in range(num_views):
        X = Xs[i]
        adj = pickle.load(open(os.path.join(script_dir,"data/adj_simulated.pickle"),"rb"))
        n = X.shape[1]
        #X = X[:,np.random.choice(n, size=int(n*0.95), replace=False)]

        ica = FastICA(n_components=X.shape[0])
        Xica = ica.fit_transform(X)
       
        print(Xica.shape)
        Xws.append(Xica[:,np.argsort(-(ica.mixing_**2).sum(0))])
        Ik.append(np.diag(np.array([1]*params[i]+[0]*(n-params[i]))))
        Ink.append(np.diag(np.array([0]*params[i]+[1]*(n-params[i]))))
        models.append(TICA(Xica.shape[1], Xica.shape[1], scaling).to(device))
        param_list+=list(models[i].parameters())
        
    
    optim = torch.optim.LBFGS(param_list, line_search_fn="strong_wolfe",max_iter=5000) 


    Wts = [np.eye(Xws[i].shape[1]) for i in range(num_views)]
    Q=None
   
    for t in range(T):   
        
        if L_t is None: 
            
            S_all = [Xws[i][:,:params[i]] for i in range(num_views)]
            #S_all = np.hstack(S_all)
           # Z_all = [Xws[i][:,params[i]:] for i in range(num_views)]
            L_t = [np.eye(Xws[i].shape[0]) for i in range(num_views)]

            # d_t is a scalar variance
            #d_t = [(1/Z_all[i].shape[0]*Z_all[i].shape[1])*np.sum(Z_all[i]**2) for i in range(num_views)]
            # D_t is the inverse diagonal of d_t*I
            #D_t = [np.diag(np.array([1/d_t[i]]*Z_all[i].shape[0])) for i in range(num_views)]
            #D_t = [0.2*np.eye(Z_all[i].shape[0]) for i in range(num_views)]
            if adj_Wo is None: 
                adj_Wo = [ abs(L_t[i].copy()) for i in range(num_views)]
            for i in range(num_views): adj_Wo[i][adj_Wo[i]!=0]=1
            adj_Wi = [ abs(L_t[i].copy()) for i in range(num_views)]
            for i in range(num_views):
                print(adj_Wi[i][0,0])
                adj_Wi[i][adj_Wi[i]!=0]=1
            for i in range(num_views):
                print("Old Edges:", 0.5*(adj_Wo[i].sum()-adj_Wi[i].shape[0]),"Edges:", 0.5*(adj_Wi[i].sum()-adj_Wi[i].shape[0]), "Intersection:",0.5*((adj_Wo[i]*adj_Wi[i]).sum()-adj_Wi[i].shape[0]),"TP:",0.5*((adj*adj_Wi[i]).sum()-adj_Wi[i].shape[0]) )
            #print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
            adj_Wo=adj_Wi
            # L_t, D_t, tauZ, tauN, muY, muN
            L_t, tauY, muY = em_tlasso_no_noise_with_mu(S_all,  1, l, L_t, with_mu=True) #L_t, D_t, glasso=True, with_mu=True
            muY_tensor = [torch.from_numpy(muY[i]).float().to(device) for i in range(num_views)]
           # muN_tensor = [torch.from_numpy(muN[i]).float().to(device) for i in range(num_views)]

            print("shape of mu", muY_tensor[0].shape)
            adj_Wi = [ abs(L_t[i].copy()) for i in range(num_views)]
            for i in range(num_views): adj_Wi[i][adj_Wi[i]!=0]=1
            for i in range(num_views):
                print("Old Edges:", 0.5*(adj_Wo[i].sum()-adj_Wi[i].shape[0]),"Edges:", 0.5*(adj_Wi[i].sum()-adj_Wi[i].shape[0]), "Intersection:",0.5*((adj_Wo[i]*adj_Wi[i]).sum()-adj_Wi[i].shape[0]),"TP:",0.5*((adj*adj_Wi[i]).sum()-adj_Wi[i].shape[0]) )
            #print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
            adj_Wo=adj_Wi

            #Tau = [np.diag(np.concatenate((tauZ[i],tauN[i]))) for i in range(num_views)]
            Tau = [np.diag(tauY[i]) for i in range(num_views)]

        
        for i in range(num_views): 
            
            for _ in range(2):


                D1 = np.sqrt(Ik[i])
                #D2 = np.sqrt(Ink[i])
                if Tau is not None:
                    D1 = np.sqrt(Ik[i]@Tau[i])
                    #D2 = np.sqrt(Ink[i]@Tau[i])
                
                Z =  torch.from_numpy(Xws[i]).float().to(device)

                D1_tensor =  torch.from_numpy(D1).float().to(device)
                #D2_tensor =  torch.from_numpy(D2).float().to(device)
                L_t_tensor =  torch.from_numpy(L_t[i]).float().to(device)
                #D_t_tensor =  torch.from_numpy(D_t[i]).float().to(device)

                def loss_closure():
                    optim.zero_grad()

                    ZQ = models[i](Z)
                    ZQm = ZQ-muY_tensor[i][:,None]
                    #ZQn = ZQ - muN_tensor[i][:, None]
                    loss = torch.trace(D1_tensor@(ZQm.T@(L_t_tensor@ZQm))@D1_tensor)#+torch.trace(D2_tensor@(ZQn.T@(D_t_tensor@ZQn))@D2_tensor)#
                    if scaling: loss-=torch.linalg.slogdet(models[i].W.weight*torch.eye(Z.shape[1]))[1]


                    loss.backward()
                    return loss

                optim.step(loss_closure)
                
            
            

        Z =  [torch.from_numpy(Xws[i]@Wts[i]).float().to(device) for i in range(num_views)]
        Z_np = [models[i](Z[i]).detach().cpu().numpy()[:,:params[i]] for i in range(num_views)]
        #Z_np = Xws
        N_np = [models[i](Z[i]).detach().cpu().numpy()[:,params[i]:] for i in range(num_views)]
        # for i in range(num_views):
        #     print(models[i].W.weight)
        #     print(models[i].Q.weight)
        #     print(Z_np[i].std(0))
        #
        
        L_t, tauY, muY = em_tlasso_no_noise_with_mu(Z_np, 1, l, L_t, with_mu=True)
         
        muY_tensor = [torch.from_numpy(muY[i]).float().to(device) for i in range(num_views)]
        #muN_tensor = [torch.from_numpy(muN[i]).float().to(device) for i in range(num_views)]

        adj_Wi = [ abs(L_t[i].copy()) for i in range(num_views)]
        for i in range(num_views): adj_Wi[i][adj_Wi[i]!=0]=1
        for i in range(num_views):
            print("Old Edges:", 0.5*(adj_Wo[i].sum()-adj_Wi[i].shape[0]),"Edges:", 0.5*(adj_Wi[i].sum()-adj_Wi[i].shape[0]), "Intersection:",0.5*((adj_Wo[i]*adj_Wi[i]).sum()-adj_Wi[i].shape[0]),"TP:",0.5*((adj*adj_Wi[i]).sum()-adj_Wi[i].shape[0]) )
        #print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
        adj_Wo=adj_Wi
        
        #Tau = [np.diag(np.concatenate((tauZ[i],tauN[i]))) for i in range(num_views)]
        Tau = [np.diag(tauY[i]) for i in range(num_views)]



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
        Y2 = np.concatenate((S[:, d1 // 2:], Z[:, d2 // 2:]), axis=1)
        A1 = np.random.normal(loc=1, scale=0.1, size=((d1 + d2) // 2) ** 2).reshape(((d1 + d2) // 2, (d1 + d2) // 2))
        A2 = np.random.normal(loc=1, scale=0.1, size=((d1 + d2) // 2) ** 2).reshape(((d1 + d2) // 2, (d1 + d2) // 2))
        # A[d1:,:] = 0.001*A[d1:,:]
        X1 = Y1 @ A1
        if d2 == 0:
            X1 = S[:, :d1 // 2] @ A1
       # X2 = Y2 @ A2
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