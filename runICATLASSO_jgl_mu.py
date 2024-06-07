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



def fit_ica_tlasso(Xs, ks, device, T, l,r, to_whiten, seed=1, params=None, scaling=True):
    np.random.seed(seed)
    num_views = len(ks)

    
    L_t, D_t, Tau, adj_Wo = None, None, None, None
    

    
    Xws, Wts = [], []
    Ik, Ink= [],[]
    models = []
    param_list = []
    for i in range(num_views):
        X = Xs[i]
        adj = pickle.load(open(os.path.join(script_dir,"data/adj_bs.pickle"),"rb"))
        n = X.shape[1]
        X = X[:,np.random.choice(n, size=int(n*0.9), replace=False)]
        n = X.shape[1]
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
            Z_all = [Xws[i][:,params[i]:] for i in range(num_views)]
            L_t = [np.eye(Xws[i].shape[0]) for i in range(num_views)]

            # d_t is a scalar variance
            d_t = [(1/Z_all[i].shape[0]*Z_all[i].shape[1])*np.sum(Z_all[i]**2) for i in range(num_views)]
            # D_t is the inverse diagonal of d_t*I
            D_t = [np.diag(np.array([1/d_t[i]]*Z_all[i].shape[0])) for i in range(num_views)]
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
            print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
            adj_Wo=adj_Wi

            L_t, D_t, tauZ, tauN, muY, muN = em_tlasso_noise_with_mu(S_all, Z_all, 1, l,r, L_t, D_t, glasso=True)
            muY_tensor = [torch.from_numpy(muY[i]).float().to(device) for i in range(num_views)]
            muN_tensor = [torch.from_numpy(muN[i]).float().to(device) for i in range(num_views)]

            print("shape of mu", muY_tensor[0].shape)
            adj_Wi = [ abs(L_t[i].copy()) for i in range(num_views)]
            for i in range(num_views): adj_Wi[i][adj_Wi[i]!=0]=1
            for i in range(num_views):
                print("Old Edges:", 0.5*(adj_Wo[i].sum()-adj_Wi[i].shape[0]),"Edges:", 0.5*(adj_Wi[i].sum()-adj_Wi[i].shape[0]), "Intersection:",0.5*((adj_Wo[i]*adj_Wi[i]).sum()-adj_Wi[i].shape[0]),"TP:",0.5*((adj*adj_Wi[i]).sum()-adj_Wi[i].shape[0]) )
            print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
            adj_Wo=adj_Wi

            Tau = [np.diag(np.concatenate((tauZ[i],tauN[i]))) for i in range(num_views)]

        
        for i in range(num_views): 
            
            for _ in range(2):


                D1 = np.sqrt(Ik[i])
                D2 = np.sqrt(Ink[i])
                if Tau is not None:
                    D1 = np.sqrt(Ik[i]@Tau[i])
                    D2 = np.sqrt(Ink[i]@Tau[i])
                
                Z =  torch.from_numpy(Xws[i]).float().to(device)

                D1_tensor =  torch.from_numpy(D1).float().to(device)
                D2_tensor =  torch.from_numpy(D2).float().to(device)
                L_t_tensor =  torch.from_numpy(L_t[i]).float().to(device)
                D_t_tensor =  torch.from_numpy(D_t[i]).float().to(device)

                def loss_closure():
                    optim.zero_grad()

                    ZQ = models[i](Z)
                    ZQm = ZQ-muY_tensor[i][:,None]
                    ZQn = ZQ - muN_tensor[i][:, None]
                    loss = torch.trace(D1_tensor@(ZQm.T@(L_t_tensor@ZQm))@D1_tensor)+torch.trace(D2_tensor@(ZQn.T@(D_t_tensor@ZQn))@D2_tensor)#
                    if scaling: loss-=torch.linalg.slogdet(models[i].W.weight*torch.eye(Z.shape[1]))[1]


                    loss.backward()
                    return loss

                optim.step(loss_closure)
                
            
            

        Z =  [torch.from_numpy(Xws[i]@Wts[i]).float().to(device) for i in range(num_views)]
        Z_np = [models[i](Z[i]).detach().cpu().numpy()[:,:params[i]] for i in range(num_views)]
       
        N_np = [models[i](Z[i]).detach().cpu().numpy()[:,params[i]:] for i in range(num_views)]

       
        
        L_t, D_t, tauZ, tauN, muY, muN = em_tlasso_noise_with_mu(Z_np, N_np, 1, l,r, L_t, D_t)
         
        muY_tensor = [torch.from_numpy(muY[i]).float().to(device) for i in range(num_views)]
        muN_tensor = [torch.from_numpy(muN[i]).float().to(device) for i in range(num_views)]

        adj_Wi = [ abs(L_t[i].copy()) for i in range(num_views)]
        for i in range(num_views): adj_Wi[i][adj_Wi[i]!=0]=1
        for i in range(num_views):
            print("Old Edges:", 0.5*(adj_Wo[i].sum()-adj_Wi[i].shape[0]),"Edges:", 0.5*(adj_Wi[i].sum()-adj_Wi[i].shape[0]), "Intersection:",0.5*((adj_Wo[i]*adj_Wi[i]).sum()-adj_Wi[i].shape[0]),"TP:",0.5*((adj*adj_Wi[i]).sum()-adj_Wi[i].shape[0]) )
        print("Intersection between views", 0.5*((adj_Wi[0]*adj_Wi[1]).sum()-adj_Wi[1].shape[0]) )
        adj_Wo=adj_Wi
        
        Tau = [np.diag(np.concatenate((tauZ[i],tauN[i]))) for i in range(num_views)]
       



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
    
    Xs = []
    ks = []
    for df in data_file:

        dff = os.path.join(script_dir, df)
        X = pd.read_csv(dff, sep=",")
        X = X.drop(X.columns[0], axis=1)
    
        if reduction:
            if cfg.k is None:
                ks.append(number_ICA_components(X))
            else:
                ks = cfg.k
            params = None
        else:
            ks.append(X.shape[0])
            params = cfg.k
        
        X = X.to_numpy() # maybe .T
        Xs.append(X)

    adj_em = [0,0]
    adj = pickle.load(open(os.path.join(script_dir,"data/adj_bs.pickle"),"rb"))


    print("shape", Xs[0].shape)
    for s in range(cfg.start,cfg.end):
        try:
            Wi=fit_ica_tlasso(Xs, ks, device, T, l, r, to_whiten, s, params, cfg.scaling)


            adj_Wi = [abs(Wi[i].copy()) for i in range(2)]
            for i in range(2): 
                adj_Wi[i][adj_Wi[i]!=0]=1
                adj_Wi[i] = coo_matrix(adj_Wi[i])
            adj_em= [adj_em[i]+adj_Wi[i] for i in range(2)]
            del Wi, adj_Wi
            
            adj_em_eval = [adj_em[i].toarray().copy() for i in range(2)]
            
            for i in range(2):
                adj_em_eval[i][adj_em_eval[i]<adj_em_eval[i][0,0]/2] = 0
                adj_em_eval[i][adj_em_eval[i]!=0]=1
                print("Edges:", 0.5*(adj_em_eval[i].sum()-adj_em_eval[i].shape[0]), "TP:",0.5*((adj*adj_em_eval[i]).sum()-adj_em_eval[i].shape[0]) )




        except Exception as e:
            print("Error:",e)
        


        open_file = open(os.path.join(script_dir,f"{path}/res_{store_as_alias}_{l}_{cfg.start}_{cfg.end}.pickle"), "wb")
        pickle.dump( adj_em, open_file)
        open_file.close()


if __name__ == '__main__':
    main()