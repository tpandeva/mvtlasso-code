import torch
import pandas as pd
import hydra
from omegaconf import OmegaConf
import numpy as np
from rpy2.robjects import r as r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
import pickle
from sklearn.decomposition import FastICA, PCA

from model._tlasso import run_glasso,em_tlasso,em_tlasso_noise
from sklearn.utils.extmath import randomized_svd
from rpy2.robjects import r as r
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()


from scipy.sparse import coo_matrix
def run_jgl(s, lg):
    print("Start R: JGL")
    r.library("JGL")

    # Prepare the call for GLasso
    glasso = r("JGL")

    rpy2.robjects.numpy2ri.activate()

    results = glasso(s,"fused",lg, 0.1, 1,"equal",False, 500, 1e-5, robjects.NULL,True)
    wi = results[0]
    return wi

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



def fit_ica_tlasso(Xs, ks, device, T, l, to_whiten, seed=1, params=None):
    np.random.seed(seed)
    num_views = 2

    Xws, Wts, XwsT = [], [], []
    for i in range(num_views):
        X = Xs[i]
        
        n = X.shape[1]
        X = X[:,np.random.choice(n, size=int(n*0.9), replace=False)]
        #ica = FastICA(n_components=X.shape[0])
        #Xica = ica.fit_transform(X)
        Xica=X
        #print(Xica.std(0))
        print(Xica.shape)
        #Xws.append(Xica)
        Xws.append(Xica)
        XwsT.append(Xica.T/Xica.std(1))
    
        
    #### run glasso
    Xg = np.hstack(Xws)
    L_g = run_glasso(np.corrcoef(Xg),l)
    #### run JGL 
    
    #L_f = run_jgl(XwsT, l)
    return [L_g]#L_f[0],L_f[1]]

@hydra.main(config_path="config/", config_name="config.yaml")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
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
    
    Xs = []
    ks = []
    for df in data_file:
        dff = os.path.join(script_dir, df)
        X = pd.read_csv(dff, sep=",")
        X = X.drop(X.columns[0], axis=1)

        params = cfg.k
        
        X = X.to_numpy() # maybe .T
        Xs.append(X)
        

    adj_em = [0,0,0]

    #####simulated data######
   # Xs = pickle.load(open("/var/scratch/tpandeva/siica/data/simulated.pickle","rb"))
   # ks = [60]
   # params = [50]
    print("shape", Xs[0].shape)
    adj = pickle.load(open(os.path.join(script_dir, "data/adj_bs.pickle"), "rb"))
    for s in range(cfg.start,cfg.end):
        # try:
        Wi=fit_ica_tlasso(Xs, ks, device, T, l, to_whiten, s, params)

        adj_Wi = [abs(Wi[i].copy()) for i in range(1)]
        for i in range(1): 
            adj_Wi[i][adj_Wi[i]!=0]=1
            adj_Wi[i] = coo_matrix(adj_Wi[i])
        adj_em= [adj_em[i]+adj_Wi[i] for i in range(1)]
        del Wi, adj_Wi
        adj_em_eval = [adj_em[i].toarray().copy() for i in range(1)]
            
        for i in range(1):
                adj_em_eval[i][adj_em_eval[i]<adj_em_eval[i][0,0]/2] = 0
                adj_em_eval[i][adj_em_eval[i]!=0]=1
                print("Edges:", 0.5*(adj_em_eval[i].sum()-adj_em_eval[i].shape[0]), "TP:",0.5*((adj*adj_em_eval[i]).sum()-adj_em_eval[i].shape[0]) )


#         adj_W0 = abs(Wi[0].copy())
#         adj_W0[adj_W0!=0]=1
#         adj_0+=adj_W0
            
        # except Exception as e:
        #     print("Error:",e)
        


        open_file = open(os.path.join(script_dir,f"{path}/res_{store_as_alias}_{l}_{cfg.start}_{cfg.end}.pickle"), "wb")
        pickle.dump( adj_em, open_file)
        open_file.close()


if __name__ == '__main__':
    main()