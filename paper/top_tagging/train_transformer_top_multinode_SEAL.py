import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch.nn as nn
from argparse import ArgumentParser
import h5py as h5
import energyflow as ef
# import hdf5plugin
import math
import time
import random
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import (
    DistributedSampler,
)  # Distribute data across multiple gpus
from torch.distributed import init_process_group, destroy_process_group

from deltaSEAL import * 


# define the transfromer model
class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_size,flash=False,mem_efficient=False):

        super().__init__()
        self.flash = flash
        self.mem_efficient = mem_efficient
        self.embed = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.ReLU(),
        )

        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        self.encoder_layer_3 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dim_feedforward=embed_dim)
        
        
        self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
        )
    
    def forward(self, x, mask=None):
        x = self.embed(x)
        # with torch.backends.cuda.sdp_kernel(enable_flash=self.flash, enable_math=True, enable_mem_efficient=self.mem_efficient): #, SDPBackend.EFFICIENT_ATTENTION
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION,torch.nn.attention.SDPBackend.MATH]):
            x = self.encoder_layer_1(x)
            x = self.encoder_layer_2(x)
            x = self.encoder_layer_3(x)
            x = torch.mean(x*mask.unsqueeze(-1), axis=1)
        return self.classifier(x)

class JetDataset(Dataset):
    def __init__(self, h5_file, factor = 1.0):
        self.h5_file = h5_file
        self.factor = factor
        with h5.File(self.h5_file, 'r') as hf:
            self.length = hf['pid'].shape[0]

    def __len__(self):
        return int(float(self.length)//self.factor)

    def __getitem__(self, idx):
        with h5.File(self.h5_file, 'r') as data:
                X = data["data"][idx,:,:7]
                Y = data["pid"][idx]
                W = data["weights"][idx]
                jet = data["jet"][idx,-4:]
                mask = np.all(np.abs(X) != 0, axis=1)
                X_jet = jet[None,:]
                X_cart = data["data"][idx,:,7:]
        
        return X.astype(np.float32), np.array(Y, dtype=np.float32), mask, W.astype(np.float32), X_cart.astype(np.float32), X_jet.astype(np.float32)
    

class JetDatasetTest(Dataset):
    def __init__(self, h5_file, factor = 1.0, shuffle = True, seed = 1000, print_idx = False,remove_wrong_jet_e=False,jet_e_sum_const = False,fix_wrong_preproc = False):
        self.h5_file = h5_file
        self.factor = factor
        self.seed = seed
        self.shuffle = shuffle
        self.print_idx = print_idx
        self.remove_wrong_jet_e = remove_wrong_jet_e
        self.jet_e_sum_const = jet_e_sum_const
        self.fix_wrong_preproc = fix_wrong_preproc
        with h5.File(self.h5_file, 'r') as hf:
            self.length = hf['pid'].shape[0]
        self.length_factor = int(float(self.length)//self.factor)
        if self.shuffle:
            np.random.seed(self.seed)
            self.vec = np.random.permutation(self.length)
        else:
            self.vec = np.arange(self.length)

    def __len__(self):
        
        return self.length_factor

    def __getitem__(self, idx):
        idx = self.vec[idx]
       
        with h5.File(self.h5_file, 'r') as data:
                X = data["data"][idx,:,:7]
                Y = data["pid"][idx]
                W = data["weights"][idx]
                jet = data["jet"][idx,-4:]
                mask = np.all(np.abs(X) != 0, axis=1)
                X_jet = jet[None,:]
                X_cart = data["data"][idx,:,7:]
                if self.remove_wrong_jet_e and self.jet_e_sum_const:
                    jet_e = np.sum(mask*np.exp(X[:,5])) #from sum of constituents' energies, as in preprocessing
                    mask_keep = (~np.all(mask)) * ((np.abs(jet_e-np.exp(X[0,5]-X[0,4]))/jet_e)<1e-5) #remove jets with jet_e that cannot be reconstructed from sum of energiess of the given constituents.
                if self.fix_wrong_preproc:
                    X_cart,X_jet = cartesian_from_X_jet_vec_np(X=X,jet_vec=X_jet)
                    X_cart = X_cart * mask[:,None]
                
        if self.remove_wrong_jet_e and self.jet_e_sum_const:
            return X.astype(np.float32), np.array(Y, dtype=np.float32), mask, W.astype(np.float32), X_cart.astype(np.float32), X_jet.astype(np.float32), mask_keep
        else:
            return X.astype(np.float32), np.array(Y, dtype=np.float32), mask, W.astype(np.float32), X_cart.astype(np.float32), X_jet.astype(np.float32)


def to_cylindrical(four_vec, log=True):
    E = four_vec[:,:,0]
    px = four_vec[:,:,1]
    py = four_vec[:,:,2]
    pz = four_vec[:,:,3]
    pt = torch.sqrt(px*px + py*py)
    phi = torch.arctan2(py,px)
    eta = torch.arcsinh(pz/pt)

    if log:
        cylindrical_four_vec = torch.cat([
            torch.log(E.unsqueeze(-1)),
            torch.log(pt.unsqueeze(-1)), 
            eta.unsqueeze(-1),
            phi.unsqueeze(-1)
        ], axis=2)

        cylindrical_four_vec = torch.where(cylindrical_four_vec < -1e30, 0, cylindrical_four_vec)
    else:
        cylindrical_four_vec = torch.cat([E.unsqueeze(-1),pt.unsqueeze(-1), eta.unsqueeze(-1),phi.unsqueeze(-1)], axis=2)

    
    return torch.nan_to_num(cylindrical_four_vec)


def to_cylindrical_np(four_vec, log=True):
    E = four_vec[:,0]
    px = four_vec[:,1]
    py = four_vec[:,2]
    pz = four_vec[:,3]
    pt = np.sqrt(px*px + py*py)
    phi = np.arctan2(py,px)
    eta = np.arcsinh(pz/pt)

    if log:
        cylindrical_four_vec = np.concatenate([
            np.log(np.expand_dims(E,  axis=-1)),
            np.log(np.expand_dims(pt, axis=-1)), 
            np.expand_dims(eta, axis=-1),
            np.expand_dims(phi, axis=-1)
        ], axis=-1)

        cylindrical_four_vec = np.where(cylindrical_four_vec < -1e30, 0, cylindrical_four_vec)
    else:
        cylindrical_four_vec = np.concatenate([
            np.expand_dims(E,  axis=-1),
            np.expand_dims(pt, axis=-1),
            np.expand_dims(eta, axis=-1),
            np.expand_dims(phi, axis=-1)
        ], axis=-1)
    
    return np.nan_to_num(cylindrical_four_vec)

def get_jet_relvars(four_vec, four_vec_cy, jet_four_vec_cy,jet_e_sum_const = False):
    
    pi = torch.tensor(math.pi)
    # log(E)
    log_E = torch.log(four_vec_cy[:,:,0])
    
    # log(pT)
    log_pt = torch.log(four_vec_cy[:,:,1])
    
    # log(E_const/E_jet)
    if jet_e_sum_const:
        jet_e = torch.sum(jet_four_vec_cy[:,:,0],dim=1,keepdim=True)
    else:
        jet_e = jet_four_vec_cy[:,:,0]
  
    log_Er = torch.log((four_vec_cy[:,:,0]/jet_e))

    # log(pt_const/pt_jet)
    log_ptr = torch.log((four_vec_cy[:,:,1]/jet_four_vec_cy[:,:,1]))

    # dEta
    dEta =  four_vec_cy[:,:,2] - jet_four_vec_cy[:,:,2]

    # dPhi
    dPhi =  four_vec_cy[:,:,3] - jet_four_vec_cy[:,:,3] 
    dPhi[dPhi>pi] -=  2*pi
    dPhi[dPhi<= - pi] +=  2*pi

    # dR
    dR = torch.sqrt(dEta**2 + dPhi**2)

    
    # order of features [dEta, dPhi, log_R_pt, log_Pt, log_R_E, log_E, dR]
    jet_features = torch.cat([
        dEta.unsqueeze(-1), 
        dPhi.unsqueeze(-1),
        log_ptr.unsqueeze(-1),
        log_pt.unsqueeze(-1),
        log_Er.unsqueeze(-1),
        log_E.unsqueeze(-1),
        dR.unsqueeze(-1)
    ], axis=2)

    zero_mask = (four_vec == 0.0).any(dim=-1, keepdim=True)
    zero_mask = zero_mask.expand_as(jet_features)
    jet_features[zero_mask] = 0.0
    
    return jet_features


def get_jet_relvars_mask(four_vec_cy, jet_four_vec_cy,jet_e_sum_const = False,mask=None,jet_e = None,debug=False):
    
    pi = torch.tensor(math.pi)
    # log(E)
    log_E = torch.log(four_vec_cy[:,:,0])
    
    # log(pT)
    log_pt = torch.log(four_vec_cy[:,:,1])
    
    # log(E_const/E_jet)
    if jet_e_sum_const:
        if jet_e is None:
            if mask is None:
                mask = (four_vec_cy[:,:,0]!=1.0) #came from an exponent of 0 which is just filling.
            
            jet_e = torch.sum(mask*four_vec_cy[:,:,0],dim=1,keepdim=True)
            if debug:
                print(f"get_jet_rel_vars: jet_e = {jet_e}")
    else:
        jet_e = jet_four_vec_cy[:,:,0]
    

    if len(jet_e.shape)==1:
        jet_e = jet_e.unsqueeze(-1)
    log_Er = torch.log((four_vec_cy[:,:,0]/jet_e))

    # log(pt_const/pt_jet)
    log_ptr = torch.log((four_vec_cy[:,:,1]/jet_four_vec_cy[:,:,1]))

    # dEta
    dEta =  four_vec_cy[:,:,2] - jet_four_vec_cy[:,:,2]

    # dPhi
    dPhi =  four_vec_cy[:,:,3] - jet_four_vec_cy[:,:,3] 
    dPhi[dPhi>pi] -=  2*pi
    dPhi[dPhi<= - pi] +=  2*pi

    # dR
    dR = torch.sqrt(dEta**2 + dPhi**2)

    
    # order of features [dEta, dPhi, log_R_pt, log_Pt, log_R_E, log_E, dR]
    jet_features = torch.cat([
        dEta.unsqueeze(-1), 
        dPhi.unsqueeze(-1),
        log_ptr.unsqueeze(-1),
        log_pt.unsqueeze(-1),
        log_Er.unsqueeze(-1),
        log_E.unsqueeze(-1),
        dR.unsqueeze(-1)
    ], axis=2)

    if mask is not None:
        zero_mask = mask.unsqueeze(-1)
    # zero_mask = (four_vec == 0.0).any(dim=-1, keepdim=True)
        zero_mask = zero_mask.expand_as(jet_features)
       
        jet_features*=zero_mask.type(jet_features.dtype)
        jet_features = torch.nan_to_num(jet_features,nan=0.0)
        
    
    return jet_features
    
def get_data(path, training=False):
    
    def shuffle(a ,b ,c, d, e):
        idx = np.random.permutation(len(a))
        return a[idx], b[idx], c[idx], d[idx], e[idx]
    

    data =  h5.File(path, 'r')
    print("Sucessfully opened h5 file...")

    X = data["data"]
    Y = data["pid"]
    w = data["weights"]
    X_jet = data["jet"][:,-4:]
    masks = np.all(np.abs(X) != 0, axis=2)
    X_jet = X_jet[:,None,:]

    print("Collected variables...")
    del data

    if training:
        X, Y, w, mask, X_jet = shuffle(X, Y, w, mask,X_jet)

    return X, Y, w, mask, X_jet 

def boost(data, device="cpu"):
    beta = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    # beta = np.repeat(beta, data.shape[1], axis=None)
    beta = beta.repeat(data.shape[1])
    gamma = (1-beta*beta)**(-0.5)

    beta = beta.to(device)
    gamma = gamma.to(device)

    E_b = gamma*(data[:,:,0].flatten()- beta* data[:,:,1].flatten() )
    px_b = gamma*(data[:,:,1].flatten() - beta* data[:,:,0].flatten())
    
    E_b = E_b.reshape(data.shape[0], data.shape[1])
    px_b = px_b.reshape(data.shape[0], data.shape[1])

    return  torch.cat([ E_b.unsqueeze(-1), px_b.unsqueeze(-1), data[:,:,2:]], axis = 2)

def boost_3d(data,jet_data, device="cpu", beta=None,beta_2_max = 0.95):

    # sample beta from sphere
    b1 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    b2 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    theta = 2 * np.pi * b1
    phi = np.arccos(1 - 2 * b2)
    
    beta_x = np.sin(phi) * np.cos(theta)
    beta_y = np.sin(phi) * np.sin(theta)
    beta_z = np.cos(phi)
    
    beta = torch.cat([beta_x.unsqueeze(-1),beta_y.unsqueeze(-1), beta_z.unsqueeze(-1)], axis=1)
    bf = torch.tensor(np.random.uniform(0, beta_2_max, size=(len(data),1)), dtype=torch.float32)
    bf = bf**(1/2)
    beta = beta*bf
    
    beta_norm = torch.norm(beta, dim=1) 

    # make sure we arent violating speed of light
    assert torch.all(beta_norm < 1)

    gamma = 1 / torch.sqrt(1 - (beta_norm)**2)

    beta_squared = (beta_norm)**2

    # make boost matrix
    L = torch.zeros((len(data), 4, 4)).to(device)
    L[:,0, 0] = gamma
    L[:,1:, 0] = L[:,0, 1:] = -gamma.unsqueeze(-1) * beta
    L[:, 1:, 1:] = torch.eye(3) + (gamma[...,None, None] - 1) * torch.einsum('bi,bj->bij', (beta, beta))/ beta_squared[...,None, None]
    
    assert torch.all (torch.linalg.det(L)) == True

    boosted_four_vector = torch.einsum('bij,bkj->bik', L.type(torch.float32), data.type(torch.float32)).permute(0, 2, 1) 
    boosted_jet_vector = torch.einsum('bij,bkj->bik', L.type(torch.float32), jet_data.type(torch.float32)).permute(0, 2, 1) 

    # Validate that energy values remain non-negative
    assert torch.all(boosted_four_vector[:, :, 0] >= 0), "Negative energy values detected in constituents!"
    assert torch.all(boosted_jet_vector[:, :, 0] >= 0), "Negative energy values detected in jets!"
    
    return boosted_four_vector, boosted_jet_vector

def to_cartesian(data_cyl, device="cpu", log=False,coords_dict=None):
    #usual transformation from cylindrical to cartesian coordinates
    if coords_dict is None:
        coords_dict = {'E': 0, 'pT': 1, 'eta':2,'phi':3}
    
    if log:
        E = torch.exp(data_cyl[:, :, coords_dict['E']])
        pT = torch.exp(data_cyl[:, :, coords_dict['pT']])
    else:
        E = data_cyl[:, :, coords_dict['E']]
        pT = data_cyl[:, :, coords_dict['pT']]
    eta = data_cyl[:, :, coords_dict['eta']]
    phi = data_cyl[:, :, coords_dict['phi']]
    pz = torch.sinh(eta)*pT
    px = pT*torch.cos(phi)
    py = pT*torch.sin(phi)
    data_cartesian = torch.stack([E, px, py, pz], axis=-1)
    return data_cartesian

def to_cartesian_np(data_cyl, log=False,coords_dict=None):
    #usual transformation from cylindrical to cartesian coordinates
    if coords_dict is None:
        coords_dict = {'E': 0, 'pT': 1, 'eta':2,'phi':3}
    
    if log:
        E = np.exp(data_cyl[ :, coords_dict['E']])
        pT = np.exp(data_cyl[:, coords_dict['pT']])
    else:
        E = data_cyl[ :, coords_dict['E']]
        pT = data_cyl[:, coords_dict['pT']]
    eta = data_cyl[ :, coords_dict['eta']]
    phi = data_cyl[ :, coords_dict['phi']]
    pz = np.sinh(eta)*pT
    px = pT*np.cos(phi)
    py = pT*np.sin(phi)
    data_cartesian = np.stack([E, px, py, pz], axis=-1)
    return data_cartesian

def undo_wrong_jet_transform(jet_vec, device="cpu", dtype = torch.float32):
    # print(f"before fixing shape of jet_vec_cartesian  = {jet_vec.shape}")
    jet_ptetaphims = ef.ptyphims_from_p4s(p4s=jet_vec.cpu())
    jet_ys = ef.ys_from_pts_etas_ms(pts=jet_ptetaphims[:,:,0], etas=jet_ptetaphims[:,:,1], ms=jet_ptetaphims[:,:,3])
    jet_vec_cartesian = ef.p4s_from_ptyphims(np.concatenate((jet_ptetaphims[:,:,0], jet_ys[:,:], jet_ptetaphims[:,:,2],jet_ptetaphims[:,:,3]), axis=-1))
    # print(f"new after fixing shape of jet_vec_cartesian  = {jet_vec_cartesian[:,None,:].shape}")
    return torch.tensor(jet_vec_cartesian[:,None,:],device=device, dtype=dtype)

def undo_wrong_jet_transform_np(jet_vec):
   
    jet_ptetaphims = ef.ptyphims_from_p4s(p4s=jet_vec)
    jet_ys = ef.ys_from_pts_etas_ms(pts=jet_ptetaphims[:,0], etas=jet_ptetaphims[:,1], ms=jet_ptetaphims[:,3])
    jet_vec_cartesian = ef.p4s_from_ptyphims(np.concatenate((jet_ptetaphims[:,0], jet_ys, jet_ptetaphims[:,2],jet_ptetaphims[:,3]), axis=-1))
    
    return jet_vec_cartesian

def cartesian_from_X_jet_vec_np(X,jet_cy=None,jet_e=None,jet_vec=None,coords_dict = None,X_dict = None,mask=None):
    if coords_dict is None:
        coords_dict = {'E': 0, 'pT': 1, 'eta':2,'phi':3}

    jet_relvars_dict = {'deta':0, 'dphi':1,'log_R_pT':2, 'log_pT':3, 'log_R_E':4,'log_E':5,'dR':6}
    if X_dict is None:
        X_dict = jet_relvars_dict
    

    if jet_e is None:
        if mask is None:
            mask = np.all(np.abs(X) != 0, axis=1)
    # Select the correct energy column
        if 'log_E' in X_dict:
            E_col = np.exp(X[:,X_dict['log_E']])
        elif 'E' in X_dict:
            E_col = X[:,X_dict['E']]
        else:
            E_col = None
        # Sum only valid energies per batch
        if E_col is not None:
            jet_e = (E_col * mask).sum()
        else:
            jet_e = None
       
    if jet_cy is None:
        if jet_vec is None:
            print(f"Cartesian jet_vec or cylindrical jet_cy are needed")
            return None, None
        else:
            jet_vec = undo_wrong_jet_transform_np(jet_vec)[None,:]
            
            jet_cy = to_cylindrical_np(jet_vec, log=False)
    else:
       
        jet_vec = to_cartesian_np(jet_cy, log=False, coords_dict=coords_dict)
    

    jet_E = jet_cy[:,coords_dict['E']]
    jet_pt = jet_cy[:,coords_dict['pT']]
    jet_eta = jet_cy[:,coords_dict['eta']]
    jet_phi = jet_cy[:,coords_dict['phi']]

    E=None
    pT = None
    # print_db(f"X_dict = {X_dict}",db=debug)
    if 'E' in X_dict:
        E = X[:,X_dict['E']]
    elif 'log_E' in X_dict:
        E = np.exp(X[:,X_dict['log_E']])
    
    if 'pT' in X_dict:
        pT = X[:,X_dict['pT']]
    elif 'log_pT' in X_dict:
        pT = np.exp(X[:,X_dict['log_pT']])

    eta = X[:,X_dict['deta']] + jet_eta if 'deta' in X_dict else X[:,X_dict['eta']] if 'eta' in X_dict else None
    phi = X[:,X_dict['dphi']] + jet_phi if 'dphi' in X_dict else X[:,X_dict['phi']] if 'phi' in X_dict else None

    data_cartesian = to_cartesian_np(np.stack([E, pT, eta, phi], axis=-1), log=False, coords_dict={'E': 0, 'pT': 1, 'eta':2,'phi':3})
    if mask is not None:
        data_cartesian*= mask[:,None]
    
    return data_cartesian, jet_vec


def undo_wrong_jet_transform(jet_vec, device="cpu", dtype = torch.float32):
   
    jet_ptetaphims = ef.ptyphims_from_p4s(p4s=jet_vec.cpu())
    jet_ys = ef.ys_from_pts_etas_ms(pts=jet_ptetaphims[:,:,0], etas=jet_ptetaphims[:,:,1], ms=jet_ptetaphims[:,:,3])
    jet_vec_cartesian = ef.p4s_from_ptyphims(np.concatenate((jet_ptetaphims[:,:,0], jet_ys[:,:], jet_ptetaphims[:,:,2],jet_ptetaphims[:,:,3]), axis=-1))
   
    return torch.tensor(jet_vec_cartesian[:,None,:],device=device, dtype=dtype)

def cartesian_from_X_jet_vec(X, device="cpu",jet_cy=None,jet_e=None,jet_vec=None,coords_dict = None,X_dict = None,mask=None):
    if coords_dict is None:
        coords_dict = {'E': 0, 'pT': 1, 'eta':2,'phi':3}

    jet_relvars_dict = {'deta':0, 'dphi':1,'log_R_pT':2, 'log_pT':3, 'log_R_E':4,'log_E':5,'dR':6}
    if X_dict is None:
        X_dict = jet_relvars_dict
    

    if jet_e is None:
        if mask is None:
            mask = torch.all(torch.abs(X) != 0, dim=2)
        
        # mask = torch.all(torch.abs(X) != 0, dim=2)  # shape: [batch, n_particles]
    # Select the correct energy column
        if 'log_E' in X_dict:
            E_col = torch.exp(X[:,:,X_dict['log_E']])
        elif 'E' in X_dict:
            E_col = X[:,:,X_dict['E']]
        else:
            E_col = None
        # Sum only valid energies per batch
        if E_col is not None:
            jet_e = (E_col * mask).sum(dim=1)
        else:
            jet_e = None
        # jet_e = torch.sum(torch.exp(X[X_dict['log_E']]),1) if 'log_E' in X_dict else (torch.sum(X[:,:,X_dict['E']],1) if 'E' in X_dict else None)
    if jet_cy is None:
        if jet_vec is None:
            print(f"Cartesian jet_vec or cylindrical jet_cy are needed")
            return None, None
        else:
            jet_vec = undo_wrong_jet_transform(jet_vec, device=device, dtype=X.dtype)
            jet_cy = to_cylindrical(jet_vec, log=False)
    else:
       
        jet_vec = to_cartesian(jet_cy, device=device, log=False, coords_dict=coords_dict)
    

    jet_E = jet_cy[:,:,coords_dict['E']]
    jet_pt = jet_cy[:,:,coords_dict['pT']]
    jet_eta = jet_cy[:,:,coords_dict['eta']]
    jet_phi = jet_cy[:,:,coords_dict['phi']]

    E=None
    pT = None
    # print_db(f"X_dict = {X_dict}",db=debug)
    if 'E' in X_dict:
        E = X[:,:,X_dict['E']]
    elif 'log_E' in X_dict:
        E = torch.exp(X[:,:,X_dict['log_E']])
    
    if 'pT' in X_dict:
        pT = X[:,:,X_dict['pT']]
    elif 'log_pT' in X_dict:
        pT = torch.exp(X[:,:,X_dict['log_pT']])

    eta = X[:,:,X_dict['deta']] + jet_eta if 'deta' in X_dict else X[:,:,X_dict['eta']] if 'eta' in X_dict else None
    phi = X[:,:,X_dict['dphi']] + jet_phi if 'dphi' in X_dict else X[:,:,X_dict['phi']] if 'phi' in X_dict else None

    

    data_cartesian = to_cartesian(torch.stack([E, pT, eta, phi], axis=-1), device=device, log=False, coords_dict={'E': 0, 'pT': 1, 'eta':2,'phi':3})
    if mask is not None:
        data_cartesian*= mask.unsqueeze(-1).type(data_cartesian.dtype)
    
    return data_cartesian, jet_vec

def sum_reduce(num, device):
    r''' Sum the tensor across the devices.
    '''
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def gather_tensors(tensor, world_size):
    # Create a list to hold the gathered tensors
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather tensors from all processes
    dist.all_gather(gather_list, tensor)
    
    # Concatenate the gathered tensors into one long tensor
    long_tensor = torch.cat(gather_list, dim=0)
    
    return long_tensor

def train_step(model, dataloader, cost, optimizer, epoch, device, penalty=None, boost_=boost, apply_GSEAL=False, apply_dSEAL = False,lam_GSEAL = 0.0, lam_dSEAL = 0.0, beta_2_max = 0.95,max_batch = "all",pt_range_GeV = "all",jet_e_sum_const=False,remove_wrong_jet_e=False,record=False):
    model.train()
    running_loss = 0.0
    rbce = 0.0
    rGSEAL = 0.0
    rdSEAL = 0.0
    tot_w = 0.0
    w_val = 0.0
    tot_jets = 0
    n_jets_val = 0
    
    time_dSEAL = 0.0
    time_bce = 0.0
    time_GSEAL = 0.0
    time_back = 0.0
    time_batch = 0.0
    time_idle = 0.0
    time_step = 0.0
    time_proc_batch = 0.0
    
    max_batches_txt = f"{max_batch}"
    max_batches_flag = max_batches_txt.isnumeric()
    max_batches = int(max_batch) if max_batches_txt.isnumeric() else int(len(dataloader))

    if device==0:
        print(f"apply_GSEAL = {apply_GSEAL}, apply_dSEAL = {apply_dSEAL}, lam_GSEAL = {lam_GSEAL}, lam_dSEAL = {lam_dSEAL}")    
    end_batch = time.time()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch: {epoch}", mininterval=20,disable=(device!=0))):
        start_batch = time.time()
        time_idle_batch = start_batch - end_batch
        time_idle += time_idle_batch
        if max_batches_flag:
            if batch_idx>=max_batches:
                break

        st_proc_batch = time.time()
        batch = [x.to(device, non_blocking=True) for x in batch]
        
        if jet_e_sum_const and remove_wrong_jet_e:
            X, y, mask, weights, X_cartesian, jet_vec, mask_keep = batch
        else:
            X, y, mask, weights, X_cartesian, jet_vec = batch
            mask_keep = None
        
        st_weights = time.time()
        tot_w+=sum(weights)
        tot_jets+=len(y)
        end_weights = time.time()
        if pt_range_GeV!="all":
            jet_pt = to_cylindrical(jet_vec, log=False)[:,:,1]
            jet_pt_valid = [any([pt_sec[0]<=jpt<=pt_sec[1] for pt_sec in pt_range_GeV]) for jpt in jet_pt]
            if jet_pt_valid.count(True) == 0:
                # if device==0:
                print(f"Warning: No valid jets in batch {batch_idx} for pt range {pt_range_GeV}. Skipping batch.")
                jet_pt_valid[0] = True  # Ensure at least one jet is valid to avoid empty tensors
                weights = weights[jet_pt_valid]
                weights = torch.zeros_like(weights) # Set weights to zero if no valid jets
                # continue
            else:
                weights = weights[jet_pt_valid]

            X = X[jet_pt_valid,:,:]
            y = y[jet_pt_valid]
            mask = mask[jet_pt_valid,:]
            X_cartesian = X_cartesian[jet_pt_valid,:,:]
            
            jet_vec = jet_vec[jet_pt_valid,:,:]

            if jet_e_sum_const and remove_wrong_jet_e:
                mask_keep = mask_keep[jet_pt_valid]
        
            # else:
            #     print(f"# valid jets in batch {batch_idx} for pt range {pt_range_GeV}: {jet_pt_valid.count(True)} out of {len(jet_pt_valid)}")
            
        w_val += sum(weights)
        n_jets_val+=len(y)
        # end_proc_batch = time.time()

        st_jet_e_sum_const =time.time()
        if jet_e_sum_const:
            if remove_wrong_jet_e:
                if mask_keep is None:
                    jet_e = np.sum(mask*np.exp(X[:,:,5]),axis=1) #from sum of constituents' energies, as in preprocessing
                    mask_keep = (~torch.all(mask,dim=1)) * ((torch.abs(jet_e-torch.exp(X[:,0,5]-X[:,0,4]))/jet_e)<1e-5) #remove jets with jet_e that cannot be reconstructed from sum of energiess of the given constituents.
                weights_SEAL = weights*mask_keep
               
                
            else:
                weights_SEAL = weights
               
        else:
       
            weights_SEAL = weights
           
        end_jet_e_sum_const = time.time()
        end_proc_batch = time.time()
        optimizer.zero_grad()  # Zero the gradients


        ############################ deltaSEAL #################################

        start_dSEAL = time.time()
        if apply_dSEAL:
            Xg = X.clone().detach().requires_grad_(True)
            outputs = model(Xg, mask)
        else:
            Xg=X
            outputs = None  # Forward pass
        if apply_dSEAL or record:
            dSEAL = deltaSEAL_scalar7(model = model, X =Xg, X_cartesian = X_cartesian, jet_vec = jet_vec, mask=mask, train = apply_dSEAL,jet_e_sum_const = jet_e_sum_const,output=outputs) * weights_SEAL 

        end_dSEAL = time.time()


            
        ############################  BCE   #################################
        
        start_bce = time.time()
        if not apply_dSEAL:
            outputs = model(X, mask)  # Forward pass
        bce = cost(outputs.reshape_as(y), y) * weights
        end_bce = time.time()

        ############################ GSEAL #################################
        # boosted variables
        start_GSEAL = time.time()
        if apply_GSEAL or record:
            X_boost, jet_boost  = boost_3d(X_cartesian, jet_vec, device=device,beta_2_max = beta_2_max)
            X_boost_cy = to_cylindrical(X_boost, log=False)
            jet_boost_cy = to_cylindrical(jet_boost, log=False)
            boost_jet_vars = get_jet_relvars_mask(four_vec_cy = X_boost_cy, jet_four_vec_cy =jet_boost_cy,jet_e_sum_const=jet_e_sum_const,mask=mask)


            outputs_boost = model(boost_jet_vars, mask)

            outputs_boost = torch.nan_to_num(outputs_boost, nan=0.5)
                
        
            GSEAL = penalty(outputs.reshape_as(y), outputs_boost.reshape_as(y)) * weights_SEAL
        end_GSEAL = time.time()
        

        
        start_back = time.time()
        # if device ==0:
        if apply_GSEAL:
            
            loss =  bce.mean() + lam_GSEAL*GSEAL.mean()
        elif apply_dSEAL:
            
            loss =  bce.mean() + lam_dSEAL*dSEAL.mean()
            
        else:
            loss = bce.mean()
            
        
        loss.backward()  # Backward pass
        
        st_step = time.time()
        #take care of unruly gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        
        optimizer.step()  # Update parameters
        
        end_step = time.time()

        if record:
            running_loss += loss.item()*len(y)
            rbce += bce.mean().item()*len(y)
            rGSEAL += GSEAL.mean().item()*len(y)
            rdSEAL += dSEAL.mean().item()*len(y)
        
        end_back = time.time()
        
        end_batch = time.time()
        

        time_dSEAL += end_dSEAL-start_dSEAL
        time_bce += end_bce-start_bce
        time_GSEAL += end_GSEAL-start_GSEAL
        time_back += end_back-start_back
        time_batch +=end_batch-start_batch
        time_proc_batch +=end_proc_batch-st_proc_batch
        time_step +=end_step - st_step

        if (batch_idx%1000==0) and device==0 and torch.distributed.get_rank()==0:
            print(f"batch {batch_idx},time for bce:{end_bce-start_bce:.2e}, GSEAL: {end_GSEAL-start_GSEAL:.2e}, dSEAL:{end_dSEAL-start_dSEAL:.2e}, back prop: {end_back-start_back:.2e}, overall:{end_batch-start_batch:.2e}, proc_batch: {end_proc_batch-st_proc_batch:.2e}, boost_coonsistent: {end_jet_e_sum_const-st_jet_e_sum_const:.2e}, idle:{time_idle_batch:.2e}, step: {end_step - st_step:.2e}")

    dist.barrier()
    
    distributed_batch = max_batches if max_batches_flag else sum_reduce(len(dataloader), device=device).item()
    times ={"dSEAL":sum_reduce(time_dSEAL, device=device).item()/distributed_batch,"bce":sum_reduce(time_bce, device=device).item()/distributed_batch,"GSEAL":sum_reduce(time_GSEAL, device=device).item()/distributed_batch,"back":sum_reduce(time_back, device=device).item()/distributed_batch,"step":sum_reduce(time_step, device=device).item()/distributed_batch,"batch": sum_reduce(time_batch, device=device).item()/distributed_batch, "idle": sum_reduce(time_idle, device=device).item()/distributed_batch} 
    
    
    tot_weights_val = sum_reduce(w_val, device=device).item()
    tot_jets_val = sum_reduce(n_jets_val, device=device).item()
    all_weights = sum_reduce(tot_w, device=device).item()
    all_jets = sum_reduce(tot_jets, device=device).item()
    txt_weights= f"no. of valid jets: {tot_jets_val} out of {all_jets} ({tot_jets_val/all_jets}:.3e), weighted fraction of valid jets: {tot_weights_val} out of {all_weights} ({(tot_weights_val/all_weights):.3e})"
    
    info = {}
    txt_weights = ''
    info["times"] = times
    info["txt_weights"] = txt_weights
    
    distributed_batch = tot_weights_val
    
   
    
    if record:
        distributed_loss = sum_reduce(running_loss, device=device).item()/distributed_batch
        distributed_bce = sum_reduce(rbce, device=device).item()/distributed_batch
        distributed_GSEAL = sum_reduce(rGSEAL, device=device).item()/distributed_batch
        distributed_dSEAL = sum_reduce(rdSEAL, device=device).item()/distributed_batch
        info["losses"] = [distributed_loss,distributed_bce,distributed_GSEAL,distributed_dSEAL]
    else:
        info["losses"] = []
    

    return info

def test_step(model, dataloader, cost, epoch, device, penalty=None, boost_=boost_3d, apply_GSEAL=False, apply_dSEAL = False,lam_GSEAL = 0.0, lam_dSEAL = 0.0,beta_2_max = 0.95,max_batch = "all",pt_range_GeV = "all",apply_weights = True,generators=None,jet_e_sum_const=False,remove_wrong_jet_e=False,record=False):
    model.eval()
    running_loss = 0.0
    rbce = 0.0
    rGSEAL = 0.0
    rdSEAL = 0.0
    tot_w = 0.0
    w_val = 0.0
    tot_jets = 0
    n_jets_val = 0
    
    max_batches_txt = f"{max_batch}"
    max_batches_flag = max_batches_txt.isnumeric()
    max_batches = int(max_batch) if max_batches_txt.isnumeric() else int(len(dataloader))
    

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch: {epoch}", mininterval=20,disable=(device!=0))):
        if max_batches_flag:
            if batch_idx>=max_batches:
                break
            
        batch = [x.to(device) for x in batch]
        if jet_e_sum_const and remove_wrong_jet_e:
            X, y, mask, weights, X_cartesian, jet_vec, mask_keep = batch
        else:
            X, y, mask, weights, X_cartesian, jet_vec = batch
            mask_keep = None


        tot_w+=sum(weights)
        tot_jets+=len(y)
        
        if pt_range_GeV!="all":
            jet_pt = to_cylindrical(jet_vec, log=False)[:,:,1]
            jet_pt_valid = [any([pt_sec[0]<=jpt<=pt_sec[1] for pt_sec in pt_range_GeV]) for jpt in jet_pt]
            X = X[jet_pt_valid,:,:]
            y = y[jet_pt_valid]
            mask = mask[jet_pt_valid,:]
            X_cartesian = X_cartesian[jet_pt_valid,:,:]
            weights = weights[jet_pt_valid]
            jet_vec = jet_vec[jet_pt_valid,:,:]
            if jet_e_sum_const and remove_wrong_jet_e:
                mask_keep = mask_keep[jet_pt_valid]
            
        w_val += sum(weights)
        n_jets_val+=len(y)
        

        if jet_e_sum_const and remove_wrong_jet_e:
            weights_SEAL = weights*mask_keep
               
        else:
            weights_SEAL = weights
    
        if apply_dSEAL:
            Xg = X.clone().detach().requires_grad_(True)
            outputs = model(Xg, mask)
        else:
            Xg=X
            outputs = None  # Forward pass
        if apply_dSEAL or record:
            dSEAL = deltaSEAL_scalar7(model, X = Xg, X_cartesian = X_cartesian, jet_vec = jet_vec, mask=mask, generators=generators, jet_e_sum_const=jet_e_sum_const,output=outputs) * weights_SEAL

        # boosted variables
        X_boost, jet_boost  = boost_3d(X_cartesian, jet_vec, device=device, beta_2_max=beta_2_max)
        X_boost_cy = to_cylindrical(X_boost, log=False)
        jet_boost_cy = to_cylindrical(jet_boost, log=False)
        boost_jet_vars = get_jet_relvars_mask(four_vec_cy=X_boost_cy, jet_four_vec_cy=jet_boost_cy, jet_e_sum_const=jet_e_sum_const, mask=mask)
    
        with torch.no_grad():
            if not apply_dSEAL:
                outputs = model(X, mask)  # Forward pass

            bce = cost(outputs.reshape_as(y), y) * weights

            if apply_GSEAL or record:
                outputs_boost = model(boost_jet_vars, mask)
                outputs_boost = torch.nan_to_num(outputs_boost, nan=0.5)
                GSEAL = penalty(outputs.reshape_as(y), outputs_boost.reshape_as(y)) * weights_SEAL
            
    
            if apply_GSEAL:
                loss =  bce.mean() + lam_GSEAL*GSEAL.mean()
            elif apply_dSEAL:
                loss =  bce.mean() + lam_dSEAL*dSEAL.mean()
            else:
                loss = bce.mean()

            running_loss += loss.item()*len(y)
            if record:
                rbce += bce.mean().item()*len(y)
                rGSEAL += GSEAL.mean().item()*len(y)
                rdSEAL += dSEAL.mean().item()*len(y)


    dist.barrier()
    
    distributed_batch = max_batches if max_batches_flag else sum_reduce(len(dataloader), device=device).item()
    
   
    
    tot_weights_val = sum_reduce(w_val, device=device).item()
    tot_jets_val = sum_reduce(n_jets_val, device=device).item()
    all_weights = sum_reduce(tot_w, device=device).item()
    all_jets = sum_reduce(tot_jets, device=device).item()
    
    txt_weights = f"no. of valid jets: {tot_jets_val} out of {all_jets}~{256*4*distributed_batch} ({(tot_jets_val/all_jets):.3e}), weighted fraction of valid jets: {tot_weights_val} out of {all_weights} ({(tot_weights_val/all_weights):.3e})"
    distributed_batch = tot_weights_val
    info = {}
    info["txt_weights"] = txt_weights

    distributed_loss = sum_reduce(running_loss, device=device).item()/distributed_batch
    if record:
        distributed_bce = sum_reduce(rbce, device=device).item()/distributed_batch
        distributed_GSEAL = sum_reduce(rGSEAL, device=device).item()/distributed_batch
        distributed_dSEAL = sum_reduce(rdSEAL, device=device).item()/distributed_batch
        info["losses"] = [distributed_loss,distributed_bce,distributed_GSEAL,distributed_dSEAL]
    else:
        info["losses"] = [distributed_loss]

    return info

def evaluate(model, loader, device, beta_2_max = 0.95,jet_e_sum_const=False,remove_wrong_jet_e = False):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    pred = []
    boost_ = []
    true = []

    with torch.no_grad():  # Disable gradient calculation
        
        for batch in loader:
            if jet_e_sum_const and remove_wrong_jet_e:
                X, y, mask, weights, X_cartesian, jet_vec, mask_keep = batch
            else:
                X, y, mask, weights, X_cartesian, jet_vec = batch
                mask_keep = None
            # X, y, mask, weights, X_cartesian, jet_vec = batch
            X = X.to(device)
            y = y.to(device)
            X_cartesian = X_cartesian.to(device)
            mask = mask.to(device)
            weights = weights.to(device)
            jet_vec = jet_vec.to(device)

            # boosted variables
            X_boost, jet_boost  = boost_3d(X_cartesian, jet_vec, device=device, beta_2_max=beta_2_max)
            X_boost_cy = to_cylindrical(X_boost, log=False)
            jet_boost_cy = to_cylindrical(jet_boost, log=False)
            boost_jet_vars = get_jet_relvars_mask(four_vec_cy=X_boost_cy, jet_four_vec_cy=jet_boost_cy, jet_e_sum_const=jet_e_sum_const, mask=mask)

            outputs = model(X, mask)  # Forward pass
            outputs_boost = model(boost_jet_vars, mask)

            pred.append(outputs)
            boost_.append(outputs_boost)
            true.append(y)

    # model.to("cpu")
    
    return torch.cat(pred), torch.cat(boost_), torch.cat(true)

def train_model(model, train_loader, test_loader, loss, optimizer, train_sampler, num_epochs=100, device='cpu', global_rank=0,patience=50, penalty=None, output_dir="", boost_=boost, apply_GSEAL=False, save_tag="",apply_dSEAL = False,lam_GSEAL = 0.0, lam_dSEAL = 0.0,beta_2_max = 0.95,num_batches="all",factor = 1.0,pt_range = "all",batch_size=256*4,jet_e_sum_const=False,remove_wrong_jet_e=False,record=False):
    
    print(f"Process ID: {device}")
    seed_tag=f"seed_{args.save_tag}"
    pen_tag = f"GSEAL_lam_{lam_GSEAL}_b2max_{beta_2_max}_" if apply_GSEAL else ""
    dSEAL_tag = f"dSEAL_lam_{lam_dSEAL}_" if apply_dSEAL else ""
    boost_tag = "3d" if boost_ is boost_3d else "1d"
  
    num_batches_tag = f"num_batches_{num_batches}_" if num_batches.isnumeric() else ""
    size_batch_tag = f"tot_batch_{batch_size*world_size}_"
    patience_tag = f"patience_{patience}_"
    factor_tag = "" if factor==1.0 else f"red_factor_{factor}_"
    num_epochs_tag = f"num_epochs_{num_epochs}_"
    pt_name = ""
    if pt_range!="all":
        pt_name = "pt_range_"
        for pt in pt_range:
            pt_name += f"{pt}_"
        pt_name+="incl_"
        pt_name = pt_name.replace(",","_").replace(" ","_").replace("__","_")
       
    pt_tag = pt_name
    pt_shuffle_tag = "pt_shuffle_" if args.pt_shuffle else ""
    boost_fix_tag = "jet_e_sum_const_" if jet_e_sum_const else ""
    boost_fix_tag += "remove_wrong_jet_e_" if args.remove_wrong_jet_e else ""
    boost_fix_tag += "fix_wrong_preproc_" if args.fix_wrong_preproc else ""

    model_save = f"best_model_{boost_tag}_boost_{pen_tag}{dSEAL_tag}{size_batch_tag}{patience_tag}{num_epochs_tag}{factor_tag}{pt_tag}{pt_shuffle_tag}{boost_fix_tag}{seed_tag}.pt"

    if global_rank==0:
        print(f"Saving model as: {model_save}")

    losses = {
        "train_loss": [],
        "train_BCE": [],
        "train_GSEAL": [],
        "train_dSEAL": [],
        "val_loss": [],
        "val_BCE": [],
        "val_GSEAL": [],
        "val_dSEAL": []
    }
    times = {}
    tracker = {
        "bestValLoss": np.inf,
        "bestEpoch": 0
    }
    debug = False 

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        info_train = train_step(model, train_loader, loss, optimizer, epoch, device, penalty, boost_, apply_GSEAL,apply_dSEAL = apply_dSEAL,lam_GSEAL = lam_GSEAL, lam_dSEAL = lam_dSEAL,beta_2_max = beta_2_max,max_batch=num_batches,pt_range_GeV = pt_range, jet_e_sum_const = jet_e_sum_const,remove_wrong_jet_e=remove_wrong_jet_e,record=record)
        info_val = test_step(model, test_loader, loss, epoch, device, penalty, boost_, apply_GSEAL, apply_dSEAL = apply_dSEAL,lam_GSEAL = lam_GSEAL, lam_dSEAL = lam_dSEAL,beta_2_max = beta_2_max,max_batch=num_batches,pt_range_GeV = pt_range, jet_e_sum_const = jet_e_sum_const,remove_wrong_jet_e=remove_wrong_jet_e,record=record)
        
        times = info_train["times"] if "times" in list(info_train.keys()) else {}
        txt_weights_train = info_train["txt_weights"] if "txt_weights" in list(info_train.keys()) else ""
        
        txt_weights_val = info_val["txt_weights"] if "txt_weights" in list(info_val.keys()) else ""
        

        losses["val_loss"].append(info_val["losses"][0]) 
        if record:
            # if "losses" in info_val:
            losses["val_BCE"].append(info_val["losses"][1]) 
            losses["val_GSEAL"].append(info_val["losses"][2]) 
            losses["val_dSEAL"].append(info_val["losses"][3]) 

            # if "losses" in info_train:
            losses["train_loss"].append(info_train["losses"][0]) 
            losses["train_BCE"].append(info_train["losses"][1]) 
            losses["train_GSEAL"].append(info_train["losses"][2]) 
            losses["train_dSEAL"].append(info_train["losses"][3]) 
            # print(f"losses = {losses}")

        if global_rank == 0:
            if epoch==0:
                print(f"train: {txt_weights_train}")
                print(f"val: {txt_weights_val}")
            local_time_struct = time.localtime()
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_struct)
            if record:
                print(f'[{formatted_time}] Epoch [{epoch+1}/{num_epochs}] Loss: {losses["train_loss"][-1]:.4e}, Val Loss: {losses["val_loss"][-1]:.4e},  BCE: {losses["train_BCE"][-1]:.4e}, Val BCE: {losses["val_BCE"][-1]:.4e}, GSEAL: {losses["train_GSEAL"][-1]:.4e}, Val GSEAL: {losses["val_GSEAL"][-1]:.4e}, Symm: {losses["train_dSEAL"][-1]:.4e}, Val Symm: {losses["val_dSEAL"][-1]:.4e} ')
            
            if times!={}:
                try:
                    print(f"time for bce:{times['bce']:.2e}, GSEAL: {times['GSEAL']:.2e}, dSEAL:{times['dSEAL']:.2e}, back prop: {times['back']:.2e}, overall:{times['batch']:.2e}, idle:{times["idle"]:.2e}, step: {times["step"]:.2e}")
                except Exception as e:
                    if debug:
                        print(e)

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
                tracker["bestValLoss"] = losses["val_loss"][-1]
                tracker["bestEpoch"] = epoch
                
                dist.barrier()

                if global_rank==0:
                    torch.save(
                        model.module.state_dict(), f"{output_dir}/{model_save}"
                    )

        dist.barrier() # syncronise (top GPU is doing more work)

        # check the validation loss from each GPU:
        debug = False 
        if debug:
            print(f"Rank: {global_rank}, Device: {device}, Train Loss: {losses['train_loss'][-1]:.5f}, Validation Loss: {losses['val_loss'][-1]:.5f}")
            print(f"Rank: {global_rank}, Device: {device}, Best Loss: {tracker['bestValLoss']}, Best Epoch: {tracker['bestEpoch']}")
        # early stopping check
        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on Rank: {global_rank}, device: {device}")
            break
        
    if global_rank==0:
        print(f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!")
    
        # save losses
        loss_save = model_save.replace(".pt",".json").replace("best_model","training")

        json.dump(losses, open(f"{output_dir}/{loss_save}", "w"))

        
        

# Each process control a single gpu
def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8870"  # select any idle port on your machine

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(world_size, global_rank, rank, args=None):
    # ddp_setup(rank, world_size)

    # set random seeds
    np.random.seed(int(args.save_tag))
    torch.manual_seed(int(args.save_tag))
    random.seed(int(args.save_tag))


    # make dataset'
  
    train_data = JetDatasetTest(f"{args.data_dir}/train_atlas_symmetry.h5",factor = args.factor, shuffle = args.pt_shuffle, seed = int(args.save_tag),remove_wrong_jet_e=args.remove_wrong_jet_e,fix_wrong_preproc = args.fix_wrong_preproc,jet_e_sum_const=args.jet_e_sum_const)
    test_data = JetDatasetTest(f"{args.data_dir}/val_atlas_symmetry.h5",factor = args.factor, shuffle = args.pt_shuffle, seed = int(args.save_tag),remove_wrong_jet_e=args.remove_wrong_jet_e,fix_wrong_preproc = args.fix_wrong_preproc,jet_e_sum_const=args.jet_e_sum_const)
   

    # set random seeds
    np.random.seed(int(args.save_tag))
    torch.manual_seed(int(args.save_tag))
    random.seed(int(args.save_tag))

    # distributed loader
    sampler_train = DistributedSampler(train_data, shuffle=True, num_replicas=world_size, rank=global_rank)
    sampler_test = DistributedSampler(test_data, shuffle=False, num_replicas=world_size, rank=global_rank)
    
    batch_size = int(args.batch_size)
    # make dataloader
    train_loader = DataLoader(
            train_data, 
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler_train,
            num_workers=args.num_workers,
            pin_memory=True,
            # prefetch_factor=4
            )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler_test,
        num_workers=args.num_workers,
        pin_memory=True,
        # prefetch_factor=4
        )

    

    # set up model
    
    model = Transformer(input_dim=7, embed_dim=256, hidden_size=128,flash=args.flash,mem_efficient = args.mem_efficient)
    model = DDP(model.to(rank), device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    if rank==0:
        d = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Training on device: {d} Rank: {global_rank}")

    # train
    BCE = nn.BCELoss(reduction='none')
    MSE = nn.MSELoss(reduction='none')
    boost_dict = {
        "1D": boost,
        "3D": boost_3d
    }
    
    #pt cuts for training
    pt_range = "all"
  
    if args.pt_intrp:
        pt_range = [[0.0,1500.0],[2500.0,float("inf")]]
    elif args.pt_extrp:
        pt_range = [[0.0,2000.0]]
    elif args.pt_extrp_1TeV:
        pt_range = [[0.0,1000.0]]
    elif args.pt_ranges!=[] and args.pt_ranges!="all":
        pt_range = args.pt_ranges
        if len(pt_range)%2!=0:
            if (0.0 in pt_range or float("-inf") in pt_range) and (not(float("inf") in pt_range)):
                pt_range.append(float("inf"))
            elif not(0.0 in pt_range or float("-inf") in pt_range) and ((float("inf") in pt_range)):
                pt_range.insert(0,0.0)
            else:
                pt_range="all"
                print("odd number of pt_ranges, not cuttiing on pt")
        
        if len(pt_range)%2==0 and pt_range!="all":
            pt_range = [[float(pt_range[2*i]),float(pt_range[2*i+1])] for i in range(len(pt_range)//2)]
    print(f"pt range for training: {pt_range}")
            
    

    train_model(
        model, 
        train_loader, 
        test_loader, 
        BCE, 
        optimizer, 
        train_sampler=sampler_train, 
        device=rank,
        global_rank=global_rank,
        patience=args.patience, 
        penalty=MSE, 
        num_epochs = args.num_epochs,
        output_dir=args.outdir, 
        boost_=boost_dict[args.boost_type],
        apply_GSEAL=args.apply_GSEAL,
        lam_GSEAL=args.lam_GSEAL,
        beta_2_max = args.beta_2_max,
        apply_dSEAL=args.apply_dSEAL,
        lam_dSEAL=args.lam_dSEAL,
        save_tag=args.save_tag,
        num_batches = args.num_batches,
        factor = args.factor,
        pt_range = pt_range,
        batch_size=batch_size,
        jet_e_sum_const= args.jet_e_sum_const,
        remove_wrong_jet_e= args.remove_wrong_jet_e,
        record = args.record
        )

    dist.barrier()

    if args.run_eval and global_rank == 0:
        print("Running Evaluation...")
        
        seed_tag=f"seed_{args.save_tag}"
        pen_tag = f"GSEAL_lam_{lam_GSEAL}_b2max_{beta_2_max}" if apply_GSEAL else ""
        dSEAL_tag = f"dSEAL_{lam}_{lam_dSEAL}_" if apply_dSEAL else ""
        boost_tag = "3d" if boost_ is boost_3d else "1d"
        
        num_batches_tag = f"num_batches_{num_batches}_" if num_batches.isnumeric() else ""
        size_batch_tag = f"tot_batch_{batch_size*world_size}_"
        patience_tag = f"patience_{args.patience}_"
        num_epochs_tag = f"num_epochs_{num_epochs}_"
        factor_tag = "" if factor==1.0 else f"red_factor_{factor}_"
        pt_name = ""
        if pt_range!="all":
            pt_name = "pt_range_"
            for pt in pt_range:
                pt_name += f"{pt}_"
            pt_name+="incl_"
            pt_name = pt_name.replace(",","_").replace(" ","_").replace("__","_")

        pt_tag = pt_name

        pt_shuffle_tag = "pt_shuffle_" if args.pt_shuffle else ""
        boost_fix_tag = "jet_e_sum_const_" if args.jet_e_sum_const else ""
        boost_fix_tag += "remove_wrong_jet_e_" if args.remove_wrong_jet_e else ""
        boost_fix_tag += "fix_wrong_preproc_" if args.fix_wrong_preproc else ""
        model_save = f"best_model_{boost_tag}_boost_{pen_tag}{dSEAL_tag}{num_batches_tag}{size_batch_tag}{patience_tag}{num_epochs_tag}{factor_tag}{pt_tag}{pt_shuffle_tag}{boost_fix_tag}{seed_tag}.pt"

       

        model2 = Transformer(input_dim=7, embed_dim=256, hidden_size=128)
        model2.load_state_dict(torch.load(f"{args.outdir}/{model_save}"))

       
        
        print("Evaluating Loaded Model...")
        preds, _, trues = evaluate(model2, test_loader, rank)
        
        plt.figure()
        plt.hist(preds[torch.where(trues==0)].cpu().flatten().numpy(), color="orange", histtype="step")
        plt.hist(preds[torch.where(trues==1)].cpu().flatten().numpy(), color="blue", histtype="step")
        plt.savefig("Evaluation_Separation_Load_full.png", dpi=100)
    
    dist.barrier()
    # destroy_process_group()

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default="",
        help="Directory of training and validation data"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="outdir",
        default="",
        help="Directory to output best model",
    )
    parser.add_argument(
        "--save_tag",
        dest="save_tag",
        default="",
        help="Extra tag for checkpoint model",
    )
    parser.add_argument(
        "--apply_GSEAL",
        dest="apply_GSEAL",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--apply_dSEAL",
        dest="apply_dSEAL",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--lam_GSEAL",
        dest="lam_GSEAL",
        default=0.0,
        type = float,
        help = "weight for GSEAL"
    )
    parser.add_argument(
        "--lam_dSEAL",
        dest="lam_dSEAL",
        default=0.0,
        type = float,
        help = "weight for deltaSEAL"
    )
    parser.add_argument(
        "--b2",
        dest="beta_2_max",
        default=0.95,
        type = float,
        help = "maximal beta^2 for GSEAL"
    )
    parser.add_argument(
        "--boost_type",
        dest="boost_type",
        default="1D",
        choices=["1D", "3D"]
    )
    parser.add_argument(
        "--run_eval",
        dest="run_eval",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--pt_shuffle",
        dest="pt_shuffle",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        default=100,
        type=int,
        help="number of epochs"
    )
    parser.add_argument(
        "--num_batches",
        dest="num_batches",
        default="all",
        help="number of batches"
    )
    parser.add_argument(
        "--factor",
        dest = "factor",
        default = 1.0,
        type = float,
        help = "data reduction factor"
        )
    parser.add_argument(
        "--pt_intrp",
        dest = "pt_intrp",
        default = False,
        action = "store_true",
        help = "pt ranges for training set to be pt<=1.5 TeV pt>=2.5 TeV"
        )
    parser.add_argument(
        "--pt_extrp",
        dest = "pt_extrp",
        default = False,
        action = "store_true",
        help = "pt ranges for training set to be pt<=2 TeV"
        )
    parser.add_argument(
        "--pt_extrp_1TeV",
        dest = "pt_extrp_1TeV",
        default = False,
        action = "store_true",
        help = "pt ranges for training set to be pt<=1 TeV"
        )
    parser.add_argument(
        "--pt_ranges",
        dest = "pt_ranges",
        nargs='+',
        type=float,
        default = [],
        help = "sets of pt_min pt_max in GeV example: 300.0 1000.0 1500.0 2000.0 means 300<pt<1000 GeV, 1500<pt<2000 GeV"
        )
    
    parser.add_argument(
        "--batch_size",
        dest = "batch_size",
        type=float,
        default = 1024,
        help = "batch size per GPU"
        )
    
    parser.add_argument(
        "--num_workers",
        dest = "num_workers",
        type=int,
        default = 16,
        help = "num workers in loader"
        )
    
    parser.add_argument(
        "--patience",
        dest = "patience",
        type=float,
        default = 5,
        help = "patience for early stopping"
        )
    
    parser.add_argument(
        "--fix_wrong_preproc",
        dest = "fix_wrong_preproc",
        action = "store_true",
        default = False,
        help = "recalculate cartesian 4-vectors since were wrong in preprocessing."
        )


    parser.add_argument(
        "--jet_e_sum_const",
        dest = "jet_e_sum_const",
        action = "store_true",
        default = False,
        help = "make boost consistent with preprocessing"
        )
    
    parser.add_argument(
        "--remove_wrong_jet_e",
        dest = "remove_wrong_jet_e",
        action = "store_true",
        default = False,
        help = "don't take jets into account in SEALs if their boosted jet energy can't be calculated from constituents (only effective if --jet_e_sum_const)."
        )
    
    parser.add_argument(
        "--flash",
        dest = "flash",
        action = "store_true",
        default = False,
        help = "enable flash attention"
        )
    

    parser.add_argument(
        "--mem_efficient",
        dest = "mem_efficient",
        action = "store_true",
        default = False,
        help = "enable mem_efficient"
        )
    
    parser.add_argument(
    "--record",
    dest = "record",
    action = "store_true",
    default = False,
    help = "record losses"
    )
    
    args = parser.parse_args()
    world_size = int(os.environ['WORLD_SIZE'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    global_rank = torch.distributed.get_rank()
    rank = int(os.environ['LOCAL_RANK'])
    
    main(world_size, global_rank, rank, args)
    # mp.spawn(
    #     main,
    #     args=(world_size, args),
    #     nprocs=world_size,
    # ) 
