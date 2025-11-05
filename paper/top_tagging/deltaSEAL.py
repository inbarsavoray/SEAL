import torch
import torch.nn as nn
import contextlib
import warnings
import h5py as h5
import energyflow as ef
import numpy as np
import math
import time
import random
from torch.utils.data import Dataset, DataLoader

def print_db(txt,db = True):
    if db:
        print(txt)


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
   
        zero_mask = zero_mask.expand_as(jet_features)
       
        jet_features*=zero_mask.type(jet_features.dtype)
        jet_features = torch.nan_to_num(jet_features,nan=0.0)
       
    return jet_features
    
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

def undo_wrong_jet_transform(jet_vec, device="cpu", dtype = torch.float32):
    # print(f"before fixing shape of jet_vec_cartesian  = {jet_vec.shape}")
    jet_ptetaphims = ef.ptyphims_from_p4s(p4s=jet_vec.cpu())
    jet_ys = ef.ys_from_pts_etas_ms(pts=jet_ptetaphims[:,:,0], etas=jet_ptetaphims[:,:,1], ms=jet_ptetaphims[:,:,3])
    jet_vec_cartesian = ef.p4s_from_ptyphims(np.concatenate((jet_ptetaphims[:,:,0], jet_ys[:,:], jet_ptetaphims[:,:,2],jet_ptetaphims[:,:,3]), axis=-1))
    # print(f"new after fixing shape of jet_vec_cartesian  = {jet_vec_cartesian[:,None,:].shape}")
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

class deltaSEAL(nn.Module):
    """
    A custom loss module designed to enforce Lorentz invariance/covariance in neural network outputs. This loss function is particularly useful in physics-based models where Lorentz symmetry is important, such as in particle physics applications.

    A key assumption of this loss function is that the list of model coordinates (`model_coords`) is passed in the same order as the input coordinates. 
    This assumption is necessary for the loss to compute the correct variations.
    In this implementation, this assumption is not validated.

    Attributes:
    -----------
    supported_coords : set
        A set of all supported coordinate types that can be used in `model_coords`.
    
    preset_system : dict
        A dictionary of preset coordinate systems (e.g., 'Cylindrical', 'ATLAS'), each containing a specific list of coordinates.

    
    model_coord_dict : dict
        A dictionary mapping each model coordinate to its index in the input tensor. Either directly provided or derived from the `preset_system`.
    
    Methods:
    --------
    forward(model, X, X_cartesian, jet_vec, mask=None, train=False, generators=None, take_mean=False)
        Computes the deltaSEAL based on the input data, the model's output, and the coordinates provided.

    compute_var_tensor(X, grads, trans_dict, trans_dict_jet, coord, coord_ind)
        Computes the variation tensor for a specific coordinate.

    get_trans_dict(cartesian_features)
        Returns a transformation dictionary that converts Cartesian coordinates into cylindrical coordinates.

    to_cylindrical(four_vec, log=True)
        Converts 4-vectors from Cartesian to cylindrical coordinates, optionally applying logarithms to energy and transverse momentum.
    """
    
    def __init__(self, model_coords,model_coord_dict = {},jet_e_sum_const= False):
        """
        Initializes the deltaSEAL module with a list of model coordinates or a preset system.
        
        Parameters:
        -----------
        model_coords : list or str
            A list of coordinate names to be used in the loss calculation or a string representing a preset coordinate system.
            If a list, should be ordered in the order of model inputs. If using preset system, make sure the default order matches the model input.
        """
        super(deltaSEAL, self).__init__()
        self.jet_e_sum_const = jet_e_sum_const
        #List of supported generators
        self.generators_dict = {"Lx":int(0),"Ly":int(1),"Lz":int(2),"Kx":int(3),"Ky":int(4),"Kz":int(5)}
        # List of supported coordinates
        self.supported_coords = {'E', 'pT', 'eta', 'phi', 'log_E', 'log_pT', 'log_R_pT', 'log_R_E', 'dR', 'deta', 'dphi', 'log_1pE','log_1ppT'}
        
        # Preset coordinate systems for convenience
        self.preset_system = {
            'Cylindrical':      ['E', 'pT', 'eta', 'phi'], 
            'Cylindrical_log':  ['log_E', 'log_pT', 'eta', 'phi'],
            'ATLAS':            ['deta', 'dphi', 'log_R_pT', 'log_pT', 'log_R_E', 'log_E', 'dR'],
            'TopTag':           ['log_E', 'log_pT', 'eta', 'phi', 'log_R_pT', 'log_R_E', 'deta', 'dphi', 'dR'],
            'ATLAS_OLpp':       ['log_1pE', 'log_1ppT', 'deta', 'dphi']
        }

        # Check if model_coords is a valid input (list of strings or a supported preset system)
        error_msg = f"Invalid 'model_coords'. Expected a list of supported coordinates or a supported preset system.\nSupported coordinates: {self.supported_coords}\nSupported preset systems: {self.preset_system.keys()}"
        if isinstance(model_coords, list):
            # If model_coords is a list, ensure all entries are valid
            if not all(c in self.supported_coords for c in model_coords):
                raise ValueError(error_msg)
        elif isinstance(model_coords, str):
            # If it's a string, check if it's a valid preset system
            if model_coords not in self.preset_system:
                raise ValueError(error_msg)
            model_coords = self.preset_system[model_coords]  # Replace the string with the corresponding preset
        else:
            raise ValueError(error_msg)
        
        # In case 'dR' is in the list, check that both 'deta' and 'dphi' are also in the list
        if 'dR' in model_coords:
            if 'deta' not in model_coords or 'dphi' not in model_coords:
                raise ValueError("Currently, the coordinate 'dR' is only supported when both 'deta' and 'dphi' are also included in the list of model coordinates.")
        if model_coord_dict =={} or model_coord_dict is None:
            self.model_coord_dict = {coord: i for i, coord in enumerate(model_coords)}
        else:
            self.model_coord_dict = model_coord_dict

    def forward(self, model, X, X_cartesian = None, jet_vec = None, mask=None, train=False, generators=None, take_mean=False, features_dict = {},X_log_cylindrical = None,jet_e=None,debug=False,output=None):
        """
        Computes the deltaSEAL for the given model and inputs.
        
        Parameters:
        -----------
        model : nn.Module
            The neural network model whose output symmetry is being enforced.
        
        X : torch.Tensor
            Input tensor to the model, shape [B, N, D] where B is the batch size, N is the number of data points, and D is the number of input features.
        
        X_cartesian : torch.Tensor
            Input data in Cartesian coordinates, shape [B, N, 4].
        
        jet_vec : torch.Tensor
            Jet 4-vectors in Cartesian coordinates, shape [B, 4].
        
        mask : torch.Tensor, optional
            Mask tensor to apply to the input, if needed.
        
        train : bool, optional
            Whether the forward pass is for training or evaluation.
        
        generators : list of bool, optional
            A list of 6 boolean values representing whether to enforce each of the 6 generators of Lorentz symmetry (boosts and rotations). Defaults to None.
        
        take_mean : bool, optional
            If True, the mean loss over the batch is returned. Otherwise, the per-sample loss is returned.
        
        Returns:
        --------
        torch.Tensor
            The computed loss, either as a per-sample tensor [B] or a single mean value.
        """
        st_symm = time.time()
        jet_e_sum_const = self.jet_e_sum_const
        
        if generators is not None:
            # if isinstance(generators, list):
            #     gens = [True]
            # Validate that 'generators' is a list of 6 boolean values
            if not isinstance(generators, list) or len(generators) != 6 or not all(isinstance(g, bool) for g in generators):
                warnings.warn("Invalid 'generators'. Expected a list of 6 boolean values. Continuing with 'generators=None'.")
                generators = None
            
        device = X.device

        # Build transformation dictionaries for the Cartesian inputs and jet vectors
        trans_coords = ["log_E","log_pT","eta","phi"]
        if ('log_1pE' in self.model_coord_dict):
            trans_coords.append('log_1pE')
        if ('log_1ppT' in self.model_coord_dict):
            trans_coords.append('log_1ppT')
        coords_cylindrical = [coo for coo in self.model_coord_dict if coo in ["E","pT","log_E","log_pT","eta","phi"]]

        st_trans_dict = time.time()
        trans_dict = self.get_trans_dict(cartesian_features =X_cartesian,log_cylindrical_features=X_log_cylindrical,features_dict = features_dict , coords = coords_cylindrical)
        end_trans_dict = time.time()
        time_trans_dict = end_trans_dict-st_trans_dict
        print_db(f"time_trans_dict = {time_trans_dict}",db=debug)

        st_mask = time.time()
        if jet_e_sum_const:
            #particles that are masked are actually non-existant, so will be boost-invariant.
            if mask is None:
                mask = torch.all(torch.abs(X)!=0,dim=-1)
            for key in trans_dict:
                trans_dict[key] = torch.einsum('b n k, b n -> b n k', trans_dict[key], mask)
        end_mask = time.time()
        time_mask = end_mask-st_mask
        print_db(f"time mask = {time_mask}",db=debug)

        st_trans_dict_jet = time.time()
        trans_dict_jet = {}
        if any(['R' in coo for coo in self.model_coord_dict]) or any(['d' in coo for coo in self.model_coord_dict]):
                trans_dict_jet = self.get_trans_dict(cartesian_features =jet_vec)
                if jet_e_sum_const:
                    if 'log_R_E' in self.model_coord_dict:
                        if mask is None:
                            mask = torch.all(torch.abs(X)!=0,dim=-1)
                        constituent_E = torch.exp(X[:,:,self.model_coord_dict['log_E']]) if 'log_E' in self.model_coord_dict else (X[:,:,self.model_coord_dict['E']] if 'E' in self.model_coord_dict else None)
                        
                        if constituent_E is None:
                            constituent_E= X_cartesian[:,:,0] if X_cartesian is not None else (torch.exp(X_log_cylindrical[:,:,0]) if X_log_cylindrical is not None else None)
                        
                        
                        if jet_e is None:
                            jet_e = torch.sum(constituent_E*mask ,dim=1)
                        if len(jet_e.shape)==1:
                            jet_e = jet_e.unsqueeze(1)
                        

                        trans_logE = torch.einsum('b n k, b n -> b n k', trans_dict['log_E'], mask)
                        trans_logE = torch.nan_to_num(trans_logE,nan = 0.0)
                        trans_dict_jet['log_E'] = torch.einsum('b n k, b n -> b k',trans_logE,constituent_E)#/jet_e
                        trans_dict_jet['log_E'] = (trans_dict_jet['log_E']/jet_e).unsqueeze(1)

                        
                    #particles that are masked are actually non-existant, so will be boost-invariant.
                    if mask is None:
                        mask = torch.all(torch.abs(X)!=0,dim=-1)
                    for key in trans_dict_jet:
                        trans_dict_jet[key] = torch.einsum('b n k, b n -> b n k', trans_dict_jet[key], mask)
                      
        end_trans_dict_jet = time.time()
        time_trans_dict_jet = end_trans_dict_jet-st_trans_dict_jet
        print_db(f"time_trans_dict_jet = {time_trans_dict_jet}",db=debug)
        


        # Prepare input by cloning, detaching, and enabling gradient tracking
        st_grads = time.time()
        if output is None:
            input = X.clone().detach().requires_grad_(True).to(device)
            output = model(input, mask)  # Evaluate the model output
        else:
            input = X
        # Compute gradients of the model output w.r.t. the input
        grads, = torch.autograd.grad(
            outputs=output, 
            inputs=input, 
            grad_outputs=torch.ones_like(output, device=device), 
            create_graph=train  # If train is True, allow gradient computation for training
        )
        del input, output
        end_grads = time.time()
        time_grads = end_grads - st_grads
        print_db(f"time_grads = {time_grads}",db=debug)

        with contextlib.nullcontext() if train else torch.no_grad():
            # if torch.is_grad_enabled():
            #     print("Symmloss: Gradients are enabled.")
            # else:
            #     print("Symmloss: Gradients are disabled (torch.no_grad() is active).")

            # Initialize the variation tensor (B, 6) to store the variations for each coordinate
            var_tensor = torch.zeros(X.shape[0], 6, device=device)
            
            # Loop over the model coordinates and compute the variation tensor for each one
            st_vars = time.time()
            for coord, i in self.model_coord_dict.items():
                var_tensor += self.compute_var_tensor(X, grads, trans_dict, trans_dict_jet, coord, i)
            end_vars = time.time()
            time_vars = end_vars - st_vars
           
            # Apply generators mask if provided
            if generators is not None:
                generators_tensor = torch.tensor(generators, dtype=torch.bool, device=device).unsqueeze(0)  # [1, 6]
                var_tensor = torch.where(generators_tensor, var_tensor, torch.zeros_like(var_tensor))

            # Compute the loss as the squared norm of the variation tensor
            loss = torch.norm(var_tensor, p=2, dim=1)**2

            # Optionally, return the mean loss
            if take_mean:
                loss = loss.mean()

            end_symm = time.time()
            
            return loss
        
    def compute_var_tensor(self, X, grads, trans_dict, trans_dict_jet, coord, coord_ind):
        """
        Computes the variation tensor for a given coordinate in the model's input.
        
        Parameters:
        -----------
        X : torch.Tensor
            Input tensor to the model, shape [B, N, D].
        
        grads : torch.Tensor
            Gradients of the model output w.r.t. the input, shape [B, N, len(model_coords)].
        
        trans_dict : dict
            Transformation dictionary for the input coordinates, providing the relevant transformations for each coordinate.
        
        trans_dict_jet : dict
            Transformation dictionary for the jet vectors, providing the relevant transformations for the jet-related coordinates.
        
        coord : str
            The coordinate name for which to compute the variation tensor.
        
        coord_ind : int
            The index of the coordinate in the input tensor.
        
        Returns:
        --------
        torch.Tensor
            The computed variation tensor for the given coordinate, shape [B, 6].
        """
        jet_e_sum_const = self.jet_e_sum_const
        # Variation tensor for logarithmic and angular coordinates (log_E, log_pT, eta, phi)
        if coord in {'log_E', 'log_pT', 'eta', 'phi','log_1pE','log_1ppT'}:
            return torch.einsum('b n, b n k -> b k', grads[:,:,coord_ind], trans_dict[coord])
        
        # Variation tensor for energy and transverse momentum (E, pT) derived from log_E and log_pT
        elif coord in {'E', 'pT'}:
            coord = f'log_{coord}'  # Use the log versions for these
            return torch.einsum('b n, b n k -> b k', grads[:,:,coord_ind]*X[:,:,coord_ind], trans_dict[coord])
        
        # Variation tensor for relative coordinates (log_R_pT, log_R_E, deta, dphi)
        elif coord in {'log_R_pT', 'log_R_E', 'deta', 'dphi'}:
            coord = coord.replace('R_', '').replace('d', '')
            
            return torch.einsum('b n, b n k -> b k', grads[:,:,coord_ind], trans_dict[coord] - trans_dict_jet[coord])
        
        # Special case for delta R (dR)
        elif coord == 'dR':
            K_tensor = (X[:,:,self.model_coord_dict['deta']].unsqueeze(-1) * (trans_dict['eta'] - trans_dict_jet['eta']))
            K_tensor += (X[:,:,self.model_coord_dict['dphi']].unsqueeze(-1) * (trans_dict['phi'] - trans_dict_jet['phi']))
            K_tensor /= X[:,:,self.model_coord_dict['dR']].unsqueeze(-1) + 1e-20#1e-10  # Avoid division by zero
            return torch.einsum('b n, b n k -> b k', grads[:,:,coord_ind], K_tensor)
        
        else:
            raise ValueError(f"Invalid coordinate '{coord}'.")
        
            
    def get_trans_dict(self, cartesian_features = None, log_cylindrical_features = None,features_dict ={} ,coords = ["log_E","log_pT","eta","phi"]):
        """
        Returns a transformation dictionary for the given cartesian_features.

        Parameters:
        -----------
        cartesian_features : torch.Tensor
            The input data in Cartesian coordinates, shape [B, N, 4].
        
        Returns:
        --------
        dict
            A dictionary mapping each coordinate to its corresponding transformation tensor, shape [B, N, 6].
        """
        if features_dict is not None and features_dict!={} :
          logE = features_dict['log_E'] if 'log_E' in features_dict else (torch.log(features_dict['E']) if 'E' in features_dict else None)
          logPt = features_dict['log_pT'] if 'log_pT' in features_dict else (torch.log(features_dict['pT']) if 'pT' in features_dict else None)
          eta = features_dict['eta'] if 'eta' in features_dict else None
          phi = features_dict['phi'] if 'phi' in features_dict else None

        elif log_cylindrical_features is not None:
            print(f"assuming the coordinates are in order [log_E, log_pT, eta, phi]")
            logE, logPt, eta, phi = log_cylindrical_features
            
        elif cartesian_features is not None:
            # Converts Cartesian coordinates to cylindrical and returns a transformation dict
            logE, logPt, eta, phi = self.to_cylindrical(cartesian_features, log=True).unbind(dim=2)
            

            
        # Intermediate quantities, shape [B, N]
        sin_phi = torch.sin(phi) 
        cos_phi = torch.cos(phi) 
        sinh_eta = torch.sinh(eta)
        cosh_eta = torch.cosh(eta)
        lamb = torch.exp(logE - logPt)
        zeros = torch.zeros_like(logE)
        ones = torch.ones_like(logE)
        
        # Create and return the transformation dictionary
        # Each key corresponds to a different coordinate
        # Each value is a tensor of shape [B, N, 6]
        trans_dict = {}
        
        trans_dict['log_E'] = torch.stack([zeros]*3 + [cos_phi/lamb, sin_phi/lamb, sinh_eta/lamb], dim=2)
        trans_dict['log_pT'] = torch.stack([sin_phi*sinh_eta, -cos_phi*sinh_eta, zeros, lamb*cos_phi, lamb*sin_phi, zeros], dim=2)
        trans_dict['eta'] = torch.stack([-cosh_eta*sin_phi, cosh_eta*cos_phi, zeros, -lamb*cos_phi*sinh_eta/cosh_eta, -lamb*sin_phi*sinh_eta/cosh_eta, lamb/cosh_eta], dim=2)
        trans_dict['phi'] = torch.stack([cos_phi*sinh_eta, sin_phi*sinh_eta, -ones, -lamb*sin_phi, lamb*cos_phi, zeros], dim=2)

        if "log_1pE" in coords:
            r_1pE = torch.stack([torch.exp(logE)/(torch.exp(logE)+1)]*6,dim=2)
            trans_dict['log_1pE'] = r_1pE*trans_dict['log_E']
        if "log_1ppT" in coords:
            r_1ppT = torch.stack([torch.exp(logPt)/(torch.exp(logPt)+1)]*6,dim=2)
            trans_dict['log_1ppT'] = r_1ppT*trans_dict['log_pT']
       
        return trans_dict
    

    def get_trans_dict_E(self, cartesian_features = None, cylindrical_features = None,features_dict ={} ,coords = ["log_E","log_pT","eta","phi"]):
        """
        Returns a transformation dictionary for the given cartesian_features.

        Parameters:
        -----------
        cartesian_features : torch.Tensor
            The input data in Cartesian coordinates, shape [B, N, 4].
        
        Returns:
        --------
        dict
            A dictionary mapping each coordinate to its corresponding transformation tensor, shape [B, N, 6].
        """
        if features_dict is not None and features_dict!={} :
          logE = features_dict['log_E'] if 'log_E' in features_dict else (torch.log(features_dict['E']) if 'E' in features_dict else None)
          logPt = features_dict['log_pT'] if 'log_pT' in features_dict else (torch.log(features_dict['pT']) if 'pT' in features_dict else None)
          eta = features_dict['eta'] if 'eta' in features_dict else None
          phi = features_dict['phi'] if 'phi' in features_dict else None

        elif cylindrical_features is not None:
            print(f"assuming the coordinates are in order [log_E, log_pT, eta, phi]")
            logE, logPt, eta, phi = cylindrical_features
            
        elif cartesian_features is not None:
            # Converts Cartesian coordinates to cylindrical and returns a transformation dict
            logE, logPt, eta, phi = self.to_cylindrical(cartesian_features, log=True).unbind(dim=2)
        
        
        # Intermediate quantities, shape [B, N]
        sin_phi = torch.sin(phi) 
        cos_phi = torch.cos(phi) 
        sinh_eta = torch.sinh(eta)
        cosh_eta = torch.cosh(eta)
        lamb = torch.exp(logE - logPt)
        zeros = torch.zeros_like(logE)
        ones = torch.ones_like(logE)
        
        # Create and return the transformation dictionary
        # Each key corresponds to a different coordinate
        # Each value is a tensor of shape [B, N, 6]
        trans_dict = {}
        
        trans_dict['log_E'] = torch.stack([zeros]*3 + [cos_phi/lamb, sin_phi/lamb, sinh_eta/lamb], dim=2)
        trans_dict['log_pT'] = torch.stack([sin_phi*sinh_eta, -cos_phi*sinh_eta, zeros, lamb*cos_phi, lamb*sin_phi, zeros], dim=2)
        trans_dict['eta'] = torch.stack([-cosh_eta*sin_phi, cosh_eta*cos_phi, zeros, -lamb*cos_phi*sinh_eta/cosh_eta, -lamb*sin_phi*sinh_eta/cosh_eta, lamb/cosh_eta], dim=2)
        trans_dict['phi'] = torch.stack([cos_phi*sinh_eta, sin_phi*sinh_eta, -ones, -lamb*sin_phi, lamb*cos_phi, zeros], dim=2)

        if "log_1pE" in coords:
            r_1pE = torch.stack([torch.exp(logE)/(torch.exp(logE)+1)]*6,dim=2)
            trans_dict['log_1pE'] = r_1pE*trans_dict['log_E']
        if "log_1ppT" in coords:
            r_1ppT = torch.stack([torch.exp(logPt)/(torch.exp(logPt)+1)]*6,dim=2)
            trans_dict['log_1ppT'] = r_1ppT*trans_dict['log_pT']
       
        return trans_dict

    def to_cylindrical(self, four_vec, log=True):
        """
        Converts 4-vectors from Cartesian coordinates to cylindrical coordinates.

        Parameters:
        -----------
        four_vec : torch.Tensor
            Input tensor containing 4-vectors in Cartesian coordinates, shape [B, N, 4].
        
        log : bool, optional
            If True, the logarithms of the energy (E) and transverse momentum (pT) will be returned.
        
        Returns:
        --------
        torch.Tensor
            The converted 4-vectors in cylindrical coordinates, shape [B, N, 4].
        """
        # Extract energy and momentum components from the 4-vector
        E = four_vec[:,:,0]
        px = four_vec[:,:,1]
        py = four_vec[:,:,2]
        pz = four_vec[:,:,3]
        
        # Compute transverse momentum, azimuthal angle (phi), and pseudorapidity (eta)
        pt = torch.sqrt(px*px + py*py)
        phi = torch.arctan2(py, px)
        eta = torch.arcsinh(pz/pt)

        if log:
            # Logarithmic version: take the log of energy and transverse momentum
            cylindrical_four_vec = torch.cat([
                torch.log(E.unsqueeze(-1)),
                torch.log(pt.unsqueeze(-1)), 
                eta.unsqueeze(-1),
                phi.unsqueeze(-1)
            ], axis=2)

            # Handle very small values (avoid -inf or NaN issues)
            cylindrical_four_vec = torch.where(cylindrical_four_vec < -1e30, 0, cylindrical_four_vec)
        else:
            # Non-logarithmic version
            cylindrical_four_vec = torch.cat([E.unsqueeze(-1), pt.unsqueeze(-1), eta.unsqueeze(-1), phi.unsqueeze(-1)], axis=2)

        # Replace NaN values with zeros for stability
        return torch.nan_to_num(cylindrical_four_vec)
    


def deltaSEAL_scalar7(model, X, X_cartesian, jet_vec, mask=None, train=False, generators=None, take_mean = False,jet_e_sum_const = False,jet_e=None,debug=False,output=None):
    # use deltaSEAL with 7 ATLAS coordinates
    deltaSEAL_loss = deltaSEAL('ATLAS',jet_e_sum_const= jet_e_sum_const)
    # if train:
    #     print("deltaSEAL: Training mode")
    # else:
    #     print("deltaSEAL: Evaluation mode")
    return deltaSEAL_loss(model, X, X_cartesian, jet_vec, mask=mask, train=train, generators=generators, take_mean=take_mean,jet_e=jet_e,debug=debug,output=output)

