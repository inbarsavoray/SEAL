import torch
import torch.nn as nn
import contextlib
import warnings
import numpy as np
import einops



######################################################## GSEAL #########################################################
def levi_civita_3d():
    e = torch.zeros((3, 3, 3), dtype=torch.float32)
    e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
    e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
    return e

def boost_3d(data, device="cpu", beta=None,beta_vec = None, beta_2_max = 0.95,dtype=torch.float32):
    # print(f"boosting data type {data.dtype}")
    # sample beta from sphere
    print(f"data shape = {data.shape}")
    if beta is not None:
        bf = beta*torch.ones(size=len(data), dtype=dtype)
    else:
        bf = torch.tensor(np.random.uniform(0, beta_2_max, size=len(data)), dtype=dtype)
        bf = bf**(1/3)

    if beta_vec is not None:
        
        beta_fixed_dir = torch.tensor(beta_vec, dtype=dtype)
        beta_x = beta_fixed_dir[0].type(dtype)*torch.ones(len(data), dtype=dtype)
        beta_y = beta_fixed_dir[1].type(dtype)*torch.ones(len(data), dtype=dtype)
        beta_z = beta_fixed_dir[2].type(dtype)*torch.ones(len(data), dtype=dtype)

    else:
        b1 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=dtype)
        b2 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=dtype)
        theta = 2 * np.pi * b1
        phi = np.arccos(1 - 2 * b2)
        
        beta_x = np.sin(phi) * np.cos(theta)
        beta_y = np.sin(phi) * np.sin(theta)
        beta_z = np.cos(phi)
    
    beta= torch.stack([bf*beta_x, bf*beta_y, bf*beta_z], dim=-1)
    
    beta_norm = torch.norm(beta, dim=-1)
    

    # make sure we arent violating speed of light
    assert torch.all(beta_norm < 1)

    gamma = 1 / torch.sqrt(1 - (beta_norm)**2)

    beta_squared = (beta_norm)**2

    # make boost matrix
   
    L = torch.zeros((len(data),4, 4), dtype=data.dtype, device=device)
    

    L[...,0, 0] = gamma
    L[...,1:, 0] = L[...,0, 1:] = -gamma.unsqueeze(-1) * beta
    L[..., 1:, 1:] = torch.eye(3) + (gamma[...,None, None] - 1) * torch.einsum('...i,...j->...ij', (beta, beta))/ beta_squared[...,None, None]

    
    assert torch.all (torch.linalg.det(L)) == True

    boosted_four_vector = torch.einsum('nij,n...j->n...i', L.type(data.dtype), data.type(data.dtype))
    

    # Validate that energy values remain non-negative
    # assert torch.all(boosted_four_vector[..., 0] >= 0), "Negative energy values detected in constituents!"
    
    return boosted_four_vector


def rotation_3d(data, device="cpu", theta_vec=None, theta_rot = None,theta_mag_max = 2*np.pi,theta_dir_fix = False,dtype=torch.float32,theta_fixed = 0.0,abs_fix=False):
    
    if theta_vec is not None:
        u = theta_vec
    else:
        # sample rotation axis from sphere
        b1 = torch.tensor(np.random.uniform(0, 1, size=(len(data))), dtype=data.dtype)
        b2 = torch.tensor(np.random.uniform(0, 1, size=(len(data))), dtype=data.dtype)
        theta = 2 * torch.pi * b1
        phi = torch.arccos(1 - 2 * b2)
        
        
        u_x = torch.sin(phi) * torch.cos(theta)
        u_y = torch.sin(phi) * torch.sin(theta)
        u_z = torch.cos(phi)
        u = torch.stack([u_x, u_y, u_z], dim=-1)

    if theta_rot is not None:
        theta_rot = theta_rot
    else:   
        #sample rotation angle around axis
        b3 = torch.tensor(np.random.uniform(0, 1, size=(len(data))), dtype=data.dtype)
        theta_rot = theta_mag_max * b3
 
    
    # make rotation matrix
    L = torch.zeros((len(data),4, 4)).to(device)
    L[...,0, 0] = 1
    L[...,1:, 0] = 0
    L[...,0, 1:] = 0

    eye_matrix = torch.eye(3).unsqueeze(0).unsqueeze(0)  # Expand dimensions to match L[:, 1:, 1:]
    cos_theta_rot = torch.cos(theta_rot[..., None, None])
    sin_theta_rot = torch.sin(theta_rot[..., None, None])
    uxu = torch.einsum('...i,...j->...ij', (u, u))

    eps = levi_civita_3d().type(data.dtype)
    ucross = torch.einsum('ijk,...k->...ij', eps, u)#maybe there's a minus here, wasn't too careful about that.
  # Expand dimensions to match L[:, 1:, 1:]

    L[..., 1:, 1:] = eye_matrix * cos_theta_rot + (1 - cos_theta_rot) * uxu + sin_theta_rot * ucross

    rotated_four_vector = torch.einsum('nij,n...j->n...i', L.type(data.dtype), data.type(data.dtype)).to(device)

   
    return rotated_four_vector


class GSEAL(nn.Module):

    def __init__(self,rotate=True,boost=True,device='cpu'):
        super(GSEAL, self).__init__()
        self.rotate = rotate
        self.boost = boost
        self.GSEAL_loss_mean = 0.0
        self.GSEAL_loss = torch.nn.MSELoss()
        self.device = device

    def transform(self,particles,preprocessed=False,mean=None,std=None,rotate=None,boost=None,beta=None,beta_vec =None,theta=None,theta_vec=None):
        data = particles.detach().clone().to(self.device)
        
        if rotate is None:
            rotate = self.rotate
        if boost is None:
            boost = self.boost

        if preprocessed:
            mean = torch.tensor(mean, device=data.device, dtype=data.dtype) if not torch.is_tensor(mean) else mean
            std = torch.tensor(std, device=data.device, dtype=data.dtype) if not torch.is_tensor(std) else std
            data = data * std + mean
           
        
        data_transformed = data
        if boost:
            data_transformed = boost_3d(data,device = data.device,beta=beta,beta_vec=beta_vec,dtype=data.dtype)
        if rotate:
            data_transformed = rotation_3d(data_transformed, device = data.device,theta_rot = theta, theta_vec=theta_vec,dtype=data.dtype)

        if preprocessed:
            data_transformed = (data_transformed - mean) / std
        


        return data_transformed
        
    def forward(self,model,x,*model_inv_args,pred=None,preprocessed=False,mean=None,std=None,rotate=None,boost=None,take_mean=False,theta=None,beta=None,beta_vec=None,theta_vec = None,**model_inv_kwargs):   
        if rotate is None:
            rotate = self.rotate
        if boost is None:
            boost = self.boost
       
        model = model.to(self.device)
        x = x.to(self.device)

        if (mean is not None) and (len(mean)>1):
            if len(x) != len(mean):
                raise ValueError("Number of means should be equal to number of datasets")
            for i in range(len(mean)):
                x_transformed[i] = self.transform(x[i],preprocessed=preprocessed,mean=mean[i],std=std[i],rotate=rotate,boost=boost,beta=beta,beta_vec=beta_vec,theta=theta,theta_vec=theta_vec)
        else:
            x_transformed = self.transform(x,preprocessed=preprocessed,mean=mean,std=std,rotate=rotate,boost=boost)

        pred_transformed = model(x_transformed, *model_inv_args,**model_inv_kwargs)

        if pred is None:
            pred = model(x, *model_inv_args,**model_inv_kwargs)

        loss = self.GSEAL_loss(pred_transformed, pred)
        if take_mean:
            loss = loss.mean()

        return loss

######################################################## delta SEAL #########################################################

def SO3_gens(dtype = torch.float32):
    Lz = torch.tensor([[0,1,0],[-1,0,0],[0,0,0]],dtype = dtype)
    Ly = torch.tensor([[0,0,-1],[0,0,0],[1,0,0]],dtype = dtype)
    Lx = torch.tensor([[0,0,0],[0,0,1],[0,-1,0]],dtype = dtype)
    gens = torch.stack([Lx, Ly, Lz])
    return gens


def SO2_gens(dtype = torch.float32):
    Lz = torch.tensor([[0,1],[-1,0]],dtype = dtype)
    gens = torch.stack([Lz])
    return gens



def Lorentz_gens(dtype = torch.float32):

    #p^a= p[a] -> Jz[a,b]p[b] (p_a = p[a] -> Jz[a,b]p[b])
    Lz = torch.tensor([[0,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,0]],dtype = dtype)
    Ly = torch.tensor([[0,0,0,0],[0,0,0,-1],[0,0,0,0],[0,1,0,0]],dtype = dtype)
    Lx = torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,-1,0]],dtype = dtype)


    #p^a = p[a] -> Kz[a,b]p[b] (p_a = p[a] -> -Kz[a,b]p[b])
    Kz = torch.tensor([[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]],dtype = dtype)
    Ky = torch.tensor([[0,0,1,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]],dtype = dtype)
    Kx = torch.tensor([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]],dtype = dtype)

    gens = torch.stack([Kx, Ky, Kz, Lx, Ly, Lz])
    return gens


gens_SO2 = SO2_gens()

gens_SO3 = SO3_gens()#einops.rearrange(SO3_gens(),'n h w -> n h w')


gens_Lorentz = Lorentz_gens()#einops.rearrange(Lorentz_gens(),'n h w -> n h w')


class deltaSEAL(nn.Module):

    def __init__(self, gens_list_input,device = 'cpu',gens_list_output = None):
        super(deltaSEAL, self).__init__()
        
        
        self.device = device
        # Initialize generators acting as matrices on input
        self.generators_input = gens_list_input.to(device)
         # Initialize generators acting as matrices on output
        self.generators_output = gens_list_output.to(device) if gens_list_output is not None else None
        
        

    
    def forward(self,model, input, pred=None,train = False,take_mean = False):
        #Assuming input is a vector
        # if input.device() != self.device:
        input = input.to(self.device)
    #   if model.device() != self.device:
        model = model.to(self.device)
        
        # Compute model output, shape [B]
        if pred is None:
            input = input.clone().detach().requires_grad_(True)
            input = input.to(self.device)
            pred = model(input)

        # Compute gradients with respect to input, shape [B, d*N], B is the batch size, d is the input irrep dimension, N is the number of particles
        grads, = torch.autograd.grad(outputs=pred, inputs=input, grad_outputs=torch.ones_like(pred, device=self.device), create_graph=train)
        N_parts=1
        # Reshape grads to [B, N, d] 
        #here we assume that the input feature dimension is divisible by the generator dimension
        if input.shape[-1] % self.generators_input.shape[-1] != 0:
            raise ValueError(f"Input feature dimension {input.shape[-1]} is not compatible with generator dimension {self.generators_input.shape[-1]}.")
        elif input.shape[-1] != self.generators_input.shape[-1] != 1:
            N_parts = input.shape[-1] // self.generators_input.shape[-1]
            grads = einops.rearrange(grads, '... (N d) -> ... N d',d = self.generators_input.shape[-1])
    

        # Contract grads with generators, shape [n (generators), B, N, d]
        gen_grads = torch.einsum('n h d, ... N h->  n ... N d ',self.generators_input, grads)
        # Reshape to [n, B, (d N)]
        if N_parts!=1:
            gen_grads = einops.rearrange(gen_grads, 'n ... N d -> n ... (N d)')

        # Dot with input [n ,B]
        differential_trans = torch.einsum('n ... N, ... N -> n ...', gen_grads, input)

        if self.generators_output is not None:
            if pred.shape[-1] % self.generators_output.shape[-1] != 0:
                raise ValueError(f"Output feature dimension {pred.shape[-1]} is not compatible with generator dimension {self.generators_output.shape[-1]}.")
            elif pred.shape[-1] != self.generators_output.shape[-1] != 1:
                N_parts_out = pred.shape[-1] // self.generators_output.shape[-1]
                pred_reshaped = einops.rearrange(pred, '... (N d) -> ... N d',d = self.generators_output.shape[-1])
            else:
                pred_reshaped = pred.clone().detach()

            gen_pred = torch.einsum('n h d, ... N h->  n ... N d ',self.generators_output, pred_reshaped)
            if N_parts_out!=1:
                gen_pred = einops.rearrange(gen_pred, 'n ... N d -> n ... (N d)')
            
            # Compute loss
            loss = ((gen_pred-differential_trans) ** 2).mean(dim=0)  # average over generators
            

        # Compute loss
        else:
            loss = (differential_trans ** 2).mean(dim=0)  # average over generators

        if take_mean:
            loss = loss.mean()#also average over batch
     
            
        return loss
        

