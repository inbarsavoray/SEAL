import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
import einops

import numpy as np
import matplotlib.pyplot as plt
import copy
import dill as pickle
import os
import time

from tqdm import tqdm

def SO3_gens():
    Lz = torch.tensor([[0,1,0],[-1,0,0],[0,0,0]],dtype = torch.float32)
    Ly = torch.tensor([[0,0,-1],[0,0,0],[1,0,0]],dtype = torch.float32)
    Lx = torch.tensor([[0,0,0],[0,0,1],[0,-1,0]],dtype = torch.float32)
    gens = [Lx, Ly, Lz]
    return gens


def Lorentz_gens():

    #p^a= p[a] -> Jz[a,b]p[b] (p_a = p[a] -> Jz[a,b]p[b])
    Lz = torch.tensor([[0,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,0]],dtype = torch.float32)
    Ly = torch.tensor([[0,0,0,0],[0,0,0,-1],[0,0,0,0],[0,1,0,0]],dtype = torch.float32)
    Lx = torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,1],[0,0,-1,0]],dtype = torch.float32)


    #p^a = p[a] -> Kz[a,b]p[b] (p_a = p[a] -> -Kz[a,b]p[b])
    Kz = torch.tensor([[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]],dtype = torch.float32)
    Ky = torch.tensor([[0,0,1,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]],dtype = torch.float32)
    Kx = torch.tensor([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]],dtype = torch.float32)

    gens = [Kx, Ky, Kz, Lx, Ly, Lz]
    return gens


gens_SO3 = einops.rearrange(SO3_gens(),'n h w -> n h w')


gens_Lorentz = einops.rearrange(Lorentz_gens(),'n h w -> n h w')


devicef = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {devicef} device")

def boost_3d(data, device="cpu", beta=None, beta_max=0.95,d=4):
    
    # sample beta from sphere
    b1 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    b2 = torch.tensor(np.random.uniform(0, 1, size=len(data)), dtype=torch.float32)
    theta = 2 * np.pi * b1
    phi = np.arccos(1 - 2 * b2)
    
    beta_x = np.sin(phi) * np.cos(theta)
    beta_y = np.sin(phi) * np.sin(theta)
    beta_z = np.cos(phi)
    
    beta = torch.cat([beta_x.unsqueeze(-1),beta_y.unsqueeze(-1), beta_z.unsqueeze(-1)], axis=1)
    bf = torch.tensor(np.random.uniform(0, beta_max, size=(len(data),1)), dtype=torch.float32)
    bf = bf#**(1/2)
    beta = beta*bf#beta*beta_max
    
    beta_norm = torch.norm(beta, dim=1) 

    # make sure we arent violating speed of light
    #assert torch.all(beta_norm < 1)

    gamma = 1 / torch.sqrt(1 - (beta_norm)**2)

    beta_squared = (beta_norm)**2

    # make boost matrix
    L = torch.zeros((len(data), 4, 4)).to(device)
    L[:,0, 0] = gamma
    L[:,1:, 0] = L[:,0, 1:] = -gamma.unsqueeze(-1) * beta
    L[:, 1:, 1:] = torch.eye(3) + (gamma[...,None, None] - 1) * torch.einsum('bi,bj->bij', (beta, beta))/ beta_squared[...,None, None]
    
    assert torch.all (torch.linalg.det(L)) == True
    
    flag =False
    if data.shape[-1]!=d or len(data.shape)<3:
        flag = True
        data = einops.rearrange(data, '... (N d) -> ... N d',d = d)
        
    boosted_four_vector = torch.einsum('bij,bkj->bki', L.type(torch.float32), data.type(torch.float32))#.permute(0, 2, 1) 

    # Validate that energy values remain non-negative
    #assert torch.all(boosted_four_vector[:, :, 0] >= 0), "Negative energy values detected!"
    
    if flag:
        boosted_four_vector = einops.rearrange(boosted_four_vector, '... N d -> ... (N d)',d = d)
    
    return boosted_four_vector



def boost_3d_fix(data, device="cpu", beta_max=0.95,d=4,beta_fix = None, beta_dir_fix = [0,0,1],abs_fix = False, dir_fix = False,dtype=torch.float32):
    
    # sample beta from sphere
    if abs_fix and beta_fix is not None:
        bf = beta_fix*torch.ones(size=(len(data),1), dtype=dtype)
        bf = bf
    else:
        bf = torch.tensor(np.random.uniform(0, beta_max, size=(len(data),1)), dtype=dtype)
        bf = bf**(1/2)

    if dir_fix and beta_dir_fix is not None:
        beta_fixed_dir = torch.tensor(beta_dir_fix, dtype=dtype)
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
        
    beta = torch.cat([beta_x.unsqueeze(-1),beta_y.unsqueeze(-1), beta_z.unsqueeze(-1)], axis=1)
    
    beta = beta*bf#beta*beta_max
    # print(f"beta = {beta}")
    beta_norm = torch.norm(beta, dim=1) 

    # make sure we arent violating speed of light
    #assert torch.all(beta_norm < 1)

    gamma = 1 / torch.sqrt(1 - (beta_norm)**2)

    beta_squared = (beta_norm)**2

    # make boost matrix
    L = torch.zeros((len(data), 4, 4)).to(device)
    L[:,0, 0] = gamma
    L[:,1:, 0] = L[:,0, 1:] = -gamma.unsqueeze(-1) * beta
    L[:, 1:, 1:] = torch.eye(3) + (gamma[...,None, None] - 1) * torch.einsum('bi,bj->bij', (beta, beta))/ beta_squared[...,None, None]
    
    assert torch.all (torch.linalg.det(L)) == True
    
    flag =False
    if data.shape[-1]!=d or len(data.shape)<3:
        flag = True
        data = einops.rearrange(data, '... (N d) -> ... N d',d = d)
        
    boosted_four_vector = torch.einsum('bij,bkj->bki', L.type(dtype).to(device), data.type(dtype).to(device))#.permute(0, 2, 1) 

    # Validate that energy values remain non-negative
    #assert torch.all(boosted_four_vector[:, :, 0] >= 0), "Negative energy values detected!"
    
    if flag:
        boosted_four_vector = einops.rearrange(boosted_four_vector, '... N d -> ... (N d)',d = d)
    
    return boosted_four_vector


def Lorentz_myfun(input):
        m2 = torch.einsum("... i, ij, ...j -> ...",input, torch.diag(torch.tensor([1.00,-1.00,-1.00,-1.00])), input)
        out = m2**2+15*m2
        return out.unsqueeze(1)

def Lorentz_myfun_broken(input,spurions = [torch.tensor([0.0,0.0,0.0,0.0])]):
    metric_tensor = torch.diag(torch.tensor([1.00,-1.00,-1.00,-1.00]))
    m2 = torch.einsum("... i, ij, ...j -> ...",input, metric_tensor, input)
    breaking_scalars = [torch.einsum("... i, ij, ...j -> ...",spurion, metric_tensor, input) for spurion in spurions]
    coeffs = [20.9983, -23.2444, 3.0459, 12.7176, -17.4378, 1.4378, 10.1877,15.8890, -11.5178,  -4.3926]
    coeffs_2 = [-0.8431,   5.7529,  19.0048,   3.2927, -14.9460,   5.6997,  -5.9202, -10.5052, 2.6883, 16.5809]
    symm_out = m2**2+15*m2
    out = symm_out
    for i in range(len(breaking_scalars)):
        out += coeffs[i%len(coeffs)]*breaking_scalars[i]+coeffs_2[i%len(coeffs_2)]*breaking_scalars[i]**2
    return out.unsqueeze(1).to(devicef)




class deltaSEAL(nn.Module):

    def __init__(self, gens_list,model,device = devicef):
        super(deltaSEAL, self).__init__()
        
        self.model = model.to(device)
        self.device = device
        # Initialize generators (in future add different reps for inputs?)
        self.generators = einops.rearrange(gens_list, 'n w h -> n w h')
        self.generators = self.generators.to(device)
        

    
    def forward(self, input, model_rep='scalar',norm = "none"):
        
        input = input.clone().detach().requires_grad_(True)
        input = input.to(self.device)
        # Compute model output, shape [B]
        output = self.model(input)

        # Compute gradients with respect to input, shape [B, d*N], B is the batch size, d is the input irrep dimension, N is the number of particles
        grads, = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones_like(output, device=self.device), create_graph=True)
        
        # Reshape grads to [B, N, d] 
        grads = einops.rearrange(grads, '... (N d) -> ... N d',d = self.generators.shape[-1])

        # Contract grads with generators, shape [n (generators), B, N, d]
        gen_grads = torch.einsum('n h d, ... N h->  n ... N d ',self.generators, grads)
        # Reshape to [n, B, (d N)]
        gen_grads = einops.rearrange(gen_grads, 'n ... N d -> n ... (N d)')

        # Dot with input [n ,B]
        differential_trans = torch.einsum('n ... N, ... N -> n ...', gen_grads, input)

        
        # # Reshape grads to [B, N, d] 
        # grads = einops.rearrange(grads, '(N d) -> N d',d = self.generators.shape[-1])

        # # Contract grads with generators, shape [n (generators), B, N, d]
        # gen_grads = torch.einsum('n h d,  N h->  n N d ',self.generators, grads)
        # # Reshape to [n, B, (d N)]
        # gen_grads = einops.rearrange(gen_grads, 'n N d -> n (N d)')

        # # Dot with input [n ,B]
        # differential_trans = torch.einsum('n N, N -> n', gen_grads, input)

        scalar_loss = (differential_trans ** 2).mean()
     
            
        return scalar_loss
    
    


class deltaSEAL_norm(nn.Module):

    def __init__(self, gens_list,model,device = devicef):
        super(deltaSEAL_norm, self).__init__()
        
        self.model = model.to(device)
        self.device = device
        # Initialize generators (in future add different reps for inputs?)
        self.generators = einops.rearrange(gens_list, 'n w h -> n w h')
        self.generators = self.generators.to(device)
        

    
    def forward(self, input, model_rep='scalar',norm = "none"):
        
        input = input.clone().detach().requires_grad_(True)
        input = input.to(self.device)
        # Compute model output, shape [B]
        output = self.model(input)

        # Compute gradients with respect to input, shape [B, d*N], B is the batch size, d is the input irrep dimension, N is the number of particles
        grads, = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones_like(output, device=self.device), create_graph=True)
        
        grads_norm = torch.einsum('... N, ... N -> ...', grads, grads)
        #print(grads_norm.mean())
        
        # Reshape grads to [B, N, d] 
        grads = einops.rearrange(grads, '... (N d) -> ... N d',d = self.generators.shape[-1])

        # Contract grads with generators, shape [n (generators), B, N, d]
        gen_grads = torch.einsum('n h d, ... N h->  n ... N d ',self.generators, grads)
        # Reshape to [n, B, (d N)]
        gen_grads = einops.rearrange(gen_grads, 'n ... N d -> n ... (N d)')

        # Dot with input [n ,B]
        differential_trans = torch.einsum('n ... N, ... N -> n ...', gen_grads, input)
        
       
        
        
        
        scalar_loss_norm = (1/len(self.generators))*(torch.sum(differential_trans**2,dim = 0)/grads_norm).mean()
     
            
        return scalar_loss_norm




class inv_model(nn.Module):

    def __init__(self,dinput = 4, doutput = 1,init = "rand"):
        super(inv_model,self).__init__()

        bi_tensor = torch.randn(dinput,dinput)

        if init=="eta":
            diag = torch.ones(dinput)*(-1.00)
            diag[0]=1.00
            bi_tensor = torch.diag(diag)

        elif init=="delta":
            bi_tensor = torch.diag(torch.ones(dinput)*1.00)
        
        
        bi_tensor = (bi_tensor+torch.transpose(bi_tensor,0,1))*0.5
        self.bi_tensor = torch.nn.Parameter(bi_tensor)
        self.bi_tensor.requires_grad_()

    def forward(self,x, sig = "euc", d = 3 ):
        #y = x @ (self.bi_tensor @ x.T)
        y = torch.einsum("...i,ij,...j-> ...",x,self.bi_tensor,x)
        # x = torch.dot(x,y)
        return y

    
    
class broken_model(nn.Module):
    
    def __init__(self,dinput = 4, doutput = 1,init = "rand",spurions = [torch.tensor([0.0,0.0,0.0,0.0])]):
        super(broken_model,self).__init__()

        bi_tensor = torch.randn(dinput,dinput)

        if init=="eta":
            diag = torch.ones(dinput)*(-1.00)
            diag[0]=1.00
            bi_tensor = torch.diag(diag)

        elif init=="delta":
            bi_tensor = torch.diag(torch.ones(dinput)*1.00)
        
        
        bi_tensor = (bi_tensor+torch.transpose(bi_tensor,0,1))*0.5
        self.bi_tensor = torch.nn.Parameter(bi_tensor)
        self.bi_tensor.requires_grad_()
        self.spurions = spurions
        self.coeffs = [20.9983, -23.2444, 3.0459, 12.7176, -17.4378, 1.4378, 10.1877,15.8890, -11.5178,  -4.3926]
        self.coeffs_2 = [-0.8431,   5.7529,  19.0048,   3.2927, -14.9460,   5.6997,  -5.9202, -10.5052, 2.6883, 16.5809]
        
    def forward(self,x, sig = "euc", d = 3 ):
        #y = x @ (self.bi_tensor @ x.T)
        m2 = torch.einsum("...i,ij,...j-> ...",x,self.bi_tensor,x)
        breaking_scalars = [torch.einsum("... i, ij, ...j -> ...",spurion.to(devicef), self.bi_tensor, x) for spurion in self.spurions]
        
        symm_out = m2**2+15*m2
        out = symm_out
        for i in range(len(breaking_scalars)):
            out += self.coeffs[i%len(self.coeffs)]*breaking_scalars[i]+self.coeffs_2[i%len(self.coeffs_2)]*breaking_scalars[i]**2
            # x = torch.dot(x,y)
        return out


class genNet(nn.Module):
    def __init__(self, input_size=4, output_size=1, hidden_size=10, n_hidden_layers=3, init="rand", equiv="False", rand = "True", freeze = "False",activation = "ReLU", skip = "False", seed = 98235):
        super().__init__()
        self.equiv=equiv
        self.skip = skip
        if rand=="False":
            np.random.seed(seed)
            torch.manual_seed(seed)

        if activation =="sigmoid":
            #input layer
            module_list = [nn.Linear(input_size, hidden_size), nn.Sigmoid()]
            # hidden layers
            for _ in range(n_hidden_layers):
                module_list.extend([nn.Linear(hidden_size, hidden_size), nn.Sigmoid()])
        elif activation== "GeLU":
            module_list = [nn.Linear(input_size, hidden_size), nn.GELU()]
            # hidden layers
            for _ in range(n_hidden_layers):
                module_list.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])    

        else:
            #input layer
            module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
            # hidden layers
            for _ in range(n_hidden_layers):
                module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

        dinput = input_size
        if self.equiv=="True":
            bi_tensor = torch.randn(input_size,input_size)

            if init=="eta":
                diag = torch.ones(dinput)*(-1.00)
                diag[0]=1.00
                bi_tensor = torch.diag(diag)

            elif init=="delta":
                bi_tensor = torch.diag(torch.ones(dinput)*1.00)
        
            bi_tensor = (bi_tensor+torch.transpose(bi_tensor,0,1))*0.5
            self.bi_tensor = bi_tensor.to(devicef)
            
            if (freeze =="False" or freeze == False):
                
                self.bi_tensor = torch.nn.Parameter(bi_tensor.to(devicef))
                self.bi_tensor.requires_grad_()

            self.equiv_layer = nn.Linear(1, input_size)
            self.skip_layer = nn.Linear(input_size, input_size)
            

    def forward(self,x):

        if self.equiv=="True":
            y = torch.einsum("...i,ij,...j-> ...",x,self.bi_tensor,x).unsqueeze(1)
            
            y = self.equiv_layer(y)
            
            if self.skip =="True":
                y = y + self.skip_layer(x)
        else:
            y = x

        return self.sequential(y)
    
    
class toy_data():
    
    def get_dataset(self, N = 1000, dinput = 4, norm = 1, true_func = Lorentz_myfun, batch_size="all", shuffle=False, seed = 98235, broken_symm = "False", spurions = [[0.0,0.0,0.0,0.0]],input_spurions = "False"):

        self.N = N
        self.dataset_seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_data = (torch.rand(N,dinput)-0.5)*norm

        self.broken_symm = broken_symm
        self.spurions = [torch.tensor(spurion) for spurion in spurions]
        self.spurions_list = spurions
        self.true_func = lambda input: true_func(input,spurions = self.spurions) if (broken_symm == "True" or broken_symm == True) else true_func
        self.input_spurions = input_spurions


        train_labels = self.true_func(train_data).squeeze() #if (broken_symm == "True" or broken_symm == True) else true_func(train_data).squeeze()

        if batch_size=="all":
            batch_size = N

        if self.input_spurions == "True" or self.input_spurions==True:
            expand_spurions = (torch.cat(self.spurions)).expand(N,dinput)
            train_data = torch.cat((train_data,expand_spurions),dim = -1)
            self.input_size = train_data.shape[-1]


        train_dataset = TensorDataset(train_data,train_labels)
        self.train_dataset = train_dataset
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        

class symm_net_train():

    def __init__(self,gens_list=gens_Lorentz,input_size = 4,init = "rand",equiv="False",rand="False",freeze = "False", activation = "ReLU", skip="False",broken_symm = "False",spurions = torch.diag(torch.tensor([0.00,0.00,0.00,0.00]))):
        self.train_loss_vec = []
        self.dSEAL_vec = []
        self.tot_loss_vec = []
        self.running_loss = 0.0
        self.dSEAL = 0.0
        self.train_loss_lam = {}
        self.dSEAL_lam = {}
        self.tot_loss_lam = {}
        self.models = {}
        self.models_best_tot = {}
        self.models_best_dSEAL = {}
        self.models_best_GSEAL = {}

        self.init = init
        self.equiv = equiv
        self.rand=rand
        self.freeze = freeze, 
        self.activation = activation
        self.skip=skip
        self.input_size = input_size
        self.gens_list=gens_list
        
        self.dataset_seed = 0
        self.train_seed = 0
        self.N = 0
        
        self.broken_symm = broken_symm
        self.spurions = spurions

#     

    def prepare_dataset(self, N = 1000, dinput = 4, norm = 1, true_func = Lorentz_myfun, batch_size="all", shuffle=False, seed = 98235, broken_symm = "False", spurions = [[0.0,0.0,0.0,0.0]],input_spurions = "False"):
        
        self.N = N
        self.dinput = dinput
        self.norm = norm
        self.dataset_seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        np.random.seed(seed)
        torch.manual_seed(seed)
        
      
        
        self.broken_symm = broken_symm
        self.spurions = [torch.tensor(spurion) for spurion in spurions]
        self.spurions_list = spurions
        self.true_func = lambda input: true_func(input,spurions = self.spurions) if (broken_symm == "True" or broken_symm == True) else true_func
        self.input_spurions = input_spurions
        

        if self.input_spurions == "True" or self.input_spurions==True:
            expand_spurions = (torch.cat(self.spurions)).expand(N,dinput)
            train_data = torch.cat((train_data,expand_spurions),dim = -1)
            self.input_size = train_data.shape[-1]
        
        data = toy_data()
        data.get_dataset(N = self.N, dinput = self.dinput, norm = self.norm, true_func = true_func, batch_size=self.batch_size, shuffle=self.shuffle, seed = self.dataset_seed, broken_symm = self.broken_symm, spurions = spurions,input_spurions = self.input_spurions)
        
        # train_dataset = TensorDataset(train_data,train_labels)
        # self.train_dataset = train_dataset
        self.train_data = data.train_data
        self.train_labels = data.train_labels
        self.train_loader = data.train_loader
        
        return data.train_loader
        

    def set_model(self,gens_list=gens_Lorentz,ML_model = "NN",input_size = 4,init = "rand",equiv="False",rand="False",freeze = "False", activation = "ReLU",skip="False", hidden_size=10, n_hidden_layers=3,input_dim = 9, rho_size = 256, phi_size = 128, embed_dim=256):
        
        self.ML_model = ML_model
        
        
        if self.ML_model == "DeepSet":
            self.input_dim = input_dim
            self.rho_size = rho_size
            self.phi_size = phi_size
            
        elif self.ML_model == "Transformer":
            self.input_size = input_size
            self.embed_dim = embed_dim
            self.hidden_size = hidden_size
        else:
            self.init = init
            self.equiv = equiv
            self.rand=rand
            self.freeze = freeze 
            self.activation = activation
            self.skip=skip
            self.input_size = input_size+sum([torch.numel(sp) for sp in self.spurions]) if (self.input_spurions == "True" or self.input_spurions==True) else input_size
            self.hidden_size = hidden_size
            self.n_hidden_layers = n_hidden_layers
        
    def train_model(self,model, dataloader, criterion, penalty,optimizer, nepochs=15, device='cpu', apply_dSEAL=False,lambda_SEAL = 1.0, apply_GSEAL = False, clip_grads = False, beta_max = 0.95):

        model.to(device)
        
        num_epochs = nepochs
        # save the losses
        apply_dSEAL = (apply_dSEAL==True) or (apply_dSEAL=="True")
        apply_GSEAL = (apply_GSEAL==True) or (apply_GSEAL=="True")
        
        self.apply_GSEAL = apply_GSEAL
        self.apply_dSEAL = apply_dSEAL
        self.beta_max = beta_max
        
        loss_tracker = {
            "Loss": [],
            "BCE": [],
            "dSEAL": [],
            "beta": [],
            "GSEAL": []
        }
        
        print(lambda_SEAL)
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            rbce = 0.0
            rdSEAL = 0.0
            rGSEAL = 0.0

            for batch_idx, batch in enumerate(dataloader):#tqdm(dataloader, desc=f"Epoch: {epoch}")):
                optimizer.zero_grad()  # Zero the gradients

                batch = [x.to(device) for x in batch]
                
                X, y= batch
                
                # X = X.to(device)
                # y = y.to(device)
                # X_cy = X_cy.to(device)
                # mask = mask.to(device)                

                outputs = model(X)
                bce = criterion(outputs.squeeze(), y)
                
                X_boost = boost_3d(X, device, beta_max = beta_max)
                
                optimizer.zero_grad()  # Zero the gradients

                outputs_boost = model(X_boost)

                # catch NaNs
                output_nan = torch.sum(torch.isnan(outputs_boost))

                if output_nan > 0:
                    print(f"Nan found in output in batch: {batch_idx}, Nans: {output_nan}")
                    outputs_boost = torch.nan_to_num(outputs_boost, nan=0.5)

                GSEAL = penalty(outputs.squeeze(), outputs_boost.squeeze())
                
                
                
                
                criterion_Lorentz = deltaSEAL(gens_list=self.gens_list, model = model)
                dSEAL = criterion_Lorentz(input = X)

                # print(dSEAL)

                loss = 1.0*bce
                if apply_dSEAL:
                    loss += lambda_SEAL*dSEAL 
                if apply_GSEAL:
                    loss += lambda_SEAL*GSEAL 
                
                if torch.isinf(dSEAL):
                    print("Found infinity")

                loss.backward()  # Backward pass
                # gradients too large? 
                if clip_grads:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

                optimizer.step()  # Update parameters

                running_loss += loss.item()
                rbce += bce.item()
                rdSEAL += dSEAL.item()
                rGSEAL += GSEAL.item() 
                # rbeta += beta.item()
                # break


            running_loss /=len(dataloader)
            rbce /= len(dataloader)
            rdSEAL /= len(dataloader)
            rGSEAL /= len(dataloader)
            # rbeta /= len(dataloader)
            if (epoch % 100 == 0) or (epoch == nepochs-1):
                print(f'lambda = {lambda_SEAL} Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}, task: {rbce:.4f}, dSEAL: {rdSEAL}, GSEAL:{rGSEAL}')
            loss_tracker["Loss"].append(running_loss)
            loss_tracker["BCE"].append(rbce)
            loss_tracker["dSEAL"].append(rdSEAL)
            loss_tracker["GSEAL"].append(rGSEAL)
            # loss_tracker["beta"].append(rbeta)
            # break
            model_clone = copy.deepcopy(model)


        return loss_tracker,model_clone
    
    def run_training(self,lam_vec, dataloader, criterion = torch.nn.MSELoss(), penalty = torch.nn.MSELoss(),opt = "Adam",lr = 5e-4, nepochs=15, device='cpu', apply_dSEAL=False, apply_GSEAL=False,set_seed = True,seed = int(torch.round(torch.rand(1)*10000)),clip_grads = False, beta_max = 0.95):
        
        self.lr = lr
        self.opt = opt
        self.apply_dSEAL = apply_dSEAL
        self.apply_GSEAL = apply_GSEAL
        self.train_seed = seed
        self.clip_grads = clip_grads
        
        if set_seed:
            np.random.seed(seed)
            torch.manual_seed(seed)

        train_loader_copy = copy.deepcopy(dataloader)
        if self.ML_model == "DeepSet":
            model = DeepSet(input_dim=self.input_dim, rho_size= self.rho_size, phi_size=self.phi_size)
        elif self.ML_model == "Transformer":
            model = Transformer(input_dim = self.input_dim, embed_dim = self.embed_dim, hidden_size = self.hidden_size)
        else:
            modelLorentz_symm = genNet(input_size = self.input_size, init = self.init ,equiv=self.equiv,rand=self.rand,freeze = self.freeze,activation = self.activation, skip = self.skip, n_hidden_layers = self.n_hidden_layers, hidden_size=self.hidden_size)
            model = modelLorentz_symm.to(devicef)
            
        train_outputs = {}
        models = {}

        for lam_val in lam_vec:
            train_loader_copy = copy.deepcopy(dataloader)
            if set_seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            model_train = copy.deepcopy(model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total model parameters: {total_params}")
            
            # Define the loss function and optimizer
            if self.opt == "SGD":
                optimizer = optim.SGD(model_train.parameters(), lr=lr, momentum = 0.9)

            elif self.opt =="GD":
                optimizer = optim.GD(model_train.parameters(), lr=lr)
            
            else:
                optimizer = optim.Adam(model_train.parameters(), lr=lr)


            train_output,model_trained = self.train_model(model = model_train, dataloader = train_loader_copy, criterion = criterion, penalty = penalty ,optimizer = optimizer, device="cuda",apply_dSEAL=apply_dSEAL,apply_GSEAL=apply_GSEAL,lambda_dSEAL = lam_val,nepochs=nepochs,clip_grads = clip_grads, beta_max = beta_max)

            models[lam_val] = copy.deepcopy(model_trained)
            train_outputs[lam_val] = copy.deepcopy(train_output)
            
        self.models = models
        self.train_outputs = train_outputs

        return train_outputs,models
                

class analysis_trained(symm_net_train):

    def __init__(self):
        super().__init__()
        
    def get_trained(self, trained_net):
        self.__dict__ = {key:copy.deepcopy(value) for key, value in trained_net.__dict__.items()}
        
        
    
    def title(self):
        #text = f"N:{self.N} hidden size:{self.hidden_size} layers:{self.n_hidden_layers} activation:{self.activation} lr:{self.lr} opt:{self.opt}"
        text = f"toy {self.ML_model} dSEAL:{self.apply_dSEAL} GSEAL:{self.apply_GSEAL}"
        self.spurions_for_print = ""
        if self.broken_symm == "True" or self.broken_symm == True:
            text=f"{text} broken symm"
            spurions_for_print = "spurions:\n"
            text_spurions = f" "
            if self.input_spurions == "True" or self.input_spurions == True:
                text = f"{text} input spurions"
            for spurion in self.spurions_list:
                spurions_for_print+= f"{spurion}\n"
                spurion_text = f"{spurion}"
                #spurion_text = spurion_text
                text_spurions += f"spurion: {spurion_text}"
            self.spurions_for_print = spurions_for_print
            text = f"{text}{text_spurions}"
        if self.equiv== "True":
            text=f"{text} bi-linear layer"
            if self.skip =="True":
                text=f"{text} skip"
            if self.init=="eta" or self.init=="delta":
                text=f"{text} init: {self.init}"
            if self.freeze=="True":
                text=f"{text} freeze"
       
        self.title_text = text
        
        text = text+f" N:{self.N} lr:{self.lr} "
        if hasattr(self,"clip_grads"):
            text = text+ f" clip_grads_{self.clip_grads}"
        if hasattr(self,"beta_max") and self.apply_GSEAL:
            text = text+ f" beta_max_{self.beta_max}"
        
       
        self.filename = text.replace("[","").replace("]","").replace(",","_").replace(":","_").replace(" ","_")+f"_data_seed_{self.dataset_seed}_train_seed_{self.train_seed}"

        return text
    
    def save_trained(self,outdir = "./storage"):
        filename = self.title_text.replace(" ","_").replace(":","_")+f"data_seed_{self.dataset_seed}_train_seed_{self.train_seed}"
        with open(f'{outdir}/{filename}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self, f)

    def plot_losses(self,save = False, outdir = "./",filename = "",print_spurions = False):

        color_vec = ["violet","blue","green","yellow","orange","red","pink","purple","teal","magenta"]
        train_loss_lam = self.train_outputs
        
        models = self.models
        
        plt.figure()
        for i,lam_val in enumerate(models.keys()):
            losses = self.train_outputs[lam_val]
            plt.semilogy(range(len(losses["BCE"])),losses["BCE"],label = rf"$\lambda$ = {lam_val}, task", color = color_vec[i%len(color_vec)])
            plt.semilogy(range(len(losses["dSEAL"])),losses["dSEAL"],label = rf"$\lambda$ = {lam_val}, $\delta$SEAL", color = color_vec[i%len(color_vec)],ls = "--")
            if "GSEAL" in losses.keys():
                plt.semilogy(range(len(losses["GSEAL"])),losses["GSEAL"],label = f"GSymm", color = color_vec[i%len(color_vec)],ls = "-.")
                
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        text = self.title()
        plt.title(text)
        
        if print_spurions == "True" or print_spurions == True:
            plt.annotate(self.spurions_for_print,xy=(0.05,0.7),xycoords = "axes fraction")
        
        if save==True or save=="True":
            if filename =="":
                filename = "plot_losses_"+self.filename
            plt.savefig(f"{outdir}/{filename}.pdf")
        
    def plot_losses_side(self,save = False, outdir = "./",filename = "",log = False):
        nfigs = 4 if "GSEAL" in self.train_outputs[0.0].keys() else 3
        
        fig, ax = plt.subplots(1,nfigs, figsize=(12,4))
        for lam_val in self.train_outputs.keys():
            losses = self.train_outputs[lam_val]
            # Total Loss
            lam = f"{lam_val:.1e}"
            ax[0].semilogy(losses["Loss"], label=rf"$\lambda = {lam}$") if log else ax[0].plot(losses["Loss"], label=rf"$\lambda = {lam}$")
           
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Total Loss")
            ax[0].legend()

            # BCE Component
            ax[1].semilogy(losses["BCE"]) if log else ax[1].semilogy(losses["BCE"])
            
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("task Loss")

            # dSEAL Component
            ax[2].semilogy(losses["dSEAL"])
            
            ax[2].set_xlabel("Epoch")
            ax[2].set_ylabel(r"$\delta$SEAL Loss")
            
            # GSEAL Component
            if "GSEAL" in losses.keys():
                ax[3].semilogy(losses["GSEAL"]) if log else ax[3].plot(losses["GSEAL"])
                
                ax[3].set_xlabel("Epoch")
                ax[3].set_ylabel("GSymm Loss")

        fig.tight_layout()
        if save==True or save=="True":
            if filename =="":
                filename = "plot_losses_side_"+self.filename
        plt.savefig(f"{outdir}/{filename}.pdf")
            
    
    def plot_dSEAL(self,save = False, outdir = "./",filename = "",print_spurions = False):
        color_vec = ["violet","blue","green","yellow","orange","red","pink","purple","teal","magenta"]
        
        dSEAL_lam = self.dSEAL_lam
        models = self.models
        
        plt.figure()
        for i,lam_val in enumerate(models.keys()):
            
            plt.semilogy(range(len(dSEAL_lam[lam_val])),dSEAL_lam[lam_val],label = rf"$\lambda$ = {lam_val}, dSEAL", color = color_vec[i%len(color_vec)])
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("dSEAL loss")
        text = self.title()
        plt.title(text)
        
        if print_spurions == "True" or print_spurions == True:
            plt.annotate(self.spurions_for_print,xy=(0.05,0.7),xycoords = "axes fraction")
        
        if save==True or save=="True":
            if filename =="":
                filename = "plot_dSEAL_"+self.filename
            plt.savefig(f"{outdir}/{filename}.pdf")

    
    def plot_task_loss(self,save = False, outdir = "./",filename = "",print_spurions = False):
        color_vec = ["violet","blue","green","yellow","orange","red","pink","purple","teal","magenta"]
        train_loss_lam = self.train_outputs
        
        models = self.models
        
        plt.figure()
        for i,lam_val in enumerate(models.keys()):
            plt.semilogy(range(len(train_loss_lam[lam_val])),train_loss_lam[lam_val],label = rf"$\lambda$ = {lam_val}, GSEAL", color = color_vec[i%len(color_vec)])
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("GSEAL loss")
        text = self.title()
        plt.title(text)
        
        if print_spurions == "True" or print_spurions == True:
            plt.annotate(self.spurions_for_print,xy=(0.05,0.7),xycoords = "axes fraction")
        
        if save==True or save=="True":
            if filename =="":
                filename = "plot_GSEAL_losses_"+self.filename
            plt.savefig(f"{outdir}/{filename}.pdf")

            
            
            
    


        
            