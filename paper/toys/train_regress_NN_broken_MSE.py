import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
import einops

import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import sys, os, time, datetime, json, argparse
from symm_MSE_loss_defs import *



#parse cmd arguments

parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jobid", help="job id", dest="jobid")
parser.add_argument("-c", "--jsonfile", help="json file", dest="jsonfile")
parser.add_argument("-d", "--defaultjsonfile", help="default json file", default ="n", dest="defaultjsonfile")
parser.add_argument("-o", "--outdir", help="output directory", default = "./storage",dest="outdir")
parser.add_argument("--plotsoutdir", help="plots directory", default = "./plots",dest="plotsoutdir")
parser.add_argument("-s", "--savenet", help="save net in pkl?", default = "y",dest="savenet")
parser.add_argument("-p", "--saveplots", help="save plots?", default = "n",dest="saveplots")
parser.add_argument("--seeddata",help = "set seed for data generation", default = "", dest = "seeddata")
parser.add_argument("--seedtrain",help = "set seed for trainning", default = "", dest = "seedtrain")

args = parser.parse_args()
jsonfile = args.jsonfile
defaultjsonfile = args.defaultjsonfile
savenet = args.savenet
saveplots = args.saveplots
outdir = args.outdir
plotsoutdir = args.plotsoutdir
seed_data = args.seeddata
seed_train = args.seedtrain

#get argumnets from json file
with open(jsonfile, 'r') as js:
    config = json.load(js)

if defaultjsonfile!="n":
    with open(defaultjsonfile, 'r') as djs:
        config_default = json.load(djs)
    for d_key in config_default.keys():
        if not (d_key in config.keys()):
            config[d_key] = config_default[d_key]
        



if args.seeddata.isnumeric():
    seed_data = int(seed_data)
else:
    seed_data = int(torch.round(torch.rand(1)*10000))
if args.seedtrain.isnumeric():
    seed_train = int(seed_train)
else:
    seed_train = int(torch.round(torch.rand(1)*10000))
print(f"data seed: {seed_data}, train seed: {seed_train}")


nepochs = config["nepochs"]
lamb_vec = config["lam_vec"]
print(lamb_vec)
lr = config["lr"]

ML_model = config["ML_model"]#"DeepSets"
input_dim = config["input_dim"]#9
rho_size = config["rho_size"]
phi_size = config["phi_size"]
hidden_size =config["hidden_size"]
n_hidden_layers = config["n_hidden_layers"]
activation = config["activation"]
apply_MSE = config["apply_MSE"]
apply_symm = config["apply_symm"]
clip_grads = config["clip_grads"]
clip_grads  = (clip_grads== "True" or clip_grads== True)

beta_max = (config["beta_max"] if "beta_max" in config.keys() else 0.95)
print(f"beta max  = {beta_max}")

apply_both_seq = ((apply_MSE == "True" or apply_MSE ==True) and (apply_symm == "True" or apply_symm ==True) )

apply_symm_vec = ([True, False]) if apply_both_seq else [apply_symm]
print(f"apply_symm_vec = {apply_symm_vec}")
apply_MSE_vec = [not symm for symm in apply_symm_vec] if apply_both_seq else [apply_MSE]
print(f"apply_MSE_vec = {apply_MSE_vec}")

for run, apply_symm in enumerate(apply_symm_vec):
    apply_MSE = apply_MSE_vec[run]
    
    symm_net = symm_net_train()
    
    broken_symm = config["broken_symm"]
    spurions = config["spurions"]
    if broken_symm == "True" or broken_symm == True:
        print("broken symmetry")
        print(f"spurions:{spurions}")
        train_loader = symm_net.prepare_dataset(seed = seed_data, N = config["N"], batch_size = config["batch_size"], broken_symm = config["broken_symm"], spurions = config["spurions"], true_func = Lorentz_myfun_broken, input_spurions = config["input_spurions"])
    else:
        train_loader = symm_net.prepare_dataset(seed = seed_data, N = config["N"], batch_size = config["batch_size"], broken_symm = config["broken_symm"], spurions = config["spurions"], input_spurions = config["input_spurions"])


    print(f"apply MSE:{apply_MSE}, apply symm = {apply_symm}")


    symm_net.set_model(ML_model = ML_model, init = config["init"],equiv=config["equiv"],rand=config["rand"], freeze =config["freeze"],skip = config["skip"],hidden_size =config["hidden_size"], n_hidden_layers = config["n_hidden_layers"],activation = config["activation"],input_dim = input_dim,rho_size =rho_size, phi_size = phi_size)


    symm_net.run_training(dataloader = train_loader,nepochs = config["nepochs"],lam_vec = config["lam_vec"],seed = seed_train, lr = config["lr"],apply_symm = apply_symm, apply_MSE = apply_MSE, clip_grads = clip_grads, beta_max = beta_max)


    anet =  analysis_trained()
    anet.get_trained(symm_net)
    anet.title()

    if not (savenet=="n" or (savenet==False) or (savenet=="False")):
        anet.save_trained(outdir = outdir)

    if not (saveplots=="n" or (saveplots==False) or (saveplots=="False")):
        anet.plot_losses(save = "True",outdir = plotsoutdir)
        anet.plot_losses_side(save = "True",outdir = plotsoutdir)

