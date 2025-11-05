"""
Code Adapated from: https://github.com/ViniciusMikuni/OmniLearn/blob/main/preprocessing/preprocess_atlas.py
Date Accessed: 22nd July 2024 
"""
import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser
import energyflow as ef
import sys

#Preprocessing for the top tagging dataset
def clustering(data,folder,sample='train.h5',nevents=1000,nparts=100, output_path="", debug=True, four_vector_only=True,jet_e_sum_const = True,paper_preproc = False):
    if not output_path:
        output_path = folder
    
    print(f"Using sample {sample}...")
    print(f"Saving files in {output_path}...")
    print(f"Only saving cartesian four-vectors: {four_vector_only}")

    # sys.exit()

    if debug:
        print(f"1. Creating jet constituent four vectors..")
    
    npid = data['labels'][:nevents]
    weights = data['weights'][:nevents]
    particles = np.stack([
        data['fjet_clus_pt'][:nevents,:nparts]/1000.,
        data['fjet_clus_eta'][:nevents,:nparts],
        data['fjet_clus_phi'][:nevents,:nparts],
        data['fjet_clus_E'][:nevents,:nparts]/1000.,
    ],-1)

    # to cartesian coordinates (assuming massless)
    if debug:
        print(f"2. Adding cartesian coordinates...")
    if paper_preproc:
        particles_cartesian = ef.p4s_from_ptyphims(particles[:,:,:3])
    else:
        Es = particles[:,:,3]
        pts=particles[:,:,0]
        etas=particles[:,:,1]
        phis = particles[:,:,2]
       
        pxs = pts*np.cos(phis)
        pys = pts*np.sin(phis)
        pzs = pts*np.sinh(etas)
        particles_cartesian = np.stack([Es, pxs, pys,pzs],axis=-1)
       
    mask = particles[:,:,0]>0

    if four_vector_only:
        del particles

    if debug:
        print(f"3. Creating jet four vectors...")
    
    jets = np.stack([
        data['fjet_pt'][:nevents]/1000.,
        data['fjet_eta'][:nevents],
        data['fjet_phi'][:nevents],
        data['fjet_m'][:nevents]/1000.,
    ],-1)

    # to cartesian coordinates with mass
    if debug:
        print(f"4. Adding cartesian coordinates...")
    if paper_preproc:
        jets_cartesian = ef.p4s_from_ptyphims(jets)
    else:
        jet_ms = jets[:,3]
        jet_pts=jets[:,0]
        jet_etas=jets[:,1]
        jet_phis = jets[:,2]
       
        jet_pxs = jet_pts*np.cos(jet_phis)
        jet_pys = jet_pts*np.sin(jet_phis)
        jet_pzs = jet_pts*np.sinh(jet_etas)
        jet_Es = np.sqrt(jet_ms**2+jet_pts**2)*np.cosh(jet_etas)
        jets_cartesian = np.stack([jet_Es, jet_pxs, jet_pys,jet_pzs],axis=-1)
       
    
    if four_vector_only:
        jets = jets_cartesian
    
    else:
        if jet_e_sum_const:
            jet_e = np.sum(data['fjet_clus_E'][:nevents]/1000,1)
        else:
            jet_e = np.sqrt(jets[:,0]**2+jets[:,3]**2)*np.cosh(jets[:,1])

        jets = np.concatenate([jets,np.sum(mask,-1)[:,None]],-1)
        jets = np.concatenate([jets, jets_cartesian], axis=-1)
    
    if debug:
        print(f"5. Calculating extra constituent variables...")
    
    if four_vector_only:
        NFEAT=4
        points = particles_cartesian
   
    else:

        NFEAT=11
        points = np.zeros((particles.shape[0],particles.shape[1],NFEAT))

        delta_phi = particles[:,:,2] - jets[:,None,2]
        delta_phi[delta_phi>np.pi] -=  2*np.pi
        delta_phi[delta_phi<= - np.pi] +=  2*np.pi


        points[:,:,0] = (particles[:,:,1] - jets[:,None,1]) # delta eta
        points[:,:,1] = delta_phi # delta phi
        points[:,:,2] = np.ma.log(particles[:,:,0]/jets[:,None,0]).filled(0) # log(pt/jet_pt)
        points[:,:,3] = np.ma.log(particles[:,:,0]).filled(0) # log(pt)
        points[:,:,4] = np.ma.log(particles[:,:,3]/jet_e[:,None]).filled(0)    # log(E/jet_E)
        points[:,:,5] = np.ma.log(particles[:,:,3]).filled(0)# log(E)
        points[:,:,6] = np.hypot(points[:,:,0],points[:,:,1])# delta R
        points[:,:,7:] = particles_cartesian
    

    points*=mask[:,:,None]

    if not four_vector_only:
        jets = np.delete(jets,2,axis=1)

    if debug:
        print(f"5. Done calculating extra constituent variables...")
        print(f"Shape of constituent variables: {points.shape}")
        print(f"Shape of jet variables: {jets.shape}")

    if 'train' in sample:
        with h5py.File('{}/train_atlas.h5'.format(output_path), "w") as fh5:
            dset = fh5.create_dataset('Pmu', data=points[:int(0.8*npid.shape[0])])
            dset = fh5.create_dataset('jet', data=jets[:int(0.8*npid.shape[0])])
            dset = fh5.create_dataset('pid', data=npid[:int(0.8*npid.shape[0])])
            dset = fh5.create_dataset('weights', data=weights[:int(0.8*npid.shape[0])])

        with h5py.File('{}/valid_atlas.h5'.format(output_path), "w") as fh5:
            dset = fh5.create_dataset('Pmu', data=points[int(0.8*npid.shape[0]):])
            dset = fh5.create_dataset('jet', data=jets[int(0.8*npid.shape[0]):])
            dset = fh5.create_dataset('pid', data=npid[int(0.8*npid.shape[0]):])
            dset = fh5.create_dataset('weights', data=weights[int(0.8*npid.shape[0]):])
        
    else:            
        with h5py.File('{}/{}_atlas.h5'.format(output_path,sample), "w") as fh5:
            dset = fh5.create_dataset('Pmu', data=points)
            dset = fh5.create_dataset('jet', data=jets)
            dset = fh5.create_dataset('pid', data=npid)
            dset = fh5.create_dataset('weights', data=weights)


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=120, help="Number of particles per event")
    parser.add_option("--njets", type=int, default=20_000_000, help="Total number of jets")
    parser.add_option("--folder", type="string", default='./', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='test.h5', help="Input file name")
    parser.add_option("--output_folder", type="string", default="", help="Folder to save outputs")
    parser.add_option("--paper_preproc", action="store_true", default=False, help="Use preprocessing used in paper: note that should be trained with --fix_wrong_preproc")
    parser.add_option("--jet_e_sum_const", action="store_true", default=False, help="Jet energy calculated from sum of constituents energies")

    (flags, args) = parser.parse_args()
        

    output_path = flags.output_folder
    samples_path = flags.folder
    sample = flags.sample
    NPARTS = flags.npoints
    njets = flags.njets
    paper_preproc=flags.paper_preproc
    jet_e_sum_const = flags.jet_e_sum_const

    data = h5py.File(os.path.join(samples_path,sample),'r')

    if njets > data['labels'].shape[0]:
        njets = data['labels'].shape[0]
    
    clustering(data,samples_path,flags.sample.replace('.h5',''),njets,NPARTS, output_path,paper_preproc=paper_preproc,jet_e_sum_const = jet_e_sum_const)
    
