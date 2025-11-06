# SEAL
A Symmetry Encouraging Loss for High Energy Physics

<img width="150" height="150" alt="SEAL" src="https://github.com/user-attachments/assets/7162fd94-2302-475d-81d5-e669e83e5cc2" />




Please cite as:
```
@article{Hebbar:2025adf,
    author = "Hebbar, Pradyun and Madula, Thandikire and Mikuni, Vinicius and Nachman, Benjamin and Outmezguine, Nadav and Savoray, Inbar",
    title = "{SEAL - A Symmetry EncourAging Loss for High Energy Physics}",
    eprint = "2511.01982",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "11",
    year = "2025"
}
```
## Training
To train the transformer model:
```bash
python -u ${scriptsdir}/train_transformer_top_multinode_SEAL.py --data_dir $DATADIR --output_dir $outdir --save_tag $save_tag --boost_type 3D --num_epochs $num_epochs --batch_size 256 --fix_wrong_preproc --remove_wrong_jet_e --jet_e_sum_const 
```
For deltaSEAL add flags: ``` --apply_dSEAL --lam_dSEAL $lambda```

For GSEAL add flags: ``` --apply_GSEAL --lam_GSEAL $lambda```

```--jet_e_sum_const``` should match the preprocessing flag (see below).

```--remove_wrong_jet_e```, if used with ```--jet_e_sum_const```, will remove jets from the calculation of SEAL which cannot be boosted since log(E_i/E_jet) cannot be restored from the boosted jet. This is not an issue if ```--jet_e_sum_const``` was not used in the preprocessing.

## ATLAS Top Tagging Dataset

The ATLAS Top Tagging Dataset can be downloaded from the following [link](https://opendata.cern.ch/record/15013#:~:text=The%20ATLAS%20Top%20Tagging%20Open,and%202.5%20million%20jets%20respectively) and preprocessed using the following script:

```bash

python preprocess_atlas.py --folder FOLDER --sample [train.h5/test.h5] --jet_e_sum_const --paper_preproc
```

```--paper_preproc``` will reproduce the preprocessing done in the paper exactly, and requires flag ```--fix_wrong_preproc``` when running the training (don't use ```--fix_wrong_preproc``` in training otherwise).

```--jet_e_sum_const``` will calculate the jet's energy as a sum of the constituents energies before truncating the jet to construct the features (but not in jet_vec). Otherwise the jet's energy is taken from jet_vec.




