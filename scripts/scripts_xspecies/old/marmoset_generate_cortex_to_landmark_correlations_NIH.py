import sys, os
sys.path.append(os.getcwd())

from src.nifti_converter import NiftiConverter

from tqdm import tqdm
import numpy as np

# Inputs
sing_shell="singularity exec -B /mnt/WD10TB /home/geoff/Desktop/containers/fmriprep_ciftify-1.3.2-2.3.3"
in_dir = '/mnt/WD10TB/datasets/data_marmosetbrainconnectome/raw/preprocess_data/cortex_data/10k/nifti'
adj_dir = './data/data_xspecies'
outfile = os.path.join(adj_dir,"marm-NIH.hippocampus-to-cortex.corr.npy") 
outfile_nifti = os.path.join(adj_dir,"marm-NIH.hippocampus-to-cortex.corr.nii.gz") 
roi = './data/data_marmoset/surfFS.atlasroi.10k.nii.gz'
mask = './data/data_marmoset/surfFS.atlasroi.10k.nii.gz'
template = './data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.dtseries.nii' 
# Inputs for homolog timeseries extraction
hipp_dir = '/mnt/WD10TB/datasets/data_marmosetbrainconnectome/raw/preprocess_data/volume_data'
dlabel =  'atlas/marmoset/surfFS.MBM_cortex_vPaxinos.10k.dlabel.nii' 
homologous_dir = 'atlas/homologous_rois'
homologs = {'HIPPOCAMPUS': 'data/data_marmoset/atlas_MBM_space-rfmri_hippocampus.nii.gz',
            'M1': 'A4ab+A4c',
            '3a': 'A3a',
            '3b': 'A3b',
            '1+2': 'A1/A2',
            'A1': 'AuA1',
            'V1': 'V1',
            'V2': 'V2',
            'MT': 'V5',
            '8Av': 'A8Av',
            'FEF': 'A45',
            'dlPFC': 'A8aD',
            'PCC': 'PGM'
            }

# settings
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

from src.nifti_converter import NiftiConverter
settings = Namespace(roi=roi,
                     mask=mask,
                     corr_sparsity_threshold=0.1,
                     corr_sparsity_rule='none',
                     sim_sparsity_threshold=0.1,
                     sim_sparsity_rule='node',
                     n_dims=1,
                     norm_flag=False,
                     embedding_method='LE',
                     subset_idx_10k = 0
                     )

nii_marmoset = "data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.nii.gz"
nifti_machine = NiftiConverter(nii_marmoset,settings)

# functions
def get_roi_data(dlabel,label,roidir=homologous_dir):
    # Re-name label in the case of backslashes in the label name '/'
    if '/' in label:
        rename_label = label.replace('/','.')
    else:
        rename_label = label
    
    out_label = os.path.join(roidir,f"marmoset_{rename_label}.dscalar.nii")
    if os.path.exists(out_label.replace('dscalar.nii','nii.gz')):
        return out_label.replace('dscalar.nii','nii.gz')

    cmds = []
    for ix,label_ in enumerate(label.split('+')):
        if '/' in label_:
            rename_label_ = label_.replace('/','.')
        else:
            rename_label_ = label_
        out_label_ = os.path.join(roidir,f"marmoset_{rename_label_}.dscalar.nii")
        cmd = f"{sing_shell} wb_command -cifti-label-to-roi {dlabel} {out_label_} -name {label_}"
        cmds.append(cmd)
        cmd = f"{sing_shell} wb_command -cifti-create-dense-from-template {template} {out_label_} -cifti {out_label_}"
        cmds.append(cmd)
        cmd = f"{sing_shell} wb_command -cifti-convert -to-nifti {out_label_} {out_label_.replace('dscalar.nii','nii.gz')}"
        cmds.append(cmd)
        cmd = f"{sing_shell} fslmaths {out_label_.replace('dscalar.nii','nii.gz')} -bin {out_label_.replace('dscalar.nii','nii.gz')}"
        cmds.append(cmd)
        if ix == 0:
            cmd = f"cp {out_label_.replace('dscalar.nii','nii.gz')} {out_label.replace('dscalar.nii','nii.gz')}"
            cmds.append(cmd)
        else:
            cmd = f"{sing_shell} fslmaths {out_label_.replace('dscalar.nii','nii.gz')} -add {out_label.replace('dscalar.nii','nii.gz')} {out_label.replace('dscalar.nii','nii.gz')}"
            cmds.append(cmd)
    for cmd in cmds:
        print(cmd)
        os.system(cmd)
    
    return out_label.replace('dscalar.nii','nii.gz')

def hippocampus_data(nifti,indir):
    # FIX
    marm_id = nifti.split('_')[0].split('-')[1]
    direction = nifti.split('_')[1].split('-')[1]
    run = nifti.split('_')[2].split('-')[1].split('.')[0]
    vol_data = f"{marm_id}_{direction}_{run}.space-template.rfMRI.wmcsf_reg.nii.gz"
    return os.path.join(indir,vol_data)

# Run script
niftis = os.listdir(in_dir)
first = True
dataset_counter = 1
for nifti in tqdm(niftis):
    print(nifti)
    if 'sub-m1_' in nifti or 'sub-m2_' in nifti or 'sub-m3_' in nifti or 'sub-m4_' in nifti or 'sub-m5_' in nifti:
        print(f"SKIPPING: {nifti}")
        pass
    else:
        # Get path of data
        nifti_ = os.path.join(in_dir,nifti)
        hipp_nifti = hippocampus_data(nifti,hipp_dir)
        # Get data of cortex
        settings.roi = roi
        obj = NiftiConverter(nifti_,settings)
        data = obj.mask_rfmri()
        Z_data = obj.preprocess(data)
        # Loop through homologous ROIs and extract timeseries
        homologous_data = np.zeros((1,Z_data['roi'].shape[-1]))
        counter=0
        for homolog_roi, label in homologs.items():
            if 'HIPPOCAMPUS' in homolog_roi:
                settings.roi = label
                obj = NiftiConverter(hipp_nifti,settings)
            else:
                _ = get_roi_data(dlabel,label)
                continue
            data = obj.mask_rfmri()
            data = obj.preprocess(data)
            homologous_data[counter,:] = data['roi'].mean(0)
            counter += 1
        # Compute correlation matrix between cortex-and-homologous ROIs
        X = Z_data['roi']
        Y = homologous_data
        R = np.corrcoef(X,Y)[:X.shape[0],-Y.shape[0]:]
        # Sum all subjects and runs correlation matrices
        if first:
            C = R
            first = False
        else:
            C += R
        # Count number of datasets (subjects and runs)
        dataset_counter += 1
# Compute average
C /= dataset_counter
# Save
np.save(outfile,C)
nifti_machine.save_embedding(C,outfile_nifti)

# cmd params
singularity_container = "singularity exec -B /mnt/WD10TB /home/geoff/Desktop/containers/fmriprep_ciftify-1.3.2-2.3.3"
dtseries_marmoset = "data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.dtseries.nii"
# nifti to dscalar cmds
cmd = f"{singularity_container} wb_command -cifti-convert -from-nifti {outfile_nifti} {dtseries_marmoset} {outfile_nifti.replace('nii.gz','dscalar.nii')} -reset-scalars"
print(cmd); os.system(cmd)
