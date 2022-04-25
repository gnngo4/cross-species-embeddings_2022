import sys, os
sys.path.append(os.getcwd())

from src.nifti_converter import NiftiConverter

from tqdm import tqdm
import numpy as np

# Inputs
in_dir = '/mnt/WD10TB/datasets/data_marmosetbrainconnectome/raw/preprocess_data/cortex_data/10k/nifti'
adj_dir = './data/data_marmoset'
outfile = os.path.join(adj_dir,"marm-NIH.cortex-to-cortex.corr.npy") 
roi = os.path.join(adj_dir,'surfFS.atlasroi.10k.nii.gz')
mask = os.path.join(adj_dir,'surfFS.atlasroi.10k.nii.gz')
# settings
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

settings = Namespace(roi=roi,
                     mask=mask,
                     corr_sparsity_threshold=0.1,
                     corr_sparsity_rule='none',
                     sim_sparsity_threshold=0.1,
                     sim_sparsity_rule='node',
                     n_dims=100,
                     norm_flag=False,
                     embedding_method='LE'
                     )


# Run script
niftis = os.listdir(in_dir)
first = True
for nifti in tqdm(niftis):
    print(nifti)
    if 'sub-m1_' in nifti or 'sub-m2_' in nifti or 'sub-m3_' in nifti or 'sub-m4_' in nifti or 'sub-m5_' in nifti:
        print(f"SKIPPING: {nifti}")
        pass
    else:
        nifti_ = os.path.join(in_dir,nifti)
        obj = NiftiConverter(nifti_,settings)
        data = obj.mask_rfmri()
        Z_data = obj.preprocess(data)
        R = obj.correlation_matrix(Z_data)
        if first:
            R_avg = R
            first = False
        else:
            R_avg += R

R_avg /= R_avg[0,0]
np.save(outfile,R_avg)
