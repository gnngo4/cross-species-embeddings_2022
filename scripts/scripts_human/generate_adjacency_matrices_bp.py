import sys, os
sys.path.append(os.getcwd())

from src.nifti_converter import NiftiConverter

from tqdm import tqdm
import numpy as np

# Inputs
in_dir = '/mnt/WD10TB/datasets/data_HCP/preprocess_data/cortex_data/10k/bp_filter/nifti'
adj_dir = './data/data_human'
outfile = os.path.join(adj_dir,"100UR.cortex-to-cortex.corr.bp.npy") 
roi = os.path.join(adj_dir,'cortex.nii.gz')
mask = os.path.join(adj_dir,'cortex.nii.gz')
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
