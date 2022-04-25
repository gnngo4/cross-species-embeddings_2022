import sys, os
sys.path.append(os.getcwd())

from src.nifti_converter import NiftiConverter

from tqdm import tqdm
import numpy as np

# Inputs
in_dir = '/mnt/WD10TB/datasets/data_marmosetbrainconnectome/raw/preprocess_data/cortex_data/10k/nifti'
adj_dir = './data/data_marmoset/single_subject_gradients'
roi = os.path.join(adj_dir,'surfFS.atlasroi.10k.nii.gz')
mask = os.path.join(adj_dir,'surfFS.atlasroi.10k.nii.gz')
participants = [1,2,3,4,5,6,7,8,
                10,11,12,14,15,16,
                18,19,20,21,22,23,
                24,25,26,27,28,29,
                30,31]
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
for participant in participants: 
    outfile = os.path.join(adj_dir,f"marm-id-{str(participant)}.cortex-to-cortex.corr.npy") 
    niftis = os.listdir(in_dir)
    first = True
    for nifti in tqdm(niftis):
        if f"sub-m{participant}_" in nifti:
            print(nifti)
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
