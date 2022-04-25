import sys, os
sys.path.append(os.getcwd())

from src.nifti_converter import NiftiConverter

from tqdm import tqdm
import numpy as np

# Inputs
sing_shell="singularity exec -B /mnt/WD10TB /home/geoff/Desktop/containers/fmriprep_ciftify-1.3.2-2.3.3"
in_dir = '/mnt/WD10TB/datasets/data_HCP/preprocess_data/cortex_data/10k/nifti'
adj_dir = './data/data_xspecies'
outfile = os.path.join(adj_dir,"100UR.hippocampus-to-cortex.corr.npy") 
outfile_nifti = os.path.join(adj_dir,"100UR.hippocampus-to-cortex.corr.nii.gz") 
roi = './data/data_human/cortex.nii.gz'
mask = './data/data_human/cortex.nii.gz'
template = './data/data_human/100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.dtseries.nii'
# Inputs for homolog timeseries extraction
hipp_dir = '/mnt/WD10TB/datasets/data_HCP/preprocess_data/hippocampus_data'
dlabel =  'atlas/10k/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.10k_fs_LR.dlabel.nii' 
homologous_dir = 'atlas/homologous_rois'
homologs = {'HIPPOCAMPUS': '/mnt/WD10TB/datasets/data_HCP/preprocess_data/ROIs/subcortical/hippocampus.nii.gz',
            'M1': 'L_4_ROI+R_4_ROI',
            '3a': 'L_3a_ROI+R_3a_ROI',
            '3b': 'L_3b_ROI+R_3b_ROI',
            '1+2': 'L_1_ROI+L_2_ROI+R_1_ROI+R_2_ROI',
            'A1': 'L_A1_ROI+R_A1_ROI',
            'V1': 'L_V1_ROI+R_V1_ROI',
            'V2': 'L_V2_ROI+R_V2_ROI',
            'MT': 'L_MT_ROI+R_MT_ROI',
            '8Av': 'L_8Av_ROI+R_8Av_ROI',
            'FEF': 'L_FEF_ROI+R_FEF_ROI',
            'dlPFC': 'L_8Ad_ROI+R_8Ad_ROI',
            'PCC': 'L_7m_ROI+R_7m_ROI'
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

nii_human = "data/data_human/100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.nii.gz"
nifti_machine = NiftiConverter(nii_human,settings)

# functions
def get_roi_data(dlabel,label,roidir=homologous_dir):
    out_label = os.path.join(roidir,f"human_{label}.dscalar.nii")
    if os.path.exists(out_label.replace('dscalar.nii','nii.gz')):
        return out_label.replace('dscalar.nii','nii.gz')
    
    cmds = []
    for ix,label_ in enumerate(label.split('+')):
        out_label_ = os.path.join(roidir,f"human_{label_}.dscalar.nii")
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
    
    return os.path.join(indir,nifti.replace('s6.cortex','hippocampus'))

# Run script
niftis = os.listdir(in_dir)
first = True
dataset_counter = 1
for nifti in tqdm(niftis):
    print(nifti)
    # Get path of data
    nifti_ = os.path.join(in_dir,nifti)
    hipp_nifti = hippocampus_data(nifti,hipp_dir)
    # Get data of cortex
    settings.roi = roi
    obj = NiftiConverter(nifti_,settings)
    data = obj.mask_rfmri()
    Z_data = obj.preprocess(data)
    # Loop through homologous ROIs and extract timeseries
    homologous_data = np.zeros((1,1200))
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
dtseries_human = "data/data_human/100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.dtseries.nii"
# nifti to dscalar cmds
cmd = f"{singularity_container} wb_command -cifti-convert -from-nifti {outfile_nifti} {dtseries_human} {outfile_nifti.replace('nii.gz','dscalar.nii')} -reset-scalars"
print(cmd); os.system(cmd)
