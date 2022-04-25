import sys, os
sys.path.append(os.getcwd())

import numpy as np
import nibabel as nib
from src.nifti_converter import NiftiConverter

"""
Load cross-species-group-gradient data & add sparsity to corr. matrices
 - specify row-wise sparsity & apply only to C_[human|marmoset]
"""

class Namespace:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

def save_matrices(X,im_out,_dir,_figsize=(10,10),_dpi=200,_aspect='auto',_vmin=0,_vmax=.8):
    import os
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=_figsize,dpi=_dpi)
    im = plt.imshow(X,cmap='YlGnBu',interpolation='nearest',aspect=_aspect,vmin=_vmin,vmax=_vmax)
    fig.colorbar(im)
    plt.savefig(os.path.join(_dir,im_out))
    plt.close('all')

def add_row_sparsity(C,threshold,binarize_corr=False):
    C_sp = np.zeros((C.shape))
    K = int(C_sp.shape[0]*threshold)
    for row_idx in range(C_sp.shape[0]):
        C_ = C[row_idx,:]
        min_ = C_[np.argpartition(C_,-K)[-K:]].min()
        C_sp[row_idx,:] += (C_>min_)
    adj = np.divide(C_sp, C_sp, out=np.zeros_like(C_sp), where=C_sp!=0)
    C_sp = np.multiply(C,C_sp)
    # Remove negative values
    if (C_sp< 0).sum() > 0:
        print('Warning: Removing correlation values less than 0.')
        C_sp = np.multiply(C_sp>0,C_sp)
    else:
        pass
    
    if binarize_corr:
        return (C_sp>0).astype(int)
    else:
        return C_sp

def Z_row_normalise(X):
    
    return (X-X.min(1,keepdims=1)) / (X.max(1,keepdims=1)-X.min(1,keepdims=1))

def read_nifti(nifti):

    x = nib.load(nifti)
    affine = x.affine

    return x.get_fdata(),affine

"""
Settings
Hyperparameters:
    `create_dir`,
    `human_dataset`, `marm_dataset`, and `row_sparsity`
"""

DIRS_ = {'JOINT0_SPARSE_v3_plot':['Hippocampus','M1','3a','3b','1+2','A1','V1','V2','MT','FEF','dlPFC','PCC','PPC'],}
for create_dir, rois in DIRS_.items():

    # Make directory
    print(f"DIRECTORY [basename]: {create_dir}\nROIs: {rois}")
    if os.path.isdir(f"./data/data_xspecies/{create_dir}"):
        pass
    else:
        os.mkdir(f"./data/data_xspecies/{create_dir}")

    # Choose human cohort
    human_dataset = "100UR"

    # Choose marmoset cohort
    for marm_dataset in ["NIH","UWO"]:

        # Add row-wise sparsity to RSFC matrix
        for row_sparsity in [.01]:

            # Preprocessed connectivity profiles
            human_cortex_to_cortex = 'data/data_human/100UR.cortex-to-cortex.corr.bp.npy'
            marm_cortex_to_cortex = f"data/data_marmoset/marm-{marm_dataset}.cortex-to-cortex.corr.npy"
            human_cortex_to_hipp = 'data/data_xspecies/100UR.hippocampus-to-cortex.corr.bp.npy'
            marm_cortex_to_hipp = f"data/data_xspecies/marm-{marm_dataset}.hippocampus-to-cortex.corr.npy"
            
            # Outputs
            outdir = f"./data/data_xspecies/{create_dir}/joint_eigenmap.human-{human_dataset}.marmoset-{marm_dataset}.thr-corr-{row_sparsity}.norm"
            if os.path.isdir(outdir):
                continue
            else:
                os.mkdir(outdir)
            out_human = os.path.join(outdir,f"joint_eigenmap_human.nii.gz")
            out_marmoset = os.path.join(outdir,f"joint_eigenmap_marmoset.nii.gz")
            out_human_landmark_rsfc = os.path.join(outdir,f"landmark_rsfc_human.nii.gz")
            out_marmoset_landmark_rsfc = os.path.join(outdir,f"landmark_rsfc_marmoset.nii.gz")
            out_human_landmark_mask = os.path.join(outdir,f"landmark_mask_human.nii.gz")
            out_marmoset_landmark_mask = os.path.join(outdir,f"landmark_mask_marmoset.nii.gz")

            print(
            f"""
            Performing joint-embedding algorithm on human and marmoset
            resting-state functional connectivity data.

            Output directory: {outdir}
            Marmoset dataset: {marm_dataset}
            Human dataset: {human_dataset}
            - Keeping only top {row_sparsity*100}% of row-wise cortex-to-cortex RSFC connections:
                1) {human_cortex_to_cortex}
                2) {marm_cortex_to_cortex}
            - Removing all negative cortex-to-landmark RSFC connections:
            """
            )

            """
            Load connectivity data
             - cortex-to-cortex connectivity data
             - cortex-to-landmark connectivity data

             ALSO, generate landmark ROIs, and cortex-to-landmark connectivity profiles
             to be saved at the end of this script.
            """
            def generate_cortex_to_landmark(roi_list,species,
                                                hdir="atlas/homologous_rois/visualize",
                                                human_cortex_to_cortex=human_cortex_to_cortex,
                                                marmoset_cortex_to_cortex=marm_cortex_to_cortex,
                                                human_cortex_to_hipp=human_cortex_to_hipp,
                                                marmoset_cortex_to_hipp=marm_cortex_to_hipp,
                                                human_cortex_mask='data/data_human/cortex.nii.gz',
                                                marmoset_cortex_mask='data/data_marmoset/surfFS.atlasroi.10k.nii.gz'):
                """
                HOMOLOGOUS ROIS MUST BE GENERATED BEFORE RUNNING THIS
                """
                if species == 'human':
                    if isinstance(human_cortex_to_cortex,np.ndarray):
                        cortex_to_cortex = human_cortex_to_cortex
                    else:
                        cortex_to_cortex = np.load(human_cortex_to_cortex)
                    cortex_to_hipp = np.load(human_cortex_to_hipp)
                    cortex_mask,aff = read_nifti(human_cortex_mask)
                elif species == 'marmoset':
                    if isinstance(marmoset_cortex_to_cortex,np.ndarray):
                        cortex_to_cortex = marmoset_cortex_to_cortex
                    else:
                        cortex_to_cortex = np.load(marmoset_cortex_to_cortex)
                    cortex_to_hipp = np.load(marmoset_cortex_to_hipp)
                    cortex_mask,aff = read_nifti(marmoset_cortex_mask)
                else:
                    NotImplemented

                cortex_to_landmark = np.zeros((int(cortex_mask.sum()),len(roi_list)))
                roi_masks = np.zeros((int(cortex_mask.sum()),len(roi_list)))
                for ix,roi in enumerate(roi_list):
                    if roi != 'Hippocampus':
                        ROI_PATH = os.path.join(hdir,f"{species}_{roi}.nii.gz")
                        if not os.path.exists(ROI_PATH):
                            print(f"{ROI_PATH} does not exist.")
                            sys.exit()
                        print(f"Homologous ROI path: {ROI_PATH}")
                        # Get roi_img
                        roi_img,aff = read_nifti(ROI_PATH)
                        roi_img = roi_img[np.where(cortex_mask==1)]
                        # Get conn_img
                        roi_to_cortex_conn = cortex_to_cortex[np.where(roi_img==1)].mean(0)
                        # Load into `cortex_to_landmark`
                        cortex_to_landmark[:,ix] = roi_to_cortex_conn
                        roi_masks[:,ix] = roi_img
                    else:
                        #cortex_to_hipp = add_row_sparsity(np.vstack([cortex_to_hipp[:,0]]*cortex_to_cortex.shape[0]),row_sparsity)[0,:]
                        cortex_to_landmark[:,ix] = cortex_to_hipp[:,0]
                
                return cortex_to_landmark,roi_masks
            
            # Get data
            C_human = np.load(human_cortex_to_cortex); C_human = add_row_sparsity(C_human,row_sparsity)           # Load human cortex-to-cortex
            C_marmoset = np.load(marm_cortex_to_cortex); C_marmoset = add_row_sparsity(C_marmoset,row_sparsity)   # Load marmoset cortex-to-cortex
            L_human, landmarks_human = generate_cortex_to_landmark(rois,'human')                                  # Load human cortex-to-homologous ROIs
            L_marmoset, landmarks_marmoset = generate_cortex_to_landmark(rois,'marmoset')                         # Load marmoset cortex-to-homologous ROIs
            # Remove negative connections
            L_human = np.multiply(L_human>0,L_human)
            L_marmoset = np.multiply(L_marmoset>0,L_marmoset)
            # Save
            save_matrices(C_human,'1_C_cortex_human.jpg',outdir,_aspect=None)
            save_matrices(C_marmoset,'1_C_cortex_marmoset.jpg',outdir,_aspect=None)
            save_matrices(L_human,'1_C_roi_human.jpg',outdir)
            save_matrices(L_marmoset,'1_C_roi_marmoset.jpg',outdir)

            # Get n_vertices
            n_vertices_human = C_human.shape[0]
            n_vertices_marmoset = C_marmoset.shape[0]

            """
            Compute similarity matrix
             - Vertex-landmark similarity matrix [marmoset, S_marmoset]
             - Vertex-landmark similarity matrix [human, S_human]

             - Similarity matrix in marmoset [W_marmoset]
             - Similarity matrix in human [W_human]
             - Cross-species similarity matrix [W_(human_to_marmoset)]
            """
            
            from sklearn.metrics.pairwise import cosine_similarity
            S_human = cosine_similarity(C_human,L_human.T)
            S_marmoset = cosine_similarity(C_marmoset,L_marmoset.T)
            # Save
            save_matrices(S_human,'2_S_roi_human.jpg',outdir)
            save_matrices(S_marmoset,'2_S_roi_marmoset.jpg',outdir)

            W_human = cosine_similarity(C_human)
            W_marmoset = cosine_similarity(C_marmoset)
            W_human_to_marmoset = cosine_similarity(S_human,S_marmoset)
            # Save
            save_matrices(W_human,'3_W_human.jpg',outdir,_aspect=None,_vmax=.8)
            save_matrices(W_marmoset,'3_W_marmoset.jpg',outdir,_aspect=None,_vmax=.8)
            save_matrices(W_human_to_marmoset,'3_W_human2marmoset.jpg',outdir,_aspect=None,_vmax=1)

            def create_joint_similarity_matrix(W_human,W_marmoset,W_human_to_marmoset):

                return np.concatenate((np.concatenate((np.eye(W_human.shape[0]),W_human_to_marmoset),axis=1),
                                       np.concatenate((W_human_to_marmoset.T,np.eye(W_marmoset.shape[0])),axis=1)),
                                       axis=0)

            W_joint = create_joint_similarity_matrix(W_human,W_marmoset,W_human_to_marmoset) 
            # Save
            save_matrices(W_joint,'4_W_joint.jpg',outdir,_aspect=None,_vmax=1.)

            # Check to see where degree matrix of `W_joint` is negative (degree should never be zero)
            negative_degree_mapping = np.zeros((n_vertices_human+n_vertices_marmoset,2))
            negative_degree_coords = np.where(W_joint.sum(1)<0)
            negative_degree_mapping[negative_degree_coords] = 1
            negative_degree_human_mapping = negative_degree_mapping[:n_vertices_human,]
            negative_degree_marmoset_mapping = negative_degree_mapping[-n_vertices_marmoset:,]

            """
            Create nifti_converters to add_sim_sparsity to W_[monkey|human|human_to_monkey]
            """
            nii_human = "data/data_human/100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.nii.gz"
            nii_marmoset = "data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.nii.gz"
            roi_human, mask_human = "data/data_human/cortex.nii.gz", "data/data_human/cortex.nii.gz"
            roi_marmoset, mask_marmoset = "data/data_marmoset/surfFS.atlasroi.10k.nii.gz", "data/data_marmoset/surfFS.atlasroi.10k.nii.gz"
            # Human nifti-converter
            sparsity_threshold_ = 1.
            if sparsity_threshold_ == 1.:
                sparsity_rule_ = 'none'
            else:
                sparsity_rule_ = 'node'
            # For saving connectivity and landmark masks
            settings = Namespace(roi=roi_human,
                                 mask=mask_human,
                                 corr_sparsity_threshold=1.,
                                 corr_sparsity_rule='none',
                                 sim_sparsity_threshold=sparsity_threshold_,
                                 sim_sparsity_rule=sparsity_rule_,
                                 n_dims=len(rois),
                                 norm_flag=False,
                                 embedding_method='LE',
                                 subset_idx_10k = 0
                                 )
            nifti_machine_human_landmarks = NiftiConverter(nii_human,settings)
            # For saving embeddings
            settings = Namespace(roi=roi_human,
                                 mask=mask_human,
                                 corr_sparsity_threshold=1.,
                                 corr_sparsity_rule='none',
                                 sim_sparsity_threshold=sparsity_threshold_,
                                 sim_sparsity_rule=sparsity_rule_,
                                 n_dims=13,
                                 norm_flag=False,
                                 embedding_method='LE',
                                 subset_idx_10k = 0
                                 )
            nifti_machine_human = NiftiConverter(nii_human,settings)
            # Marmoset nifti-converter
            # For saving connectivity and landmark masks
            settings = Namespace(roi=roi_marmoset,
                                 mask=mask_marmoset,
                                 corr_sparsity_threshold=1.,
                                 corr_sparsity_rule='none',
                                 sim_sparsity_threshold=sparsity_threshold_,
                                 sim_sparsity_rule=sparsity_rule_,
                                 n_dims=len(rois),
                                 norm_flag=False,
                                 embedding_method='LE',
                                 subset_idx_10k = 0
                                 )
            nifti_machine_marmoset_landmarks = NiftiConverter(nii_marmoset,settings)
            # For saving embeddings
            settings = Namespace(roi=roi_marmoset,
                                 mask=mask_marmoset,
                                 corr_sparsity_threshold=1.,
                                 corr_sparsity_rule='none',
                                 sim_sparsity_threshold=sparsity_threshold_,
                                 sim_sparsity_rule=sparsity_rule_,
                                 n_dims=13,
                                 norm_flag=False,
                                 embedding_method='LE',
                                 subset_idx_10k = 0
                                 )
            nifti_machine_marmoset = NiftiConverter(nii_marmoset,settings)
            """
            Create graph Laplacian
            """
            print('Creating graph Laplacian.')
            adj = nifti_machine_human.add_sim_sparsity(W_joint)
            L = nifti_machine_human.adjacency_to_graph_laplacian(adj)
            save_matrices(L,'5_Graph.jpg',outdir,_aspect=None,_vmax=.5)

            """
            Create embedding
             - Split into human and marmoset components (for visualization)
            """
            print('Creating joint embeddings.')
            from src.embedding_machinery import LaplacianEigenmaps

            embedding_machine = LaplacianEigenmaps(L,settings)
            embedding_machine.create_embedding()

            joint_gradients = embedding_machine.eigen_vectors
            ## normalize [0,1]
            joint_gradients = (joint_gradients-joint_gradients.min(0)) / (joint_gradients.max(0)-joint_gradients.min(0))
            save_matrices(joint_gradients[:,:13],'6_joint_gradients.jpg',outdir,_vmax=1.)
            # split joint gradients in order to map onto each species
            joint_gradients_human = joint_gradients[:n_vertices_human,:]
            joint_gradients_marmoset = joint_gradients[-n_vertices_marmoset:,:]
           
            """
            Save joint_gradients_[human|marmoset]
            """
            print(f"Saving all data to {outdir}")
            # save joint embeddings
            nifti_machine_human.save_embedding(joint_gradients_human,out_human)
            nifti_machine_marmoset.save_embedding(joint_gradients_marmoset,out_marmoset)
            nifti_machine_human_landmarks.save_embedding(L_human,out_human_landmark_rsfc)
            nifti_machine_marmoset_landmarks.save_embedding(L_marmoset,out_marmoset_landmark_rsfc)
            nifti_machine_human_landmarks.save_embedding(landmarks_human,out_human_landmark_mask)
            nifti_machine_marmoset_landmarks.save_embedding(landmarks_marmoset,out_marmoset_landmark_mask)
            # cmd params
            singularity_container = "singularity exec -B /mnt/WD10TB /home/geoff/Desktop/containers/fmriprep_ciftify-1.3.2-2.3.3"
            dtseries_human = "data/data_human/100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.dtseries.nii"
            dtseries_marmoset = "data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.dtseries.nii"
            # nifti to dscalar cmds
            for out_human_ in [out_human,out_human_landmark_rsfc,out_human_landmark_mask]:
                cmd = f"{singularity_container} wb_command -cifti-convert -from-nifti {out_human_} {dtseries_human} {out_human_.replace('nii.gz','dscalar.nii')} -reset-scalars"
                print(cmd); os.system(cmd)
            for out_marmoset_ in [out_marmoset,out_marmoset_landmark_rsfc,out_marmoset_landmark_mask]:
                cmd = f"{singularity_container} wb_command -cifti-convert -from-nifti {out_marmoset_} {dtseries_marmoset} {out_marmoset_.replace('nii.gz','dscalar.nii')} -reset-scalars"
                print(cmd); os.system(cmd)

# CIFTI_PALETTE
            cmd = f"{singularity_container} wb_command -cifti-palette {out_human.replace('nii.gz','dscalar.nii')} MODE_USER_SCALE {out_human.replace('nii.gz','dscalar.nii')} -disp-neg FALSE -pos-user 0 1 -palette-name JET256"
            print(cmd); os.system(cmd)
            cmd = f"{singularity_container} wb_command -cifti-palette {out_marmoset.replace('nii.gz','dscalar.nii')} MODE_USER_SCALE {out_marmoset.replace('nii.gz','dscalar.nii')} -disp-neg FALSE -pos-user 0 1 -palette-name JET256"
            print(cmd); os.system(cmd)

            print("Done")
