"""Running ConnectomeEmbedding Machine"""

import os,sys
import numpy as np
from param_parser import parameter_parser
from idx_parser import idx_surface_parser
from nifti_converter import NiftiConverter
from embedding_machinery import LaplacianEigenmaps 
from embedding_machinery import GraphWave

def convert_to_cifti(settings):
    out = settings.output.replace('nii.gz','dscalar.nii')
    cmd = f"{settings.singularity_container} wb_command -cifti-convert -from-nifti {settings.output} {settings.input_rfmri_template_dtseries} {out} -reset-scalars"
    print(f"Convert nifti-output to cifti-format.\n[COMMAND]:{cmd}")
    os.system(cmd)

def generate_graph_laplacian(nifti_machine,settings):
    """
    Read either (1) rfmri data [.nii.gz] or (2) pre-generated
    adjacency matrix [.npy] to generate the graph Laplacian
    """
    if settings.input_type == 'npy':
        R = np.load(settings.input)
        if isinstance(settings.subset_idx_npy,np.ndarray):
            R = R[settings.subset_idx_npy,:][:,settings.subset_idx_npy]
            if settings.sim_sparsity_rule == 'preload':
                print("sim_sparsity_rule can't be preload when subsetting npy.\nExiting.")
                sys.exit()
        R = nifti_machine.add_corr_sparsity(R)
        S = nifti_machine.similarity_matrix(R)
        adj = nifti_machine.add_sim_sparsity(S)
        L = nifti_machine.adjacency_to_graph_laplacian(adj)
    else:
        nifti_machine.create_graph_laplacian()
        L = nifti_machine.L

    return L

if __name__ == "__main__":
    settings = parameter_parser()

    """
    INTEGRATE idx_converter here
    """
    if (settings.subgraph_10k) and settings.input_type == 'npy':
        print(f"Processing subset of npy.\nROIs: {settings.dlabel_rois}")    
        surface_idx_machine = idx_surface_parser(settings.dlabel_10k,settings.dlabel_rois,settings)
        settings.subset_idx_npy, settings.subset_idx_10k = surface_idx_machine.get_roi_idx()
    elif settings.input_type == 'npy':
        print('Processing full npy.')
        settings.subset_idx_npy = 0
        settings.subset_idx_10k = 0
    else:
        settings.subset_idx_npy = 0
        settings.subset_idx_10k = 0
        pass
    
    """
    Set up nifti_machine
    """
    if settings.input_type == 'npy':
        nifti_machine = NiftiConverter(settings.input_rfmri_template,settings)
    else:
        nifti_machine = NiftiConverter(settings.input,settings)

    """
    Set up graph Laplacian
    """
    print('Generating graph Laplacian.')
    L = generate_graph_laplacian(nifti_machine,settings)
    if settings.embedding_method == 'LE':
        algo_name = 'Laplacian eigenmaps'
        embedding_machine = LaplacianEigenmaps(L,settings)
    elif settings.embedding_method == 'GW':
        algo_name = 'Graph wave'
        embedding_machine = GraphWave(L,settings)
    else:
        NotImplemented

    print(f"Running [{algo_name}]: Creating embedding.")
    if settings.embedding_method == 'LE':
        embedding_machine.create_embedding()
        print(embedding_machine.eigen_values)
        nifti_machine.save_embedding(embedding_machine.eigen_vectors,settings.output)
        convert_to_cifti(settings)
    elif settings.embedding_method == 'GW':
        embedding_machine.create_embedding()
        nifti_machine.save_embedding(embedding_machine.real_and_imaginary,settings.output)
        convert_to_cifti(settings)
    else:
        NotImplemented
