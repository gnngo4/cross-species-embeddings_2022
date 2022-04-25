"""Parsing up the command line parameters."""

import argparse

def numrange(string):
    try:
        start = int(string.split('-')[0])
        end = int(string.split('-')[1])
        start -= 1
        return range(start,end)
    except:
        return True

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run embedding algorithm.")

    parser.add_argument("--input-type",
                        nargs="?",
                        default="npy",
                        help="Input type ([rfmri].nii.gz or [adjacency].npy). Default is npy.")
    parser.add_argument("--input",
                        nargs="?",
                        default="./data/100UR.cortex-to-cortex.npy",
                        help="Path to input data. Default is 100UR.cortex-to-cortex.npy.")
    parser.add_argument("--input-rfmri-template",
                        nargs="?",
                        default="./data/100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.nii.gz",
                        help="Path to template rfmri data [nii.gz]. Default is 100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.nii.gz.")
    parser.add_argument("--input-rfmri-template-dtseries",
                        nargs="?",
                        default="./data/100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.dtseries.nii",
                        help="Path to template rfmri data [dtseries.nii]. Default is 100307.rfMRI_REST1_LR.wmcsf_reg.s6.cortex.dtseries.nii")
    parser.add_argument("--roi",
                        nargs="?",
                        default="./data/cortex.nii.gz",
                        help="Path to roi data. Default is cortex.nii.gz.")
    parser.add_argument("--mask",
                        nargs="?",
                        default="./data/cortex.nii.gz",
                        help="Path to mask data. Default is cortex.nii.gz.")
    parser.add_argument("--output",
                        nargs="?",
                        default="./data/100307.rfMRI_REST1_LR.cortex-to-cortex.nii.gz")
    parser.add_argument("--embedding-method",
                        nargs="?",
                        default="LE",
                        help="Embedding method. Default is LE.")
    parser.add_argument("--n_dims",
                        type=int,
                        default=100,
                        help="Number of saved embedding dimensions. Default is 100.")
    parser.add_argument("--norm_flag",
                        action='store_true',
                        help="Normalise embedding dimensions between [0,1]. Default is False")
    parser.add_argument("--singularity-container",
                        nargs="?",
                        default="singularity exec -B /mnt/WD10TB /home/geoff/Desktop/containers/fmriprep_ciftify-1.3.2-2.3.3",
                        help="Singularity container exec command. Default is fmriprep_ciftify-1.3.2-2.3.3.")
    """
    Graph generation OPTIONS
    """
    parser.add_argument("--corr-sparsity-threshold",
                        type=float,
                        default=0.1,
                        help="Correlation sparsity threshold. Default is 0.1.")
    parser.add_argument("--corr-sparsity-rule",
                        nargs="?",
                        default="row",
                        help="Correlation sparsity threshold rule [row/none]. Default is row.")
    parser.add_argument("--sim-sparsity-threshold",
                        type=float,
                        default=0.1,
                        help="Adjacency sparsity threshold. Default is 0.1.")
    parser.add_argument("--sim-sparsity-rule",
                        nargs="?",
                        default="node",
                        help="Adjacency sparsity threshold rule [node/kNN/full]. Default is node.")
    """ 
    idx surface parser OPTIONS
    """
    parser.add_argument("--subgraph-10k",
                        action='store_true',
                        help="Extract subset adjacency matrix from full cortex-to-cortex adjacency matrix. Default is False")
    parser.add_argument("--dlabel-10k",
                        nargs="?",
                        default="atlas/10k/RSN-networks.10k_fs_LR.dlabel.nii",
                        help="Atlas [dlabel.nii] file. Default is RSN-networks.10k_fs_LR.dlabel.nii")
    parser.add_argument("--dlabel-rois",
                        nargs="?",
                        default="7Networks_1",
                        help="Name of ROIs in dlabel files [extra ROIs are + separated]. Default is 7Networks_1")
    """
    Laplacian eigenmap OPTIONS
    """
    """
    GraphWave OPTIONS
    """
    parser.add_argument("--heat-coefficient-auto-min",
                        action='store_true',
                        help="Calculate the theoretical minimum heat kernel exponent. Default is False")
    parser.add_argument("--heat-coefficient-auto-max",
                        action='store_true',
                        help="Calculate the theoretical maximum heat kernel exponent. Default is False")
    parser.add_argument("--heat-coefficient",
                        type=float,
                        default=1000.0,
                        help="Heat kernel exponent. Default is 1000.0.")
    parser.add_argument("--sample-number",
                        type=int,
                        default=50,
                        help="Number of characteristic function sample points. Default is 50.")
    parser.add_argument("--step-size",
                        type=float,
                        default=20,
                        help="Number of steps. Default is 20.")
    parser.add_argument("--gw-range-nodes",
                        type=numrange,
                        default='',
                        help="Process wavelet coefficients for a range of nodes between. Default is to process all nodes.")

    return parser.parse_args()
