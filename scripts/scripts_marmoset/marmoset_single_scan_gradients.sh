#!/bin/bash

for participant in 10 11 12 14 15 16 18 19 20 21 22 23 24 25 26 27 28 29 30 31
do

  npy=data/data_marmoset/single_subject_gradients/marm-id-${participant}.cortex-to-cortex.corr.npy
  out=marm-id-${participant}.cortex-to-cortex.embedding-LE.thr-corr-0.1.nii.gz
  echo python src/main.py --input-type npy --input ${npy} --input-rfmri-template data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.nii.gz --input-rfmri-template-dtseries data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.dtseries.nii --roi data/data_marmoset/surfFS.atlasroi.10k.nii.gz --mask data/data_marmoset/surfFS.atlasroi.10k.nii.gz --output data/data_marmoset/single_subject_gradients/${out} --embedding-method LE --n_dims 100 --norm_flag --corr-sparsity-threshold 0.1 --corr-sparsity-rule row --sim-sparsity-rule none
  python src/main.py --input-type npy --input ${npy} --input-rfmri-template data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.nii.gz --input-rfmri-template-dtseries data/data_marmoset/sub-m31_dir-down_run-1.rfMRI.wmcsf_reg.s1.5.cortex.dtseries.nii --roi data/data_marmoset/surfFS.atlasroi.10k.nii.gz --mask data/data_marmoset/surfFS.atlasroi.10k.nii.gz --output data/data_marmoset/single_subject_gradients/${out} --embedding-method LE --n_dims 100 --norm_flag --corr-sparsity-threshold 0.1 --corr-sparsity-rule row --sim-sparsity-rule none

done
