"""
class for joint embedding analysis
"""

import nibabel as nib
import numpy as np
import os
import sys
import tempfile

def read_nifti(nifti):

    return nib.load(nifti).get_fdata()

class joint_gradient_landmarks:
    """
    Extract joint gradient values from each species
    """
    def __init__(self,jointgrad_dir,landmark_dir,roi_list,n_gradients):
        """
        Initialization.
        :param jointgrad_dir (directory with joint gradients)
        :param landmark_dir (directory with landmark/homologous rois)
        :param roi_list (list of all rois for plotting)
        """
        self.jointgrad_dir = jointgrad_dir
        self.landmark_dir = landmark_dir
        self.roi_list = roi_list
        self.n_gradients = n_gradients
        self.tmpdir = tempfile.TemporaryDirectory()
        self.sing_container = "singularity exec -B /mnt/WD10TB /home/geoff/Desktop/containers/fmriprep_ciftify-1.3.2-2.3.3"
        self.cmds = []

        print(f"Output dir: {self.tmpdir.name}")
        print(f"Singularity container: {self.sing_container}")

        # intermediate
        self.joint_human_dscalar = os.path.join(jointgrad_dir,"joint_eigenmap_human.dscalar.nii")
        self.joint_marmoset_dscalar = os.path.join(jointgrad_dir,"joint_eigenmap_marmoset.dscalar.nii")
        self.joint_human_nifti = os.path.join(self.tmpdir.name,"joint_eigenmap_human.nii.gz")
        self.joint_marmoset_nifti = os.path.join(self.tmpdir.name,"joint_eigenmap_marmoset.nii.gz")
        self.n_rois = len(roi_list)
        self.landmark_human_dscalar = [f"{self.landmark_dir}/human_{roi}.dscalar.nii" for roi in self.roi_list]
        self.landmark_marmoset_dscalar = [f"{self.landmark_dir}/marmoset_{roi}.dscalar.nii" for roi in self.roi_list]
        self.landmark_human_nifti = [f"{self.tmpdir.name}/human_{roi}.nii.gz" for roi in self.roi_list]
        self.landmark_marmoset_nifti = [f"{self.tmpdir.name}/marmoset_{roi}.nii.gz" for roi in self.roi_list]
        # set-up tmpdir
        cmd = f"{self.sing_container} wb_command -cifti-convert -to-nifti {self.joint_human_dscalar} {self.joint_human_nifti}"
        self.cmds.append(cmd)
        cmd = f"{self.sing_container} wb_command -cifti-convert -to-nifti {self.joint_marmoset_dscalar} {self.joint_marmoset_nifti}"
        self.cmds.append(cmd)
        for i in range(self.n_rois):
            cmd = f"{self.sing_container} wb_command -cifti-convert -to-nifti {self.landmark_human_dscalar[i]} {self.landmark_human_nifti[i]}"
            self.cmds.append(cmd)
            cmd = f"{self.sing_container} wb_command -cifti-convert -to-nifti {self.landmark_marmoset_dscalar[i]} {self.landmark_marmoset_nifti[i]}"
            self.cmds.append(cmd)
        for cmd in self.cmds:
            os.system(cmd)

    def read_gradients_with_roi(self,roi,gradient):
        """
        Extract gradient value for a single roi
        returns (1 X n_gradients)
        """
        roi_coords = np.where(read_nifti(roi) == 1)[:3]
        gradient_nii = read_nifti(gradient)
        gradient_values = np.zeros((self.n_gradients,))
        for i in range(self.n_gradients+1):
            if i == 0:
                continue
            else:
                gradient_values[i-1] = gradient_nii[:,:,:,i][roi_coords].mean()
        """
        CHECK AVERAGE GRADIENT VALUES FOR THE ROI | USE self.n_gradients to extract finite values 
        """
        return gradient_values

    def get_data(self):
        """
        Extract gradient values across 
        all rois and stores to 
        [.roi_gradient_values: (n_rois X n_gradients)]
        """
        self.roi_gradient_values = {}
        self.roi_gradient_values['human'] = np.zeros((self.n_rois,self.n_gradients))
        self.roi_gradient_values['marmoset'] = np.zeros((self.n_rois,self.n_gradients))
        for i in range(self.n_rois):
            print(self.landmark_human_nifti[i],self.joint_human_nifti)
            self.roi_gradient_values['human'][i,:] = self.read_gradients_with_roi(self.landmark_human_nifti[i],self.joint_human_nifti)
            print(self.landmark_marmoset_nifti[i],self.joint_marmoset_nifti)
            self.roi_gradient_values['marmoset'][i,:] = self.read_gradients_with_roi(self.landmark_marmoset_nifti[i],self.joint_marmoset_nifti)
