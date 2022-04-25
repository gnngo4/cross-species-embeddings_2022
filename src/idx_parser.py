"""
idx parser class
Used to extract idx from dlabel.nii
related to specific ROIs of a 10k surface.
"""
import os
import nibabel as nib
import numpy as np

class idx_surface_parser:
    """
    Convert 10k surface dlabel into idx to parse npy row/columns
    """
    def __init__(self,dlabel,labels,settings):
        self.settings = settings

        self.dlabel = dlabel
        self.labels = labels.split('+')
        self.cortex_10k = self.settings.mask

    def generate_random_filename(self,outtype):
        import uuid
        if outtype == 'dscalar':
            return (str(uuid.uuid4())+".dscalar.nii")
        elif outtype == 'nifti':
            return (str(uuid.uuid4())+".nii.gz")
        else:
            NotImplemented
    
    def read_nifti(self,nifti):

        x = nib.load(nifti)
        self.affine = x.affine

        return x.get_fdata()

    def create_roi(self):

        out_labels = []
        for label in self.labels:
            cmds = []
            out_label = self.generate_random_filename('dscalar')
            out_labels.append(out_label.replace('dscalar.nii','nii.gz'))
            cmd = f"{self.settings.singularity_container} wb_command -cifti-label-to-roi {self.dlabel} {out_label} -name {label}"
            cmds.append(cmd)
            cmd = f"{self.settings.singularity_container} wb_command -cifti-create-dense-from-template {self.settings.input_rfmri_template_dtseries} {out_label} -cifti {out_label}"
            cmds.append(cmd)
            cmd = f"{self.settings.singularity_container} wb_command -cifti-convert -to-nifti {out_label} {out_label.replace('dscalar.nii','nii.gz')}"
            cmds.append(cmd)
        
            for cmd in cmds:
                os.system(cmd)
        
        first = True
        outfile = self.generate_random_filename('nifti')
        for out_label in out_labels:
            if first:
                roi = self.read_nifti(out_label)
                first = False
            else:
                roi += self.read_nifti(out_label)
        roi = nib.Nifti1Image(roi,self.affine)
        nib.save(roi,outfile)
        self.outfile = outfile
        # Clean up
        for i in out_labels:
            os.remove(i)
            os.remove(i.replace('nii.gz','dscalar.nii'))

    def get_roi_idx(self):
        # Create label roi
        self.create_roi()
        self.cortex_10k = self.read_nifti(self.cortex_10k)
        coords = np.where(self.cortex_10k == 1)[0]

        # Read roi
        data = self.read_nifti(self.outfile)
        # clean-up
        os.remove(self.outfile)
        # return idx
        idx_npy = np.where(data[coords][:,0,0,0]==1)[0]
        idx_10k = np.where(data[:,0,0,0]==1)[0]

        return idx_npy,idx_10k
