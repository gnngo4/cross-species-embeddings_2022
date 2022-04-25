"""connectome embedding class implementations."""

import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm
import os
import sys

class LaplacianEigenmaps:
    '''
    An implementation of Laplacian Eigenmaps
    '''
    def __init__(self,L,settings):
        """
        Initialization.
        :param L: Input numpy array of the graph Laplacian.
        """
        self.L = L

    def create_embedding(self):
        """
        Create embedding
        """
        self.eigen_values, self.eigen_vectors = eigh(self.L)

class GraphWave:
    '''
    An implementation of Graph Wave
    '''
    def __init__(self,L,settings):
        self.L = L
        self.number_of_nodes = self.L.shape[0]
        self.settings = settings

        if isinstance(self.settings.gw_range_nodes,range):
            self.node_list = self.settings.gw_range_nodes
            self.subset_nodes_flag = True
        else:
            self.node_list = range(self.number_of_nodes)
            self.subset_nodes_flag = False

        self.steps = [x*self.settings.step_size for x in range(self.settings.sample_number)]
        
    def calculate_heat_kernel_coefficient(self):
        """
        Calculate optimal heat kernel coefficient value as
        proposed by Donnat et al. 2017.
        """
        eig_2 = self.eigen_values[1]
        eig_N = self.eigen_values[-1]
        if eig_2 < 0:
            print("Lower bound eigen-value is negative.\nExiting.")
            sys.exit(0)
        s_min = -np.log(.95) / np.sqrt(eig_2*eig_N)
        s_max = -np.log(.85) / np.sqrt(eig_2*eig_N)
        if self.settings.heat_coefficient_auto_min:
            self.settings.heat_coefficient = s_min
        elif self.settings.heat_coefficient_auto_max:
            self.settings.heat_coefficient = s_max
        else:
            pass
        print(f"Heat coefficient set: {self.settings.heat_coefficient}")

    def save_real_and_imaginary(self):
        """
        Save wavelet_coefficients if processing
        multiple range of nodes independently.
        """
        start = str(self.settings.gw_range_nodes[0])
        end = str(self.settings.gw_range_nodes[-1])
        suffix = start + '-' + end
        savefile = os.path.join(self.settings.output.replace('nii.gz',suffix+'.npy'))
        print(f"Saving wavelet coefficients to {savefile}\nExiting.")
        np.save(savefile,self.real_and_imaginary)
        sys.exit()

    def single_wavelet_generator(self, node):
        """
        Calculating the characteristic function for a given node, using the eigendecomposition.
        :param node: Node that is being embedded.
        """
        impulse = np.zeros((self.number_of_nodes))
        impulse[node] = 1.0
        diags = np.diag(np.exp(-self.settings.heat_coefficient*self.eigen_values))
        eigen_diag = np.dot(self.eigen_vectors,diags)
        waves = np.dot(eigen_diag, np.transpose(self.eigen_vectors))
        wavelet_coefficients = np.dot(waves,impulse)

        return wavelet_coefficients

    def exact_wavelet_calculator(self):
        """
        Calculates the structural role embedding using the exact eigenvalue decomposition
        """
        self.real_and_imaginary = []
        for node in tqdm(self.node_list):
            wave = self.single_wavelet_generator(node)
            self.steps = [x*self.settings.step_size for x in range(self.settings.sample_number)]
            wavelet_coefficients = [np.mean(np.exp(wave*1.0*step*1j)) for step in self.steps]
            self.real_and_imaginary.append(wavelet_coefficients)
        self.real_and_imaginary = np.array(self.real_and_imaginary)
        if self.subset_nodes_flag:
            self.save_real_and_imaginary()
            
    def exact_structural_wavelet_embedding(self):
        """
        Calculates the eigenvectors, eigenvalues and an exact embedding is created
        """
        self.eigen_values, self.eigen_vectors = eigh(self.L)
        self.calculate_heat_kernel_coefficient()
        self.exact_wavelet_calculator()
        
    def create_embedding(self):
        """
        Create embedding
        """
        self.exact_structural_wavelet_embedding()
