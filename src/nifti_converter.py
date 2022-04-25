"""Nifti to graph converter class."""

import nibabel as nib
import numpy as np
import networkx as nx
import os
import sys
from tqdm import tqdm

from sklearn.decomposition import PCA

class NiftiConverter:
    """
    Converter to transform a nifti (*nii.gz) into a graph (G).
    """
    def __init__(self,nii,settings):
        """
        Initialization.
        :param nii: Input (*nii.gz) rfMRI path.
        :param settings: argparse object with settings.
        """
        self.settings = settings
        
        self.nii = nii
        self.nii_A = self.settings.roi
        self.nii_B = self.settings.mask
        self.corr_threshold = self.settings.corr_sparsity_threshold
        self.corr_sparsity_rule = self.settings.corr_sparsity_rule
        self.sim_threshold = self.settings.sim_sparsity_threshold
        self.sim_sparsity_rule = self.settings.sim_sparsity_rule
        self.n_dims = self.settings.n_dims
        self.norm_flag = self.settings.norm_flag
        self.embedding_method = self.settings.embedding_method
        self.coords = dict()
        
        req = [os.path.exists(x) for x in [self.nii,self.nii_A,self.nii_B]]
        if False in req:
            print([self.nii,self.nii_A,self.nii_B])
            sys.exit(0)
        
    def read_nifti(self,nifti):

        x = nib.load(nifti)
        self.affine = x.affine

        return x.get_fdata()

    def mask_rfmri(self):
        roi = dict()
        data = dict()

        rfmri_data = self.read_nifti(self.nii)
        roi['roi'] = self.read_nifti(self.nii_A)
        roi['mask'] = self.read_nifti(self.nii_B)
        for roi_, data_ in roi.items():
            self.coords[roi_] = np.where(data_==1)
            try:
                data[roi_] = np.squeeze(rfmri_data[self.coords[roi_]])
            except:
                print(f"Problem extracting timeseries from {roi_}.")

        return data

    def preprocess(self,data):

        Z_data = dict()
        for roi_, data_ in data.items():
            mean = data_.mean(1,keepdims=True)
            std = data_.std(1,keepdims=True)
            Z_data[roi_] = (data_-mean)/std

        return Z_data

    def correlation_matrix(self,Z_data):

        X = Z_data['roi']
        Y = Z_data['mask']
        
        return np.corrcoef(X,Y)[:X.shape[0],-Y.shape[0]:]

    def add_corr_sparsity(self,C):
        # Add sparsity to correlation matrix
        if self.corr_sparsity_rule == 'row':
            C_sp = np.zeros((C.shape))
            K = int(C_sp.shape[0]*self.corr_threshold)
            for row_idx in range(C_sp.shape[0]):
                C_ = C[row_idx,:]
                min_ = C_[np.argpartition(C_,-K)[-K:]].min()
                C_sp[row_idx,:] += (C_>min_)
            adj = np.divide(C_sp, C_sp, out=np.zeros_like(C_sp), where=C_sp!=0)
            C_sp = np.multiply(C,C_sp)
            # Remove negative values
            if (C_sp< 0).sum() > 0:
                print('Warning: Removing negative correlation values.')
                C_sp = np.multiply(C_sp>0,C_sp)
            else:
                pass
            return C_sp
        elif self.corr_sparsity_rule == 'none':
            return C
        else:
            print(f"corr_sparsity_rule:{self.corr_sparsity_rule} does not exist.\nExiting.")
            sys.exit()

    def similarity_matrix(self,X):
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        return cosine_similarity(X)

    def add_sim_sparsity(self,S):
        # Add sparsity to similarity matrix
        if self.sim_sparsity_rule == 'full':
            S_ = S.flatten()
            K = int(S_.shape[0]*self.sim_threshold)
            min_ = S_[np.argpartition(S_,-K)[-K:]].min()
            adj = np.multiply(S,S>min_)
        elif self.sim_sparsity_rule == 'node':
            adj = np.zeros((S.shape))
            K = int(S.shape[0]*self.sim_threshold)
            for row_idx in range(S.shape[0]):
                S_ = S[row_idx,:]
                min_ = S_[np.argpartition(S_,-K)[-K:]].min()
                adj[row_idx,:] += (S_>min_)
                adj[:,row_idx] += (S_>min_)
            adj = np.divide(adj, adj, out=np.zeros_like(adj), where=adj!=0)
            adj = np.multiply(S,adj)
        elif self.sim_sparsity_rule == 'kNN':
            K = int(S.shape[0]*self.sim_threshold)
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=K,algorithm='ball_tree').fit(S)
            adj = nbrs.kneighbors_graph(S).toarray()
            adj = np.multiply(adj,S)
            for i in tqdm(range(adj.shape[0])):
                for j in range(adj.shape[0]):
                    if adj[i,j] != adj[j,i]:
                        if adj[i,j] == 0:
                            adj[i,j] = adj[j,i]
                        else:
                            adj[j,i] = adj[i,j]
        elif self.sim_sparsity_rule == 'none':
            np.fill_diagonal(S,0)
            return S
        elif self.sim_sparsity_rule == 'preload':
            """
            data/kNN_test.npy uses 'kNN' method with sparsity of 0.1
            """
            adj = np.load('data/kNN_test.npy')
        else:
            print(f"sim_sparsity_rule:{self.sim_sparsity_rule} does not exist.\nExiting.")
            sys.exit()

        # Remove negative values
        if (adj < 0).sum() > 0:
            print('Warning: Removing negative values.')
            adj = np.multiply(adj>0,adj)
        else:
            pass

        # Remove self-loops
        np.fill_diagonal(adj,0)

        return adj

    def adjacency_to_graph_laplacian(self,adj):
        '''
        Constructs a symmetric normalized Laplacian of a similarity matrix
        L_sym = I - D^(-1/2)AD^(-1/2)
        '''
        n_nodes = adj.shape[0]
        # Create degree matrix, D^(-1/2)
        rowsum = np.array(adj.sum(1))
        D_inv_sqrt = np.power(rowsum,-0.5).flatten()
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_inv_sqrt = np.diag(D_inv_sqrt)
        # Symmetric normalized Laplacian
        L_sym = np.eye(n_nodes) - np.dot(np.dot(D_inv_sqrt,adj),D_inv_sqrt)

        return L_sym

    def adjacency_to_graph(self,adj):
        '''
        Converts adjacency matrix to a networkx 
        graph object.
        * Prints a warning if the graph is not
        connected.
        '''

        G = nx.from_numpy_matrix(adj)
        if nx.is_connected(G):
            pass
        else:
            print("Warning: Graph is not connected.")
        
        return G

    def create_graph_laplacian(self):
        '''
        Creates a normalized graph Laplacian numpy 
        array matrix.
        '''
        data = self.mask_rfmri()
        Z_data = self.preprocess(data)
        R = self.correlation_matrix(Z_data)
        R = self.add_corr_sparsity(R)
        S = self.similarity_matrix(R)
        adj = self.add_sim_sparsity(S)
        self.L = self.adjacency_to_graph_laplacian(adj)

    def create_graph(self):
        '''
        Creates a networkx graph object from the rfmri,
        roi, and mask nifti data.
        '''
        data = self.mask_rfmri()
        Z_data = self.preprocess(data)
        R = self.correlation_matrix(Z_data)
        R = self.add_corr_sparsity(R)
        S = self.similarity_matrix(R)
        adj = self.add_sim_sparsity(S)
        self.G = self.adjacency_to_graph(adj)

    def choose_coords(self):
        """
        Choose coords to save-to
        """
        if isinstance(self.settings.subset_idx_10k,np.ndarray):
            self.save_coords = (self.settings.subset_idx_10k,)
        else:
            _ = self.mask_rfmri()
            self.save_coords = self.coords['roi']

    def save_embedding(self,embeddings,outfile):
        """
        Save embeddings as nifti
        """
        
        self.choose_coords() # Initialize self.save_coords
        
        if self.embedding_method == "LE":
            embeddings_ = self.read_nifti(self.nii)[:,:,:,:self.n_dims] * 0
            if self.norm_flag:
                embeddings = (embeddings-embeddings.min(0)) / (embeddings.max(0)-embeddings.min(0))
            for n in range(self.n_dims):
                # Match dimensions of eigvec with embeddings
                emb = np.expand_dims(embeddings[:,n],axis=(1,2))
                embeddings_[:,:,:,n][self.save_coords] = emb
            affine = self.affine
            embeddings_ = nib.Nifti1Image(embeddings_,affine)
            nib.save(embeddings_,outfile)

        elif self.embedding_method == "GW":
            embeddings_ = self.read_nifti(self.nii)[:,:,:,:self.n_dims] * 0
            re = embeddings.real 
            im = embeddings.imag
            struct_emb = np.concatenate((re,im),axis=1)
            pca = PCA()
            pca.fit(struct_emb)
            pca_emb = pca.transform(struct_emb)
            embeddings = pca_emb
            if self.norm_flag:
                embeddings = (embeddings-embeddings.min(0)) / (embeddings.max(0)-embeddings.min(0))
            for n in range(self.n_dims):
                # Match dimensions of eigvec with embeddings
                emb = np.expand_dims(embeddings[:,n],axis=(1,2))
                embeddings_[:,:,:,n][self.save_coords] = emb
            affine = self.affine
            embeddings_ = nib.Nifti1Image(embeddings_,affine)
            nib.save(embeddings_,outfile)

        else:
            NotImplemented

    def save_npy(self,embeddings,outfile):
        '''
        Save npy as nifti
        '''
        _ = self.mask_rfmri() # Initialize self.coords
        embeddings_ = self.read_nifti(self.nii)[:,:,:,0]
        # Match dimensions of eigvec with embeddings
        embeddings_[:,0,0][self.save_coords] = embeddings
        affine = self.affine
        embeddings_ = nib.Nifti1Image(embeddings_,affine)
        nib.save(embeddings_,outfile)
