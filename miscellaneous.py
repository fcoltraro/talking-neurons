import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg as LA
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import measurements,morphology

def center_of_mass(neuron_num):
    r = np.nonzero(data_neurons==neuron_num)  
    mass = float(len(r[0]))
    #print mass
    R = sum(r,axis=1)/mass
    return R 
 
def get_coord_neurons(m):
    neurons_coord = np.zeros((m,3))
    for j in range(0,m):
        neurona = neurons_ids[j]
        neurons_coord[j,:] = center_of_mass(neurona)
    return neurons_coord

def features_neuron(neuron_num,cleft_num,rad):  
    data = coord_neuron(neuron_num,cleft_num,rad)
    dims_rescaled_data=3
    m, n = data.shape
    # mean center the data
    # data -= data.mean(axis=0).astype(data.dtype)
    # calculate the covariance/correlation matrix
    R = np.corrcoef(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(data,evecs), evals, evecs