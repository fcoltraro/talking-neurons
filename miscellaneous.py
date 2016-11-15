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

def heuristic_solver(C, neuron_names):
    n = len(neuron_names)
    C_copy = C.copy()
    neu_neg = 1*(C_copy<0)
    graph = {}
    for k in range(0,n):
        ind_neg = np.nonzero(neu_neg[k,:]==1)
        graph[neuron_names[k]] = neuron_names[ind_neg]
    print graph
    strong_comp = np.array(tarjan(graph))
    print strong_comp
    nOfNodesComp = np.array(map(len, strong_comp))
    cycles = any((nOfNodesComp!=1))
    while cycles==True:
       ind_loops = np.nonzero(nOfNodesComp>1)
       nmb_cycles = len(ind_loops[0])
       for p in range(0,nmb_cycles):
          loop_neurons = strong_comp[ind_loops[0][p]]
          c_max = - np.inf; c_max_ind = (); len_loop = len(loop_neurons)
          for l in range(0,len_loop):
              for m in range(0,len_loop):
                      ind_i = np.nonzero(neuron_names==loop_neurons[l])[0][0]
                      ind_j = np.nonzero(neuron_names==loop_neurons[m])[0][0]
                      c_ij = C_copy[ind_i,ind_j]
                      if l!=m and c_ij > c_max and c_ij < 0:
                         c_max = c_ij
                         c_max_ind = (ind_i,ind_j)
          C_copy[c_max_ind] = 0    
          neuron_i = neuron_names[c_max_ind[0]]; neuron_j = neuron_names[c_max_ind[1]] 
          neuron_delete = np.nonzero(graph[neuron_i]==neuron_j)
          graph[neuron_i] = np.delete(graph[neuron_i],neuron_delete)  
       print graph             
       strong_comp = np.array(tarjan(graph))
       print strong_comp
       nOfNodesComp = np.array(map(len, strong_comp))
       cycles = any((nOfNodesComp!=1))
    return graph

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

def energy(rad,cleft_num,Theta,pairs):
    features = features_neurons(cleft_num,rad)
    C,neurons = cost_function(features,Theta)
    energy = 0.; neurons = np.array(neurons)
    for pair in pairs:
        neu_i = pair[0]; neu_j = pair[1];
        ind_i = np.nonzero(neurons==neu_i)[0][0]; ind_j = np.nonzero(neurons==neu_j)[0][0]
        energy = energy + C[ind_i,ind_j]
    return energy
    