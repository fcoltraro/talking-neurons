from __future__ import print_function
import numpy as np
from scipy.sparse import lil_matrix
import h5py
from scipy.ndimage import measurements,morphology

#Loading of the images
with h5py.File('sample_A_20160501.hdf','r') as hf:
    raw = hf.get('volumes/raw')
    data_image = np.array(raw)
    cleft = hf.get('volumes/labels/clefts')
    data_cleft = np.array(cleft)
    neurons = hf.get('volumes/labels/neuron_ids')
    data_neurons = np.array(neurons)
    locations = hf.get('annotations/locations')
    data_locations = np.array(locations)
    types = hf.get('annotations/types')
    data_types = np.array(types)

# Pre and Post synaptic processing
data_types_bol = 1*(data_types==u'presynaptic_site') 
del data_types 
locations_ind = np.zeros((len(data_locations.T[0]),3),dtype=np.int32)
locations_ind[:,0] = map(int,data_locations[:,0]/40.0)
locations_ind[:,1] = map(int,data_locations[:,1]/4.0)
locations_ind[:,2] = map(int,data_locations[:,2]/4.0)
del data_locations 
pre_syn_ind = locations_ind[np.nonzero(data_types_bol==1),:][0]
pos_syn_ind = locations_ind[np.nonzero(data_types_bol==0),:][0]
del locations_ind     
del data_types_bol       
synaptic_map = np.zeros((125,1250,1250),dtype=np.int16)  
synaptic_map[(pre_syn_ind[:,0],pre_syn_ind[:,1],pre_syn_ind[:,2])] = -1   
synaptic_map[(pos_syn_ind[:,0],pos_syn_ind[:,1],pos_syn_ind[:,2])] = +1  
"""
syn_map_sparse = []             
for z in range(synaptic_map.shape[0]):
    syn_map_sparse.append(lil_matrix(synaptic_map[z,:,:]))
del synaptic_map   
"""                    

#Labeling of the clefts
labels, nbr_clefts = measurements.label(1*(data_cleft<100000000))
print (nbr_clefts); del data_cleft; labels = np.array(labels,dtype=np.int16)

#getting the features

all_features = []
all_yprimes = []
rad = 100; num_training = 11;
for cleft in range(1,num_training):
    all_features.append(features_matrix(rad,cleft))
    all_yprimes.append(y_prime(rad,cleft))
    print (cleft)

    
    

"""
labels_sparse = []
for z_coord in range(labels.shape[0]):
    labels_sparse.append(lil_matrix(labels[z_coord,:,:]))
del labels     

#neurons_ids = np.unique(data_neurons)
#nbr_neurons = len(neurons_ids)
#print (nbr_neurons)

"""
 
    
