from __future__ import print_function
import numpy as np
from scipy import sparse
import h5py
from scipy.ndimage import measurements,morphology

#Loading of the images
with h5py.File('sample_A_20160501.hdf','r') as hf:
    raw = hf.get('volumes/raw')
    data_image = np.array(raw)
    del raw 
    cleft = hf.get('volumes/labels/clefts')
    data_cleft = np.array(cleft)
    del cleft 
    neurons = hf.get('volumes/labels/neuron_ids')
    data_neurons = np.array(neurons)
    del neurons
    locations = hf.get('annotations/locations')
    data_locations = np.array(locations)
    del locations
    types = hf.get('annotations/types')
    data_types = np.array(types)
    del types

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
synaptic_map = np.zeros((125,1250,1250),dtype=np.int32)  
synaptic_map[(pre_syn_ind[:,0],pre_syn_ind[:,1],pre_syn_ind[:,2])] = -1   
synaptic_map[(pos_syn_ind[:,0],pos_syn_ind[:,1],pos_syn_ind[:,2])] = +1                        

#Labeling of the clefts
im = 1*(data_cleft<100000000)
del data_cleft
labels, nbr_clefts = measurements.label(im)
labels = np.array(labels,dtype=np.int32)
del im
print (nbr_clefts)
neurons_ids = np.unique(data_neurons)
nbr_neurons = len(neurons_ids)
print (nbr_neurons)

 
    
