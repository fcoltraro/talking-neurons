import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import measurements,morphology
import scipy.stats as stats
import operator


def find_neurons(rad,cleft_num):
    neuron_box, dist_cleft = find_dist(cleft_num)
    neigh_ind = np.nonzero(dist_cleft<=rad)
    return np.unique(neuron_box[neigh_ind])   

def find_pre_pos(rad,cleft_num):
    neuron_box, dist_cleft, synaptic_box = find_dist(cleft_num,syn_map=True)
    neigh_ind = np.nonzero(dist_cleft<=rad)
    pre = np.nonzero(synaptic_box[neigh_ind]==-1)
    pre_ind = (neigh_ind[0][pre],neigh_ind[1][pre],neigh_ind[2][pre])
    pre_loc = np.array(pre_ind).T
    nmb_pre = len(pre_ind[0])
    pre_neurons = neuron_box[pre_ind]; 
    pos = np.nonzero(synaptic_box[neigh_ind]==1)
    pos_ind = (neigh_ind[0][pos],neigh_ind[1][pos],neigh_ind[2][pos])
    pos_loc = np.array(pos_ind).T
    nmb_pos = len(pos_ind[0])
    pos_neurons = neuron_box[pos_ind]; 
    if nmb_pre==0 or nmb_pos==0:
        return set({})
    else:
        pre_loc[:,0] = pre_loc[:,0]*40.0; pos_loc[:,0] = pos_loc[:,0]*40.0
        pre_loc[:,1] = pre_loc[:,1]*4.0;  pos_loc[:,1] = pos_loc[:,1]*4.0
        pre_loc[:,2] = pre_loc[:,2]*4.0;  pos_loc[:,2] = pos_loc[:,2]*4.0
        
        pairs = set({})
        for i in range(0,nmb_pre):
            presyn_i = pre_neurons[i]
            loc_i = pre_loc[i,:]
            dist_i = np.sum((pos_loc - loc_i)**2,axis=1)
            min_index, min_value = min(enumerate(dist_i), key=operator.itemgetter(1))
            possyn_i = pos_neurons[min_index]
            pairs = pairs.union({(presyn_i,possyn_i)})
            pos_loc[min_index] = np.inf
        return pairs   
         
def f1_score(pairs_pred,pairs_real):
    correct_pairs = pairs_pred.intersection(pairs_real)
    nmb_correct = len(correct_pairs); 
    if nmb_correct==0:
        return 1.0
    else:
        nmb_pred = float(len(pairs_pred)); nmb_real = float(len(pairs_real));
        presicion = nmb_correct/nmb_pred
        recall = nmb_correct/nmb_real
        f1 = 2./((1./presicion) + (1./recall))
        return 1 - f1