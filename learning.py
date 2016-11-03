import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import measurements,morphology
import scipy.stats as stats

def find_neurons(rad,cleft_num):
    neuron_box, dist_cleft = find_dist(cleft_num)
    neigh_ind = np.nonzero(dist_cleft<=rad)
    return np.unique(neuron_box[neigh_ind])   

def find_pre_pos(rad,cleft_num):
    neuron_box, dist_cleft, synaptic_box = find_dist(cleft_num,syn_map=True)
    neigh_ind = np.nonzero(dist_cleft<=rad)
    pre = np.nonzero(synaptic_box[neigh_ind]==-1)
    pre_ind = (neigh_ind[0][pre],neigh_ind[1][pre],neigh_ind[2][pre])
    pos = np.nonzero(synaptic_box[neigh_ind]==1)
    pos_ind = (neigh_ind[0][pos],neigh_ind[1][pos],neigh_ind[2][pos])
    return pre_ind, neuron_box[pre_ind], pos_ind, neuron_box[pos_ind]  