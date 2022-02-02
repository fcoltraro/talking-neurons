import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg as LA
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import measurements,morphology
from pylab import *

def plot_neigh(rad,cleft_num,i):
    dist_cleft, raw_box, z_min, z_max = find_dist(cleft_num,raw=True,syn_map=False,z_slice=True)
    border_ind = np.nonzero(dist_cleft==rad)
    raw_box[border_ind] = 1
    #pre_ind, pos_ind, pre_neurons, pos_neurons = find_pre_pos(rad,cleft_num)  
    plt.imshow(raw_box[i,:,:], cmap=plt.cm.gray)
    return z_min, z_max  

def plot_neuron_size(cleft_num,neuron_num):
    rads = np.linspace(0,15,15)
    neuron_den = np.zeros((15))
    neuron_box, dist_cleft = find_dist(cleft_num)
    for i in range(0,15):
        neigh_ind = np.nonzero(dist_cleft<=rads[i])
        region_vol = float(len(neigh_ind[0]))
        neuron_den[i] = sum(1*(neuron_box[neigh_ind]==neuron_num))
    plt.plot(rads,neuron_den)
    #plt.axis([0, 15, 0, 1])
    plt.show()
    print neuron_den[14]/region_vol
