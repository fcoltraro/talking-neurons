import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import measurements,morphology
import scipy.stats as stats
import scipy.spatial.distance as dst

def find_dist(cleft_num,raw=False,syn_map=False,z_slice=False):
    ind = np.nonzero(labels==cleft_num)
    z_min = max(min(ind[0]) - 5,0); z_max = min(max(ind[0]) + 5,124)
    y_min = max(min(ind[1]) - 50,0); y_max = min(max(ind[1]) + 50,1249)
    x_min = max(min(ind[2]) - 50,0); x_max = min(max(ind[2]) + 50,1249)
    cleft_box = labels[z_min:z_max,y_min:y_max,x_min:x_max].copy()
    cleft_bol = 1 - 1*(cleft_box==cleft_num)
    dist_cleft = ndimage.distance_transform_edt(cleft_bol,sampling=[40,4,4])
    neuron_box = data_neurons[z_min:z_max,y_min:y_max,x_min:x_max].copy()
    
    if raw==True and syn_map==False and z_slice==False:
        raw_box = data_image[z_min:z_max,y_min:y_max,x_min:x_max].copy()
        return neuron_box, dist_cleft, raw_box
    
    elif syn_map==True and raw==False and z_slice==False:   
        synaptic_box = synaptic_map[z_min:z_max,y_min:y_max,x_min:x_max].copy()
        return neuron_box, dist_cleft, synaptic_box
    
    elif syn_map==True and raw==True and z_slice==False:  
        raw_box = data_image[z_min:z_max,y_min:y_max,x_min:x_max].copy()
        synaptic_box = synaptic_map[z_min:z_max,y_min:y_max,x_min:x_max].copy()
        return neuron_box, dist_cleft, raw_box, synaptic_box
    
    elif raw==True and syn_map==False and z_slice==True:
         raw_box = data_image[z_min:z_max,y_min:y_max,x_min:x_max].copy()
         return dist_cleft, raw_box, min(ind[0]) - z_min, max(ind[0]) - z_min
        
    else:    
        return neuron_box, dist_cleft
        
  
def features_neurons(cleft_num,rad):
    #data
    neuron_box, dist_cleft, raw_box = find_dist(cleft_num,raw=True)
    #region characteristics
    neigh_ind = np.nonzero(dist_cleft<=rad); neigh_vol = float(len(neigh_ind[0]))
    #cleft characteristics
    cleft_ind = np.nonzero(dist_cleft<=0.25); mass_cleft = float(len(cleft_ind[0])); 
    R_cleft = sum(np.asarray(cleft_ind),axis=1)/mass_cleft; den_cleft = mass_cleft/neigh_vol
    #neurons in the cleft
    neurons = np.unique(neuron_box[neigh_ind]); nmb_neurons = len(neurons)
    #features  
    R_coord_neurons = np.zeros((nmb_neurons,3))
    stats_neurons = np.zeros((nmb_neurons,13))
    geom_feat = np.zeros((nmb_neurons,3))
    
    for i in range(0,nmb_neurons):
        neuron_i = neurons[i]; neuron_ind = np.nonzero(neuron_box[neigh_ind]==neuron_i)
        new_ind = (neigh_ind[0][neuron_ind],neigh_ind[1][neuron_ind],neigh_ind[2][neuron_ind]) 
        mass_i = float(len(new_ind[0]))
        #density
        den_i = mass_i/neigh_vol        
        #contact region 
        contact_i = sum(1*(neuron_box[cleft_ind]==neuron_i))/mass_cleft; 
        #center of mass
        R_i = sum(np.asarray(new_ind),axis=1)/mass_i
        R_coord_neurons[i,:] = R_i - R_cleft
        #distance to the cleft
        dist_i = np.linalg.norm(R_i - R_cleft)
        #Intensities
        intensities_i = raw_box[new_ind]
        stats_neurons[i,:] = statistics(intensities_i)
        geom_feat[i,:] = np.array([den_i,dist_i,contact_i])
    return  [cleft_num,rad,den_cleft,neurons, R_coord_neurons, geom_feat,stats_neurons]
        
def statistics(intensities):
    mean = np.mean(intensities); med = np.median(intensities); mode = float(stats.mode(intensities)[0]);
    variance = np.var(intensities); std_dev = np.sqrt(variance); 
    min_int = min(intensities);  max_int = max(intensities); rang = max_int - min_int;
    q1 = np.percentile(intensities,25); q3 = np.percentile(intensities,75); IQR = q3 - q1;
    kurt = stats.kurtosis(intensities); skw = stats.skew(intensities);
    return mean,med,mode,variance,std_dev,min_int,max_int,rang,q1,q3,IQR,kurt,skw

mu, sigma = 0, 100; teta = np.random.normal(mu, sigma, 36)

def cost_function(features_neurons,Theta):
    cleft_num,rad,den_cleft,neurons, R_coord_neurons, geom_feat,stats_neurons = features_neurons[:]
    nmb_neurons = len(neurons); C = np.zeros((nmb_neurons,nmb_neurons))
    distances = dst.squareform(dst.pdist(R_coord_neurons,'euclidean'))
    angles = np.arccos((1 - dst.squareform(dst.pdist(R_coord_neurons,'cosine'))))
    for i in range(0,nmb_neurons):
        for j in range(0,nmb_neurons):
            if i!=j:
               x_ij = np.concatenate((np.array([1]),geom_feat[i,:],stats_neurons[i,:],np.array([distances[i,j],den_cleft,angles[i,j]]), geom_feat[j,:],stats_neurons[j,:]),axis=0)
               C[i,j] = np.dot(x_ij,Theta.T)
    return C, neurons
    
def ILPsolver(C, neuron_names):
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
    



         


         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

