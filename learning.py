import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import measurements,morphology
import scipy.stats as stats
import operator
    

def features_matrix(rad,cleft_num):
    #len(Theta) = 1 +35 + 35 + (35:2) = 666 
    cleft_num,rad,den_cleft,neurons, R_coord_neurons, geom_feat,stats_neurons = features_neurons(rad,cleft_num)
    nmb_neurons = len(neurons); feat_matrix = np.zeros((nmb_neurons,nmb_neurons,666));
    distances = dst.squareform(dst.pdist(R_coord_neurons,'euclidean'))
    angles = np.arccos((1 - dst.squareform(dst.pdist(R_coord_neurons,'cosine'))))
    for i in range(0,nmb_neurons):
        for j in range(0,nmb_neurons):
            if i!=j:
               x_ij = np.concatenate((geom_feat[i,:],stats_neurons[i,:],np.array([distances[i,j],den_cleft,angles[i,j]]), geom_feat[j,:],stats_neurons[j,:]),axis=0)
               X_ij = np.concatenate((np.array([1]),x_ij,quadratic_feat(x_ij)),axis=0)
               feat_matrix[i,j,:] = X_ij
    return feat_matrix


def y_prime(rad,cleft_num):
    neuron_box, dist_cleft, synaptic_box = find_dist(cleft_num,syn_map=True)
    neigh_ind = np.nonzero(dist_cleft<=rad)
    neurons = np.unique(neuron_box[neigh_ind]); neurons_num = len(neurons)
    #Presynaptic
    pre = np.nonzero(synaptic_box[neigh_ind]==-1)
    pre_ind = (neigh_ind[0][pre],neigh_ind[1][pre],neigh_ind[2][pre])
    pre_loc = np.array(pre_ind).T
    nmb_pre = len(pre_ind[0])
    pre_neurons = neuron_box[pre_ind]; 
    #Possynaptic
    pos = np.nonzero(synaptic_box[neigh_ind]==1)
    pos_ind = (neigh_ind[0][pos],neigh_ind[1][pos],neigh_ind[2][pos])
    pos_loc = np.array(pos_ind).T
    nmb_pos = len(pos_ind[0])
    pos_neurons = neuron_box[pos_ind]; 
    #Pairs
    y_prime = np.zeros((neurons_num,neurons_num))
    if nmb_pre==0 or nmb_pos==0:
        return y_prime
    else:
        pre_loc[:,0] = pre_loc[:,0]*40.0; pos_loc[:,0] = pos_loc[:,0]*40.0
        pre_loc[:,1] = pre_loc[:,1]*4.0;  pos_loc[:,1] = pos_loc[:,1]*4.0
        pre_loc[:,2] = pre_loc[:,2]*4.0;  pos_loc[:,2] = pos_loc[:,2]*4.0
        
        for i in range(0,nmb_pre):
            presyn_i = pre_neurons[i]
            presyn_ind = np.nonzero(neurons==presyn_i)[0][0]
            loc_i = pre_loc[i,:]
            dist_i = np.sum((pos_loc - loc_i)**2,axis=1)
            min_index, min_value = min(enumerate(dist_i), key=operator.itemgetter(1))
            possyn_i = pos_neurons[min_index]
            possyn_ind = np.nonzero(neurons==possyn_i)[0][0]
            y_prime[presyn_ind,possyn_ind] = 1
            pos_loc[min_index] = np.inf
        return y_prime  
             
    
def l_func(cleft_num,yprime,Theta):
    #assume y_prime is a matrix
    Theta = np.array(Theta)
    features = all_features[cleft_num - 1]
    nmb_neurons = features.shape[0]
    C = np.dot(features,Theta.T)    
    if yprime.shape[0]!=nmb_neurons:
        raise error 
    yprime_norm2 = np.sum(yprime)
    C_dot_yprime = np.sum(np.multiply(C,yprime))
    l = 1 - 2*yprime; C_hat = l - C;
    #we are maximizing so we put a minus 
    solver = FastSolver(-C_hat)
    solution = solver.solve()
    sol_array = solution.toarray()
    y = 1*(sol_array>0)
    C_hat_dot_y = np.sum(np.multiply(C_hat,y))
    l_value = C_hat_dot_y + C_dot_yprime + yprime_norm2
    return l_value

    
def L(Theta):
    L = 0
    for cleft in range(1,num_training):
        yprime = all_yprimes[cleft - 1]
        L = L + l_func(cleft,yprime,Theta)
    return L     
                
def grad_l(cleft_num,yprime,Theta):
    Theta = np.array(Theta)
    features = all_features[cleft_num - 1]
    C = np.dot(features,Theta.T)    
    solver = FastSolver(C)
    solution = solver.solve()
    sol_array = solution.toarray()
    y = 1*(sol_array>0)
    num_param = features.shape[2]; grad = []; 
    for k in range(num_param):
        dCdk = features[:,:,k]
        dCdk_dot_yprime = np.sum(np.multiply(dCdk,yprime))
        dCdk_dot_y = np.sum(np.multiply(dCdk,y))
        dldk = dCdk_dot_yprime - dCdk_dot_y
        grad.append(dldk)
    return grad 
        
def grad_L(Theta):
    dL = np.zeros(len(Theta))
    for cleft in range(1,num_training):
        yprime = all_yprimes[cleft - 1]
        dL = dL + grad_l(cleft,yprime,Theta)
        #print cleft 
    return list(dL)           

    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            