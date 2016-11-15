import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import measurements,morphology
import scipy.stats as stats
import scipy.spatial.distance as dst
from sklearn import preprocessing

def find_cleft_ind(cleft_num):
    z_slices = len(labels_sparse)
    x = np.array([]); y = np.array([]); z = np.array([])
    for z_slc in range(z_slices):
        (y_coord,x_coord) = np.nonzero(labels_sparse[z_slc]==cleft_num)
        num_ind = len(y_coord)
        z_coord = z_slc*np.ones((num_ind))
        x = np.concatenate((x,x_coord),axis=0)
        y = np.concatenate((y,y_coord),axis=0)
        z = np.concatenate((z,z_coord),axis=0)
    return (z,y,x)

        
def find_dist(cleft_num,raw=False,syn_map=False,z_slice=False):
    ind = np.nonzero(labels==cleft_num)
    #ind = find_cleft_ind(cleft_num)
    z_min = int(max(min(ind[0]) - 3,0));  z_max = int(min(max(ind[0]) + 3,124))
    y_min = int(max(min(ind[1]) - 30,0)); y_max = int(min(max(ind[1]) + 30,1249))
    x_min = int(max(min(ind[2]) - 30,0)); x_max = int(min(max(ind[2]) + 30,1249))
    
    """
    cleft_box = np.zeros((z_max - z_min + 1,y_max - y_min + 1 ,x_max - x_min + 1))
    for z in range(z_min,z_max + 1):
        cleft_box[z-z_min,:,:] = labels_sparse[z][y_min:y_max+1,x_min:x_max+1].toarray()
    """
    cleft_box = labels[z_min:z_max,y_min:y_max,x_min:x_max]   
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
        
  
def features_neurons(rad,cleft_num):
    #data
    neuron_box, dist_cleft, raw_box = find_dist(cleft_num,raw=True)
    #region characteristics
    neigh_ind = np.nonzero(dist_cleft<=rad); neigh_vol = float(len(neigh_ind[0]))
    #cleft characteristics
    cleft_ind = np.nonzero(dist_cleft<=0.25); mass_cleft = float(len(cleft_ind[0])); 
    R_cleft = np.sum(np.asarray(cleft_ind),axis=1)/mass_cleft; den_cleft = mass_cleft/neigh_vol
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
        contact_i = np.sum(1*(neuron_box[cleft_ind]==neuron_i))/mass_cleft; 
        #center of mass
        R_i = np.sum(np.asarray(new_ind),axis=1)/mass_i
        R_coord_neurons[i,:] = R_i - R_cleft
        #distance to the cleft
        dist_i = np.linalg.norm(R_i - R_cleft)
        #Intensities
        intensities_i = raw_box[new_ind]
        stats_neurons[i,:] = statistics(intensities_i)
        geom_feat[i,:] = np.array([den_i,dist_i,contact_i])
    stats_neurons = preprocessing.scale(stats_neurons)
    return  [cleft_num,rad,den_cleft,neurons, R_coord_neurons, geom_feat,stats_neurons]
        
def statistics(intensities):
    mean = np.mean(intensities); med = np.median(intensities); mode = float(stats.mode(intensities)[0]);
    variance = np.var(intensities); std_dev = np.sqrt(variance); 
    min_int = min(intensities);  max_int = max(intensities); rang = max_int - min_int;
    q1 = np.percentile(intensities,25); q3 = np.percentile(intensities,75); IQR = q3 - q1;
    kurt = stats.kurtosis(intensities); skw = stats.skew(intensities);
    return mean,med,mode,variance,std_dev,min_int,max_int,rang,q1,q3,IQR,kurt,skw

mu, sigma = 0, 1; teta = np.random.normal(mu, sigma, 666)

def cost_function(features_neurons,Theta):
    #len(Theta) = 1 +35 + 35 + (35:2) = 666 
    cleft_num,rad,den_cleft,neurons, R_coord_neurons, geom_feat,stats_neurons = features_neurons[:]
    nmb_neurons = len(neurons); C = np.zeros((nmb_neurons,nmb_neurons))
    distances = dst.squareform(dst.pdist(R_coord_neurons,'euclidean'))
    angles = np.arccos((1 - dst.squareform(dst.pdist(R_coord_neurons,'cosine'))))
    for i in range(0,nmb_neurons):
        for j in range(0,nmb_neurons):
            if i!=j:
               x_ij = np.concatenate((geom_feat[i,:],stats_neurons[i,:],np.array([distances[i,j],den_cleft,angles[i,j]]), geom_feat[j,:],stats_neurons[j,:]),axis=0)
               X_ij = np.concatenate((np.array([1]),x_ij,quadratic_feat(x_ij)),axis=0)
               C[i,j] = np.dot(X_ij,Theta.T)
    return C, neurons
    
def quadratic_feat(x):
    n = len(x); pol2 = np.array([]);
    for i in range(n):
        for j in range(i,n):
            pol2 = np.concatenate((pol2,np.array([x[i]*x[j]])),axis=0)
    return pol2
    
def cubic_feat(x):
    n = len(x); pol3 = np.array([]);
    for i in range(n):
        for j in range(i,n):
            for k in range(j,n):
                pol3 = np.concatenate((pol3,np.array([x[i]*x[j]*x[k]])),axis=0)
    return pol3    
    
# C = np.array([[0,-5,-3,0,0],[0,0,0,0,-8],[0,0,0,-1.5,0],[5,2,-1,0,0],[-7,2,6,8,0]])

def ILPsolver(C, neuron_names):
    n = len(neuron_names)
    neu_neg = 1*(C<0)
    graph = C*neu_neg  
    solver = FastSolver(graph)
    solution = solver.solve()
    pairs = set({})
    for i in range(n):
        for j in range(n):
            if solution[i,j]>0:
                neu_i = neuron_names[i]; neu_j = neuron_names[j]; 
                pairs = pairs.union({(neu_i,neu_j)})
    return pairs 
    
def get_pred_pairs(rad,cleft_num,Theta):
    features = features_neurons(rad,cleft_num)
    C,neurons = cost_function(features,Theta)
    pairs = ILPsolver(C,neurons)
    return pairs
    
       


         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

