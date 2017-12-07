# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

### Import modules
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from ncupy import lnp            # For lnp

def calculate_distance_matrix(pos):
    dist_mat = sp.spatial.distance.cdist(pos, pos, metric='euclidean')
    return dist_mat

def log_connection_probability(dist, max_p=0.5, window_d=0.4):
    buf = dist / window_d
    prob = np.log(max_p) - 0.5 * buf * buf
    return prob

def generate_connection_matrix(dist):
    log_prob = log_connection_probability(dist)
    prob = np.exp(log_prob)
    conn_mat = np.random.rand(prob.shape[0], prob.shape[1]) < prob
    for i in range(conn_mat.shape[0]):
        conn_mat[i,i] = False
    
    return conn_mat


def draw_network_configuration(P_neuron, conn_matrix):
    G=nx.DiGraph()
    G.add_nodes_from(range(P_neuron.shape[0]))
    list_edge = np.where(conn_matrix)
    for n in range(len(list_edge[0])):
        G.add_edge(list_edge[0][n],list_edge[1][n])
    nx.draw_networkx(G,P_neuron)
    plt.show()
    return

def generate_random_weight(conn_matrix, max_w = 10, ref_w = 20):
    conn_weight = conn_matrix * max_w * (2.0 * np.random.rand(conn_matrix.shape[0], conn_matrix.shape[0]) - 1.0)
    if ref_w > 0:
        ref_w = -ref_w
        
    for i in range(conn_matrix.shape[0]):
        conn_weight[i][i] = ref_w
    return conn_weight

def generate_spike_train_data(conn_weight, dt = 0.01, max_t = 60.0, tau = 0.1, basal_rate = 20.0):
    time_points = np.arange(0.0, max_t + 1e-10, dt)
    num_time_points = len(time_points)
    num_neurons = conn_weight.shape[0]
    data = np.zeros((num_time_points, num_neurons))
    x = np.zeros(num_neurons)
    for k in range(num_time_points):
        for i in range(num_neurons):
            rate = np.exp(np.dot(conn_weight[i],x)+np.log(basal_rate))
            prob = rate * dt
            if np.random.rand() < prob:
                data[k][i] = 1
            else:
                data[k][i] = 0
        x = np.exp(-dt/tau) * x + data[k]
    return data

def draw_spike_train_data(data):
    #num_time_points = data.shape[0]
    num_neurons = data.shape[1]
    for i in range(num_neurons):
        spike_times = np.where(data[:,i]==1)
        plt.vlines(spike_times, i-0.5, i+0.5)
    #for k in range(num_time_points):
    #    for i in range(num_neurons):
    #        plt.vlines(k,i-0.5,i+0.5)
    #plt.show()
    return
    
def run_simple_estimation(data, dt=0.01, tau=0.1):
    num_time_points = data.shape[0]
    num_neurons = data.shape[1]
    x = np.zeros((num_time_points, num_neurons))
    for k in range(num_time_points):
        if k > 0:
            x[k] = x[k-1]*np.exp(-dt/tau) + data[k-1]
    # Create the object of LNP class
    MODEL = lnp.LNP()
    
    # Change the options of the object
    MODEL.set_options(method='MLE2', delta_t=dt)
    
    # Run the fitting algorithm
    Y = np.reshape(data[:,3], (data.shape[0], 1))
    MODEL.fit(x, Y)
    return

def run_main_estimation(data, dist_mat, dt=0.01, tau=0.1):
    num_time_points = data.shape[0]
    num_neurons = data.shape[1]
    x = generate_explanatory_variables(data, dt, tau)
    #conn_matrix = (-np.eye(num_neurons) + np.random.rand(num_neurons, num_neurons)) < 0.5
    MODEL = lnp.LNP()
    MODEL.set_options(method='MLE2', delta_t=dt, lmbd=1e-2)

    i=0
    conn_vector = np.random.rand(num_neurons) < 0.5
    conn_vector[i] = True
    
    Y = data[:,[i]]  #np.reshape(data[:,i], (data.shape[0], 1))
    idx = np.where(conn_vector)
    X = x[:,idx[0]]
    MODEL.fit(X, Y)
    weight, llh, pnlt = MODEL.get_fit_result()
    log_post = compute_log_posterior(i, conn_vector, dist_mat[i], llh, pnlt, num_time_points)
    
    conn_vector_best = np.copy(conn_vector)
    log_post_best = log_post
    
    for k in range(10000):
        for j in range(num_neurons):
            if j != i:
                conn_vector_tmp = np.copy(conn_vector)
                conn_vector_tmp[j] = not conn_vector_tmp[j]
                if(np.sum(conn_vector_tmp) > 1):
                    idx = np.where(conn_vector_tmp)
                    X = x[:,idx[0]]
                    MODEL.fit(X, Y)
                    weight, llh, pnlt = MODEL.get_fit_result()
                    log_post_tmp = compute_log_posterior(i, conn_vector, dist_mat[i], llh, pnlt, num_time_points)
                    print(conn_vector_tmp)
                    print(log_post_tmp)
                    print(conn_vector)
                    print(log_post)
                    if log_post_tmp > log_post:
                        conn_vector = np.copy(conn_vector_tmp)
                        log_post = log_post_tmp
                    elif np.random.rand() < np.exp(log_post_tmp-log_post):
                        conn_vector = np.copy(conn_vector_tmp)
                        log_post = log_post_tmp                    
                    if log_post_tmp > log_post_best:
                        conn_vector_best = np.copy(conn_vector_tmp)
                        log_post_best = log_post_tmp
            print(conn_vector_best)
            print(log_post_best)
            
    #print(weight)
    #print(llh/X.shape[0])
    #print(pnlt)
    #print(x[:,idx])
#    for j in range(num_neurons):
#        if j != i:
#            conn_matrix[i,j] = not conn_matrix[i,j]
#            print(conn_matrix)
#            print(j)
#    conn_matrix = conn_vector
    return conn_vector_best

def compute_log_posterior(i, conn_vec, dist_vec, llh, pnlt, num_data):
    log_posterior = llh - num_data * pnlt
    num_neurons = conn_vec.size
    for j in range(num_neurons):
        if j != i:
            if conn_vec[j]:
                log_prob = log_connection_probability(dist_vec[j])                
            else:
                log_prob = np.log(1.0 - np.exp(log_connection_probability(dist_vec[j])))
            log_posterior = log_posterior + log_prob
    return log_posterior
                

def generate_explanatory_variables(data, dt, tau):
    num_time_points = data.shape[0]
    num_neurons = data.shape[1]
    x = np.zeros((num_time_points, num_neurons))
    for k in range(num_time_points):
        if k > 0:
            x[k] = x[k-1]*np.exp(-dt/tau) + data[k-1]
    return x

### Initialize the seed of RNG
np.random.seed(seed=2)

### Number of neuronss
N_neuron = 4

### Dimensionality of neuron placement
Dim_P = 2

### Set the placement of neurons randomly
P_neuron = 2*np.random.rand(N_neuron, Dim_P)-1

### Distance matrix
D_matrix = calculate_distance_matrix(P_neuron)

Conn_matrix = generate_connection_matrix(D_matrix)

Conn_weight = generate_random_weight(Conn_matrix)

#draw_network_configuration(P_neuron, Conn_matrix)

data = generate_spike_train_data(Conn_weight)

#draw_spike_train_data(data)

#run_simple_estimation(data)

conn_mat_est = run_main_estimation(data, D_matrix)

#G=nx.Graph()
#G.add_node("nodeA")
#pos={}
#pos["nodeA"]=(0,0)
#nx.draw(G,pos)