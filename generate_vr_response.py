# 
# @Function: This function generate the response, the rate only.
#            Spiking time will be generated in the next step
#
# @Return: response_space : shape=(vr_count,graph_count)
#  Hence this function return the reaction of each vr to all graphs
#

import numpy as np

vr_space_dimension = 50
data_dimension = 784
mnistDir = "./data/mnist/mnist_test.txt"
vrDir = "./GNG-optimum-VR-set.csv"

def generate_vr_response():
    
    vr_space = np.loadtxt(vrDir,delimiter=',')
    feature_space = np.loadtxt(mnistDir,delimiter=',')

    vr_space_length = len(vr_space)
    feature_space_length = len(feature_space)
    distance_space = np.zeros((vr_space_length,feature_space_length))
    response_space = np.zeros((vr_space_length,feature_space_length))

    for vr_index in xrange(vr_space_length):
        for graph_index in xrange(feature_space_length):
            distance_vr_graph = np.linalg.norm((vr_space[vr_index]-feature_space[graph_index]),ord=1)
            distance_space[vr_index][graph_index] = distance_vr_graph
    
    
    average_distance_space = np.mean(distance_space,axis=1) 
    for vr_index in xrange(vr_space_length):
        average_distance = average_distance_space[vr_index]
        for graph_index in xrange(feature_space_length):
            response_vr_graph = np.exp(-((5 * distance_space[vr_index][graph_index])/average_distance)**0.7)
            response_space[vr_index][graph_index] = response_vr_graph
    return response_space
