# -*- coding:utf-8 -*-
import numpy as np
import pyNN.spiNNaker as spynnaker
import csv
import random

from generate_vr_response import generate_vr_response
from generate_spiking_time import generate_spiking_time
from analyse import *

# Testing module
from debug_module import * 

#
#  SpiNN-3 board configuration
#  @Before: Epochs              -> set to 100ms   
#           Spiking_time_space  -> shape: (50 dimensional 2d-array)
#           
#
#  Global parameters:

#  Number of cells 
NUM_PN_CELLS = 50
NUM_KC_CELLS = 2000

#  Connection configuration
WEIGHT_PN_KC = 5
DELAY_PN_KC  = 1.0
NEURON_PARAMS = {
                 'cm'        : 0.25,
                 'i_offset'  : 0.0,
                 'tau_m'     : 20.0,
                 'tau_refrac': 0.0,
                 'tau_syn_E' : 10.0,
                 'tau_syn_I' : 10.0,
                 'v_reset'   : -70.0,
                 'v_rest'    : -65.0,
                 'v_thresh'  : -64.0
                }

#  Other stuff
TIME_SLOT   = 100
DATA_AMOUNT = 100

def setupLayer_PN(time_space):
    '''
     PN ─┬─── pn_neuron_01
         ├─── pn_neuron_02
         ├─── pn_neuron_03
         ├─── ...
         └─── pn_neuron_100

     PN was used as input layer
    '''
    print(len(time_space))
    input_population = spynnaker.Population(NUM_PN_CELLS,
                                            spynnaker.SpikeSourceArray(spike_times=time_space),
                                            label='PN_population')
    return input_population

def setupLayer_KC():

    '''
                        ┌────── KC_cell_0001
                        ├────── KC_cell_0002       ┌──────> PN_cell_[i]
                KC ─────┼────── KC_cell_0003  <────┼──────> ...
                        ├────── ....               └──────> PN_cell_[k]
                        └────── KC_cell_2000
               1.Each KC neuron map to around ~20 PN_cells
                 which was chosen randomly from 100  of all
               2.By the property of SpiNNaker Board.
                 Each core contains MAX 256 neurons.
                 Hence 2000 KC_neurons will spreads to around ~10 cores
    '''
    kc_population = spynnaker.Population(NUM_KC_CELLS,
                                         spynnaker.IF_curr_exp,
                                         NEURON_PARAMS,
                                         label='KC_population')
    return kc_population

def setupProjection_PN_KC(pn_population,kc_population):

    connectionList = list()                                        # Connection list between PN and KC
    for each_kc_cell in xrange(NUM_KC_CELLS):

        count          = 6
        selectedCells = random.sample(xrange(NUM_PN_CELLS),count) 

        for each_pn_cell in selectedCells:
            single_coonection = (each_pn_cell,each_kc_cell)
            connectionList.append(single_coonection)

    pnkcProjection = spynnaker.Projection(pn_population,
                                          kc_population,
                                          spynnaker.FromListConnector(connectionList),
                                          spynnaker.StaticSynapse(weight=WEIGHT_PN_KC, delay=DELAY_PN_KC))
    return pnkcProjection


def readData():     # This function may be discarded since 
                    # data preparation will be integrate in
                    # file to avoid file reading overhead
    spikeLists= []
    c = open("InputSpikingTime.csv", "rb")
    read = csv.reader(c)
    for line in read:
        spikeLists.append(map(float, line))
    return spikeLists



def retrieve_data(spikeData_original):      # This function may be discarded 
                                            # use analyse.get_count instead
    '''
        This function obtain the original KC reaction to each graph
        @Return_val[0] : original   -> DATA_AMOUNT * KC_REACTIONS_TO_THIS_GRAPH
        @Return_val[1] : simplified -> DATA_AMOUNT * SIMPLIFIED_KC_REACTIONS
    '''
    original  = np.zeros((DATA_AMOUNT,NUM_KC_CELLS)) 
    simplified= np.zeros((DATA_AMOUNT,NUM_KC_CELLS)) 

    for graph_index in xrange(DATA_AMOUNT):
        _begin_time = graph_index*TIME_SLOT
        _end_time  = _begin_time+TIME_SLOT
        for neuron_index in xrange(NUM_KC_CELLS):
            graph_reaction = np.array(spikeData_original[neuron_index])
            count= ((graph_reaction>_begin_time)and(graph_reaction<_end_time)).sum()
            rate = float(count)/(float(TIME_SLOT)/1000)
            original[graph_index][neuron_index] = rate

        indices = np.argpartition(original[graph_index], 100)[:100]
        simplified[graph_index, :][indices] = original[graph_index, :][indices]
 
    return [original,simplified]

def save_data(src):   # This function may be discarded since all 
                      # reading process maybe integred together
                      # to avoid file reading time cost
    LEN     = 2000
    csvFile = open("src.csv", "w")
    writer  = csv.writer(csvFile)

    for neuron_index in xrange(LEN):
        writer.writerow(src[neuron_index])
    csvFile.close()


def mapping_process():

    ###################################
    SIM_TIME = TIME_SLOT*DATA_AMOUNT ##
    ###################################

    response_space = generate_vr_response()
    spiking_space  = generate_spiking_time(response_space) 

    spynnaker.setup(timestep=1)
    spynnaker.set_number_of_neurons_per_core(spynnaker.IF_curr_exp, 250)

    pn_population  = setupLayer_PN(spiking_space)
    kc_population  = setupLayer_KC()
    kc_population.record(["spikes"])

    pn_kc_projection  = setupProjection_PN_KC(pn_population,kc_population)
    spynnaker.run(SIM_TIME)

    neo = kc_population.get_data(variables=["spikes"])
    spikeData_original= neo.segments[0].spiketrains
    spynnaker.end()
    return spikeData_original

def analysing_process(spikeData_original):
    
    simplified = get_count(spikeData_original)
    NN_list = get_nearest_neighbor(simplified)
    return NN_list

if(__name__=='__main__'):

    spikeData_original = mapping_process()
    save_as_pickle(spikeData_original,"spike_data_original")
#    NN_list = analysing_process(spikeData_original)
